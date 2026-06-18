"""
Stage: TSRTR (Train Synthetic+Real, Test Real).

For each CV fold:
  1. Take the patient-capped REAL training set (n_real spectra).
  2. Generate exactly ``round(synth_ratio * n_real)`` synthetic spectra ON DEMAND
     from a trained CVAE decoder (reused across ratio experiments), class-balanced.
  3. Train the 4 benchmark models on the shuffled real+synth mix; evaluate on the
     real validation set.

Because synthetic spectra are generated on demand from the saved decoder, any
ratio (incl. 1:2) is produced exactly — there is no fixed-pool ceiling.

Config keys (under ``tsrtr:``):
  synth_ratio:       synthetic-to-real ratio (e.g. 0.5, 1.0, 2.0)
  cvae_weights_dir:  directory holding decoder_{fold}.weights.h5
                     (defaults to config['paths']['models_cvae_dir'])
"""

from pathlib import Path

import numpy as np

from pipeline._common import (
    make_loader, prepare_fold, train_four_models, set_seeds, num_classes_of,
    print_benchmark_summary,
)
from pipeline._cvae_gen import build_cvae_config, load_decoder, generate_synthetic


def tsrtr_stage(config, output_dir=None):
    seed = set_seeds(config)
    num_classes = num_classes_of(config)

    tsrtr_cfg = config.get('tsrtr', {})
    synth_ratio = float(tsrtr_cfg.get('synth_ratio', 1.0))
    weights_dir = tsrtr_cfg.get('cvae_weights_dir') or config['paths']['models_cvae_dir']
    model_dir = config['paths']['models_tsrtr_dir']
    save_mix = config['paths'].get('synthetic_dir')  # optional provenance copy

    print(f"\n{'='*80}\nTSRTR STAGE (train real+synth / test real): {config['name']}\n{'='*80}")
    print(f"synth_ratio={synth_ratio} | decoder weights <- {weights_dir} | models -> {model_dir}")

    loader = make_loader(config)
    n_features = loader.X.shape[1]
    cvae_config = build_cvae_config(config, n_features)
    latent_dim = cvae_config['latent_dim']

    per_fold = []
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        print(f"\n--- {fold_name} ---")

        # 1. Patient-capped real training data + val
        X_real, y_real, X_val, y_val = prepare_fold(fold_data, config, cap_train=True)
        n_real = X_real.shape[0]

        # 2. On-demand synthetic generation at the target ratio
        n_synth = int(round(synth_ratio * n_real))
        n_per_class = n_synth // num_classes
        print(f"  n_real={n_real} -> target n_synth={n_synth} ({n_per_class}/class x {num_classes})")

        decoder = load_decoder(cvae_config, weights_dir, fold_name)
        X_synth, y_synth = generate_synthetic(decoder, n_per_class, num_classes, latent_dim)
        print(f"  Generated synthetic: {X_synth.shape}")

        if save_mix:
            Path(save_mix).mkdir(parents=True, exist_ok=True)
            np.save(Path(save_mix) / f"{fold_name}_X_synth_r{synth_ratio}.npy", X_synth)
            np.save(Path(save_mix) / f"{fold_name}_y_synth_r{synth_ratio}.npy", y_synth)

        # 3. Mix, shuffle, train
        X_mix = np.vstack([X_real, X_synth])
        y_mix = np.concatenate([y_real, y_synth])
        perm = np.random.permutation(len(y_mix))
        X_mix, y_mix = X_mix[perm], y_mix[perm]
        print(f"  Mixed real+synth: {X_mix.shape}  "
              f"(real={n_real}, synth={X_synth.shape[0]}, "
              f"actual ratio={X_synth.shape[0] / max(n_real, 1):.3f})")

        accs = train_four_models(X_mix, y_mix, X_val, y_val, config, model_dir, fold_name, seed)
        per_fold.append(accs)

    print_benchmark_summary(per_fold, title=f"TSRTR BENCHMARK COMPLETE (ratio {synth_ratio})")
    return {'model_dir': str(model_dir), 'synth_ratio': synth_ratio, 'per_fold_accuracies': per_fold}


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TSRTR training (train on real+synthetic)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_401")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    tsrtr_stage(cfg, out_dir)
