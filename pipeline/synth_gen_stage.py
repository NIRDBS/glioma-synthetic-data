"""
Stage: synthetic-pool generation from an EXISTING trained decoder (no training).

``cvae_stage`` trains the CVAE and, as its last step, generates the fixed
per-fold TSTR pool (``samples_per_class`` per class). When an experiment only
varies the pool itself -- e.g. TSTR on 2x the synthetic data of exp_608 -- re-running
``cvae_stage`` would retrain the CVAE (and, pointed at a reused ``models_cvae_dir``,
overwrite its weights). This stage instead loads the saved
``decoder_{fold}.weights.h5`` from ``paths.models_cvae_dir`` and only generates:
``models.cvae.samples_per_class`` per class per fold, written to
``paths.synthetic_dir`` as ``{fold}_X_synth.npy`` / ``{fold}_y_synth.npy`` -- the
exact files ``tstr_stage`` reads.

Generation uses the same ``load_decoder`` / ``generate_synthetic`` pair as
``tsrtr_stage`` (on-demand ratio generation), so draws are independent across
experiments, not nested subsets of one another.
"""

from pathlib import Path

import numpy as np

from pipeline._common import make_loader, num_classes_of, set_seeds
from pipeline._cvae_gen import build_cvae_config, generate_synthetic, load_decoder


def synth_gen_stage(config, output_dir=None):
    set_seeds(config)
    num_classes = num_classes_of(config)
    weights_dir = config['paths']['models_cvae_dir']
    synth_dir = Path(config['paths']['synthetic_dir'])
    samples_per_class = int(config['models']['cvae']['samples_per_class'])
    n_folds = int(config.get('cv_strategy', {}).get('n_folds', 5))

    print(f"\n{'='*80}\nSYNTH_GEN STAGE (generate only, no training): {config['name']}\n{'='*80}")
    print(f"decoder weights <- {weights_dir} | pool -> {synth_dir} | "
          f"{samples_per_class}/class x {num_classes} classes per fold")

    loader = make_loader(config)
    n_features = loader.X.shape[1]
    cvae_config = build_cvae_config(config, n_features)
    latent_dim = cvae_config['latent_dim']

    synth_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_folds):
        fold_name = f"fold_{i}"
        out_x = synth_dir / f"{fold_name}_X_synth.npy"
        out_y = synth_dir / f"{fold_name}_y_synth.npy"
        if out_x.exists() and out_y.exists():
            print(f"  [skip] {fold_name}: pool already exists.")
            continue
        decoder = load_decoder(cvae_config, weights_dir, fold_name)
        X_synth, y_synth = generate_synthetic(decoder, samples_per_class, num_classes, latent_dim)
        np.save(out_x, X_synth)
        np.save(out_y, y_synth)
        print(f"  [saved] {fold_name}: X{X_synth.shape}  class counts={np.bincount(y_synth.astype(int)).tolist()}")

    return {'synthetic_dir': str(synth_dir), 'samples_per_class': samples_per_class}


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate a synthetic pool from a saved decoder")
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    synth_gen_stage(cfg, out_dir)
