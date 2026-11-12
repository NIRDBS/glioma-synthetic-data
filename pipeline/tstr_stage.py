"""
Stage: TSTR (Train Synthetic, Test Real).

Trains the 4 benchmark models on the CVAE-generated synthetic pool for each fold
(``{synthetic_dir}/{fold}_X_synth.npy`` / ``_y_synth.npy``) and evaluates on the
real validation set of that fold. Models are written to
``config['paths']['models_tstr_dir']``.
"""

from pathlib import Path

import numpy as np

from pipeline._common import (
    make_loader, prepare_fold, train_four_models, set_seeds, print_benchmark_summary,
)


def _load_synth_pool(synth_dir, fold_name):
    X = np.load(Path(synth_dir) / f"{fold_name}_X_synth.npy").astype(np.float32)
    y = np.load(Path(synth_dir) / f"{fold_name}_y_synth.npy")
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    valid = ~np.isnan(X).any(axis=1)
    n_removed = int((~valid).sum())
    if n_removed:
        print(f"  Removed {n_removed} synthetic samples with NaNs")
    return X[valid], y[valid]


def tstr_stage(config, output_dir=None):
    seed = set_seeds(config)
    model_dir = config['paths']['models_tstr_dir']
    synth_dir = config['paths']['synthetic_dir']

    print(f"\n{'='*80}\nTSTR STAGE (train synthetic / test real): {config['name']}\n{'='*80}")
    loader = make_loader(config)
    print(f"Synthetic pool: {synth_dir} | models -> {model_dir}")

    per_fold = []
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        print(f"\n--- {fold_name} ---")
        # Real fold only needed for the (binary-mapped) validation set.
        _, _, X_val, y_val = prepare_fold(fold_data, config, cap_train=False)
        X_train, y_train = _load_synth_pool(synth_dir, fold_name)
        print(f"  Synthetic train: {X_train.shape}")
        accs = train_four_models(X_train, y_train, X_val, y_val, config, model_dir, fold_name, seed)
        per_fold.append(accs)

    print_benchmark_summary(per_fold, title="TSTR BENCHMARK COMPLETE")
    return {'model_dir': str(model_dir), 'per_fold_accuracies': per_fold}


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TSTR training (train on synthetic)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_201")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    tstr_stage(cfg, out_dir)
