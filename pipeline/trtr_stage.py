"""
Stage: TRTR (Train Real, Test Real).

Trains the 4 benchmark models on the (patient-capped) real training set of each
CV fold and evaluates on the real validation set. Models are written to
``config['paths']['models_baseline_dir']`` using the canonical filename scheme.
"""

from pathlib import Path

from pipeline._common import (
    make_loader, prepare_fold, train_four_models, set_seeds, print_benchmark_summary,
)


def trtr_stage(config, output_dir=None):
    seed = set_seeds(config)
    model_dir = (config['paths'].get('models_baseline_dir')
                 or config['paths']['models_trtr_dir'])

    print(f"\n{'='*80}\nTRTR STAGE (train real / test real): {config['name']}\n{'='*80}")
    loader = make_loader(config)
    print(f"Dataset: {loader.X.shape[0]} spectra, {loader.X.shape[1]} features | "
          f"models -> {model_dir}")

    per_fold = []
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        print(f"\n--- {fold_name} ---")
        X_train, y_train, X_val, y_val = prepare_fold(fold_data, config, cap_train=True)
        accs = train_four_models(X_train, y_train, X_val, y_val, config, model_dir, fold_name, seed)
        per_fold.append(accs)

    print_benchmark_summary(per_fold, title="TRTR BENCHMARK COMPLETE")
    return {'model_dir': str(model_dir), 'per_fold_accuracies': per_fold}


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TRTR baseline training")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_201")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    trtr_stage(cfg, out_dir)
