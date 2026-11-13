"""
Suite orchestrator: run multiple experiments end-to-end from the command line.

Each experiment's stage sequence is read from its config's ``pipeline.stages``
(overridable with --stages). Stages run in order through a dispatch map; outputs
are written to the per-experiment results directories so artifacts can be reused
across experiments (e.g. exp_401/402/403 reuse exp_201's CVAE + TRTR).

Examples
--------
  # Full run for exp_201, then the three reuse-based ratio ablations
  python -m pipeline.run_suite --configs exp_201 exp_401 exp_402 exp_403

  # Only the ratio ablations (exp_201 already done), CVAE on Modal if needed
  python -m pipeline.run_suite --configs exp_401 exp_402 exp_403 --cvae-backend local

  # Smoke test: tiny CVAE, skip-if-exists disabled
  python -m pipeline.run_suite --configs exp_201 --stages cvae --epochs 2 --force
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ensure project root on path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STAGES     = ['trtr', 'cvae', 'tstr', 'tsrtr', 'stats_eval', 'mil_eval', 'mil_fit']
CLASSIFIER_STAGES  = {'trtr', 'tstr', 'tsrtr'}
STAGE_ALIASES = {
    'preprocess': 'preprocessing', 'preprocessing': 'preprocessing',
    'trtr': 'trtr', 'cvae': 'cvae', 'synth_gen': 'synth_gen', 'tstr': 'tstr', 'tsrtr': 'tsrtr',
    'eval': 'stats_eval', 'evaluation': 'stats_eval', 'stats_eval': 'stats_eval',
    'mil_eval': 'mil_eval', 'mil': 'mil_eval', 'tier0': 'mil_eval',
    'mil_fit': 'mil_fit', 'misvm': 'mil_fit',
}


def _configure_gpu_memory_growth():
    """Avoid TF grabbing all VRAM up front."""
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _n_folds(config):
    return int(config.get('cv_strategy', {}).get('n_folds', 5))


def _stage_outputs_exist(stage, config):
    """Heuristic skip-if-done check: all folds' primary artifact present."""
    n = _n_folds(config)
    paths = config['paths']
    fold_names = [f"fold_{i}" for i in range(n)]
    try:
        if stage == 'trtr':
            d = Path(paths.get('models_baseline_dir') or paths['models_trtr_dir'])
            return all((d / f"apollo_{fn}.joblib").exists() for fn in fold_names)
        if stage == 'tstr':
            d = Path(paths['models_tstr_dir'])
            return all((d / f"apollo_{fn}.joblib").exists() for fn in fold_names)
        if stage == 'tsrtr':
            d = Path(paths['models_tsrtr_dir'])
            return all((d / f"apollo_{fn}.joblib").exists() for fn in fold_names)
        if stage == 'cvae':
            md = Path(paths['models_cvae_dir'])
            sd = Path(paths['synthetic_dir'])
            return all((md / f"decoder_{fn}.weights.h5").exists()
                       and (sd / f"{fn}_X_synth.npy").exists() for fn in fold_names)
        if stage == 'synth_gen':
            sd = Path(paths['synthetic_dir'])
            return all((sd / f"{fn}_X_synth.npy").exists()
                       and (sd / f"{fn}_y_synth.npy").exists() for fn in fold_names)
        if stage == 'stats_eval':
            d = Path(paths['statistical_dir'])
            return d.exists() and any(d.glob('statistical_evaluation_*.json'))
        if stage == 'mil_eval':
            # Distinct filename prefix (mil_tier0_*) from stats_eval's
            # (statistical_evaluation_*) in the same statistical_dir -- each
            # stage's skip-check globs only its own prefix, so neither fires
            # spuriously off the other's output.
            d = Path(paths['statistical_dir'])
            return d.exists() and any(d.glob('mil_tier0_*.json'))
        if stage == 'mil_fit':
            store_path = Path(paths.get('experiment_root', '')) / 'latent' / 'latent_labels.h5'
            if not store_path.exists():
                return False
            import h5py
            with h5py.File(store_path, 'r') as f:
                return 'misvm' in f
    except Exception:
        return False
    return False


def _run_stage(stage, config, output_dir, cvae_backend, epochs):
    """Dispatch a single stage to its implementation."""
    from pipeline.trtr_stage import trtr_stage
    from pipeline.tstr_stage import tstr_stage
    from pipeline.tsrtr_stage import tsrtr_stage
    from pipeline.cvae_stage import cvae_stage
    from pipeline.evaluation_stage import evaluation_stage

    if stage == 'preprocessing':
        from pipeline.preprocessing_stage import preprocessing_stage
        return preprocessing_stage(config, output_dir)
    if stage == 'trtr':
        return trtr_stage(config, output_dir)
    if stage == 'cvae':
        return cvae_stage(config, output_dir, backend=cvae_backend, epochs=epochs)
    if stage == 'synth_gen':
        from pipeline.synth_gen_stage import synth_gen_stage
        return synth_gen_stage(config, output_dir)
    if stage == 'tstr':
        return tstr_stage(config, output_dir)
    if stage == 'tsrtr':
        return tsrtr_stage(config, output_dir)
    if stage == 'stats_eval':
        paths = config['paths']
        baseline_dir = paths.get('models_baseline_dir') or paths.get('models_trtr_dir')
        n_folds = _n_folds(config)
        fold_names = [f"fold_{i}" for i in range(n_folds)]

        def _has_models(d):
            if not d:
                return False
            p = Path(d)
            return p.exists() and any((p / f"apollo_{fn}.joblib").exists() for fn in fold_names)

        results = {}
        # Thesis 1: can synthetic replace real? (TRTR vs TSTR)
        tstr_dir = paths.get('models_tstr_dir')
        if _has_models(tstr_dir):
            results['tstr'] = evaluation_stage(
                config, output_dir,
                baseline_dir=baseline_dir, candidate_dir=tstr_dir,
                baseline_label='TRTR', candidate_label='TSTR',
                cbr=True,
            )
        # Thesis 2: does synthetic augmentation improve? (TRTR vs TSRTR)
        tsrtr_dir = paths.get('models_tsrtr_dir')
        if _has_models(tsrtr_dir):
            results['tsrtr'] = evaluation_stage(
                config, output_dir,
                baseline_dir=baseline_dir, candidate_dir=tsrtr_dir,
                baseline_label='TRTR', candidate_label='TSRTR',
                cbr=True,
            )
        if not results:
            print("  [stats_eval] No candidate model dirs found — skipping.")
        return results
    if stage == 'mil_eval':
        from pipeline.mil_eval_stage import mil_eval_stage

        paths = config['paths']
        n_folds = _n_folds(config)
        fold_names = [f"fold_{i}" for i in range(n_folds)]

        def _has_models(d):
            if not d:
                return False
            p = Path(d)
            return p.exists() and any((p / f"apollo_{fn}.joblib").exists() for fn in fold_names)

        results = {}
        for label, key in (('TRTR', 'models_baseline_dir'), ('TSTR', 'models_tstr_dir'),
                           ('TSRTR', 'models_tsrtr_dir')):
            d = paths.get(key) or (paths.get('models_trtr_dir') if label == 'TRTR' else None)
            if _has_models(d):
                results[label] = mil_eval_stage(config, output_dir, regimen_dir=d, regimen_label=label)
        if not results:
            print("  [mil_eval] No regimen model dirs found — skipping.")
        return results
    if stage == 'mil_fit':
        from pipeline.mil_fit_stage import mil_fit_stage
        return mil_fit_stage(config, output_dir)
    raise ValueError(f"Unknown stage: {stage}")


def _log_suite(config, record):
    """Append a stage record to results/{exp}/suite_log.json."""
    root = Path(config['paths'].get('experiment_root', f"results/{config['name']}"))
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / 'suite_log.json'
    log = []
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            log = []
    log.append(record)
    log_path.write_text(json.dumps(log, indent=2))


def _record(summary, config, stage, status, secs):
    """Append to summary list and write to suite_log.json."""
    from src.utils.experiment_utils import log_stage_completion
    summary.append((config['name'], stage, status, secs))
    _log_suite(config, {'stage': stage, 'status': status, 'seconds': round(secs, 1),
                        'timestamp': datetime.now().isoformat()})
    try:
        log_stage_completion(config['name'], stage, success=(status == 'ok'))
    except Exception:
        pass


def _run_one_stage(stage, config, output_dir, cvae_backend, epochs, summary, continue_on_error):
    """Run a single stage for a single config, recording outcome in summary."""
    t0 = time.time()
    try:
        _run_stage(stage, config, output_dir, cvae_backend, epochs)
        _record(summary, config, stage, 'ok', time.time() - t0)
    except Exception as e:
        _record(summary, config, stage, 'FAILED', time.time() - t0)
        print(f"\n!!! [{config['name']}:{stage}] FAILED: {e}")
        traceback.print_exc()
        if not continue_on_error:
            _print_summary(summary)
            raise


def run_suite(config_names, stages_override=None, cvae_backend='local',
              classifiers_backend='local', epochs=None,
              continue_on_error=False, force=False):
    from src.utils.experiment_utils import setup_experiment

    _configure_gpu_memory_growth()
    summary = []

    # Load all configs upfront so both execution modes can see the full picture.
    all_items = []
    for name in config_names:
        config, output_dir = setup_experiment(name)
        stages = stages_override or config.get('pipeline', {}).get('stages', DEFAULT_STAGES)
        stages = [STAGE_ALIASES.get(s, s) for s in stages]
        all_items.append({'name': name, 'config': config, 'output_dir': output_dir, 'stages': stages})

    if classifiers_backend == 'modal':
        _run_suite_parallel(all_items, summary, cvae_backend, epochs, continue_on_error, force)
    else:
        _run_suite_sequential(all_items, summary, cvae_backend, epochs, continue_on_error, force)

    _print_summary(summary)
    return summary


def _run_suite_sequential(all_items, summary, cvae_backend, epochs, continue_on_error, force):
    """Original config-first sequential loop (local classifiers)."""
    for item in all_items:
        config, output_dir = item['config'], item['output_dir']
        print(f"\n{'#'*80}\n# EXPERIMENT {item['name']}  |  stages: {item['stages']}\n{'#'*80}")
        for stage in item['stages']:
            if (not force) and _stage_outputs_exist(stage, config):
                print(f"\n[{item['name']}:{stage}] outputs exist — skipping")
                summary.append((item['name'], stage, 'skipped', 0.0))
                continue
            print(f"\n>>> [{item['name']}] STAGE: {stage}")
            _run_one_stage(stage, config, output_dir, cvae_backend, epochs,
                           summary, continue_on_error)


def _run_suite_parallel(all_items, summary, cvae_backend, epochs, continue_on_error, force):
    """
    Stage-first execution: all configs' instances of the same stage are batched
    and dispatched in parallel on Modal.  Non-classifier stages (cvae, stats_eval)
    still run locally and sequentially.

    Ordering guarantee: DEFAULT_STAGES order is respected across all configs, so
    e.g. exp_201 cvae always finishes before any config's tsrtr starts — which is
    required because exp_401/402/403 reuse exp_201's decoder weights.
    """
    from pipeline.modal_classifiers import run_stages_parallel

    # Walk stages in DEFAULT_STAGES order; skip any not appearing in any config.
    stage_universe = [s for s in DEFAULT_STAGES
                      if any(s in item['stages'] for item in all_items)]

    for stage in stage_universe:
        pending = [
            item for item in all_items
            if stage in item['stages']
            and (force or not _stage_outputs_exist(stage, item['config']))
        ]
        skipped = [
            item for item in all_items
            if stage in item['stages']
            and not force and _stage_outputs_exist(stage, item['config'])
        ]
        for item in skipped:
            print(f"\n[{item['name']}:{stage}] outputs exist — skipping")
            summary.append((item['name'], stage, 'skipped', 0.0))

        if not pending:
            continue

        if stage in CLASSIFIER_STAGES:
            print(f"\n{'#'*80}")
            print(f"# MODAL PARALLEL  stage={stage}  "
                  f"configs={[i['name'] for i in pending]}")
            print(f"{'#'*80}")
            t0 = time.time()
            try:
                run_stages_parallel(stage, [i['config'] for i in pending])
                secs = time.time() - t0
                for item in pending:
                    _record(summary, item['config'], stage, 'ok', secs)
            except Exception as e:
                secs = time.time() - t0
                for item in pending:
                    _record(summary, item['config'], stage, 'FAILED', secs)
                print(f"\n!!! [modal:{stage}] FAILED: {e}")
                traceback.print_exc()
                if not continue_on_error:
                    _print_summary(summary)
                    raise

        else:
            # cvae and stats_eval: run locally, one config at a time.
            for item in pending:
                print(f"\n>>> [{item['name']}] STAGE: {stage}")
                _run_one_stage(stage, item['config'], item['output_dir'],
                               cvae_backend, epochs, summary, continue_on_error)


def _print_summary(summary):
    print(f"\n{'='*80}\nSUITE SUMMARY\n{'='*80}")
    print(f"{'experiment':<14}{'stage':<14}{'status':<10}{'seconds':>10}")
    print("-" * 48)
    for cfg, stage, status, secs in summary:
        print(f"{cfg:<14}{stage:<14}{status:<10}{secs:>10.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run a suite of experiments end-to-end")
    parser.add_argument('--configs', nargs='+', required=True,
                        help="experiment config names, e.g. exp_201 exp_401 exp_402 exp_403")
    parser.add_argument('--stages', nargs='+', default=None,
                        help="override the stages to run (default: each config's pipeline.stages)")
    parser.add_argument('--cvae-backend', default='local', choices=['local', 'modal'],
                        help="where to run CVAE training (default: local)")
    parser.add_argument('--classifiers-backend', default='local', choices=['local', 'modal'],
                        help="where to run classifier training — 'modal' dispatches all configs' "
                             "instances of the same stage in parallel on Modal (default: local)")
    parser.add_argument('--epochs', type=int, default=None, help="override CVAE epochs (smoke tests)")
    parser.add_argument('--continue-on-error', action='store_true',
                        help="keep going to the next stage/experiment on failure")
    parser.add_argument('--force', action='store_true', help="rerun stages even if outputs exist")
    args = parser.parse_args()

    run_suite(
        args.configs,
        stages_override=args.stages,
        cvae_backend=args.cvae_backend,
        classifiers_backend=args.classifiers_backend,
        epochs=args.epochs,
        continue_on_error=args.continue_on_error,
        force=args.force,
    )


if __name__ == '__main__':
    main()
