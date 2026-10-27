"""
Train the 1D-CNN of several (experiment, regimen, fold) jobs CONCURRENTLY on Modal
GPUs from ONE local process, leaving every other model untouched.

Why this exists: in pipeline/_common.py::train_four_models the CNN is the slow
step (~35 min per fold on a Modal T4) and runs strictly after the fold's local
Apollo/RF/XGBoost fits, one fold at a time. The Modal jobs are independent, so
they can all run at once -- but a process per fold is not an option on this
machine (each process that loads the dataset costs ~2 GB). Here the dataset is
loaded once; each job's arrays are prepared, uploaded and the remote job is
STARTED (``modal_cnn.submit_cnn_on_modal``, non-blocking), the local arrays are
freed, and the next job is prepared. Results are collected at the end
(``collect_cnn_from_modal``, retried on failure).

Data per regimen mirrors the stage that trained the other three models:
  tstr   pipeline/tstr_stage.py: the saved synthetic pool {synthetic_dir}/{fold}_X_synth.npy
         (identical to what the other models saw); validation = real fold, no bag ids.
  tsrtr  pipeline/tsrtr_stage.py: the saved ratio draw
         {synthetic_dir}/{fold}_X_synth_r{ratio}.npy (identical to what the other models
         saw) mixed with the patient-capped real training set. The cap draws from the
         global numpy RNG, whose state depends on everything consumed earlier in the
         original run, so it cannot be replayed: it is re-drawn here (seeded per fold,
         reproducible), same size and scheme, a different realization of the capped
         real subset.

Usage:
    python scripts/train_cnn_modal_parallel.py \
        --jobs exp_707:tsrtr:0,1,2,3 exp_708:tstr:0,1,2,3,4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import make_loader, prepare_fold, set_seeds  # noqa: E402
from pipeline.modal_cnn import app, collect_cnn_from_modal, submit_cnn_on_modal  # noqa: E402
from pipeline.tstr_stage import _load_synth_pool  # noqa: E402


def _parse_job(text):
    name, regimen, folds = text.split(':')
    assert regimen in ('tstr', 'tsrtr'), regimen
    return name, regimen, [int(f) for f in folds.split(',')]


def _prepare(config, regimen, fold_data, fold_idx, seed):
    """Return (X_train, y_train, X_val, y_val, bag_ids, model_dir) for one job."""
    fold_name = fold_data['fold_name']
    synth_dir = Path(config['paths']['synthetic_dir'])
    if regimen == 'tstr':
        fd = prepare_fold(fold_data, config, cap_train=False)
        X_train, y_train = _load_synth_pool(synth_dir, fold_name)
        return X_train, y_train, fd.X_val, fd.y_val, None, config['paths']['models_tstr_dir']

    ratio = float(config['tsrtr']['synth_ratio'])
    np.random.seed(seed + fold_idx)
    fd = prepare_fold(fold_data, config, cap_train=True)
    X_real, y_real = fd.X_train, fd.y_train
    n_real = X_real.shape[0]
    X_synth = np.load(synth_dir / f"{fold_name}_X_synth_r{ratio}.npy")
    y_synth = np.load(synth_dir / f"{fold_name}_y_synth_r{ratio}.npy")
    expected = int(round(ratio * n_real)) // 2 * 2
    assert len(y_synth) == expected, (
        f"{fold_name}: saved synthetic set has {len(y_synth)} rows, expected {expected} "
        f"(ratio {ratio} x n_real {n_real}) -- not the draw the other models used")
    bag_ids = None
    if fd.train_bag_ids is not None:
        bag_ids = np.concatenate([np.asarray(fd.train_bag_ids, dtype=object),
                                  np.array([None] * len(y_synth), dtype=object)])
    X = np.vstack([X_real, X_synth])
    y = np.concatenate([y_real, y_synth])
    perm = np.random.permutation(len(y))
    return (X[perm], y[perm], fd.X_val, fd.y_val,
            bag_ids[perm] if bag_ids is not None else None, config['paths']['models_tsrtr_dir'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', nargs='+', required=True, help="config:regimen:fold,fold,...")
    args = ap.parse_args()

    from src.utils.config_manager import ConfigManager
    jobs = [_parse_job(j) for j in args.jobs]
    configs = {name: ConfigManager.load_config(name).to_dict() for name, _, _ in jobs}
    first = configs[jobs[0][0]]
    for name, c in configs.items():
        assert c['paths']['processed_file'] == first['paths']['processed_file'], name
        assert c['paths']['splits_file'] == first['paths']['splits_file'], name
        assert c['models']['cnn'].get('backend') == 'modal', f'{name}: cnn backend is not modal'
    seed = set_seeds(first)

    loader = make_loader(first)
    pending = []
    with app.run():
        for fold_idx, fold_data in enumerate(loader):
            for name, regimen, folds in jobs:
                if fold_idx not in folds:
                    continue
                cfg = configs[name]
                X, y, Xv, yv, bag_ids, model_dir = _prepare(cfg, regimen, fold_data, fold_idx, seed)
                print(f"[{name} {regimen} {fold_data['fold_name']}] submitting: "
                      f"train={X.shape} val={Xv.shape}", flush=True)
                handle = submit_cnn_on_modal(X, y, Xv, yv, cfg['models']['cnn'],
                                             dict(cfg.get('augmentation', {})), seed, bag_ids)
                pending.append((name, regimen, fold_data['fold_name'], model_dir, handle))
                del X, y, Xv, yv, bag_ids
        print(f"\nall {len(pending)} jobs submitted; collecting...\n", flush=True)

        results = []
        for name, regimen, fold_name, model_dir, handle in pending:
            acc = collect_cnn_from_modal(handle, model_dir, fold_name)
            results.append((name, regimen, fold_name, acc))

    print("\nCNN val accuracy (instance level):")
    for name, regimen, fold_name, acc in results:
        print(f"  {name} {regimen:5s} {fold_name}: {acc:.4f}")


if __name__ == '__main__':
    main()
