"""
Modal GPU offload for the 1D-CNN step of pipeline/_common.py::train_four_models
ONLY -- Apollo/RandomForest/XGBoost always train locally (fast on CPU, no GPU
benefit), and were never the bottleneck. The CNN was: trained with up to 100
epochs (now configurable, see note below) and was measured taking several
minutes per fold locally during exp_601's first TSTR/TSRTR run, the dominant
per-fold wall-clock cost of the whole 4-model benchmark by a wide margin.

Enabled per config via ``models.cnn.backend: "modal"`` (default "local",
unchanged behaviour for every existing config that doesn't set this key).
``pipeline/_common.py::train_four_models`` reads this and calls
``run_cnn_on_modal`` below in place of the local ``train_cnn_only`` --
everything else about train_four_models (Apollo/RF/XGBoost, the returned
accuracy dict, the saved file layout) is identical either way.

Usage
-----
  # via config (the normal path) -- train_four_models dispatches automatically
  models:
    cnn:
      backend: "modal"

  # direct (advisory CLI, mirrors modal_cvae.py's own docstring)
  modal run pipeline/modal_cnn.py::train_cnn_remote --run-id test

Notes
-----
Unlike modal_cvae.py (whose remote worker re-derives its training data from
the config + a shared HDF5 dataset already on the Volume), the CNN's training
data varies by regimen (TRTR=real, TSTR=synthetic, TSRTR=a mix) and is only
available as the in-memory arrays train_four_models was already called with
-- so this module uploads THOSE specific arrays (as .npy) to the Volume per
call, rather than re-deriving them remotely. Each call gets its own
uuid-named prefix and the uploaded inputs are deleted after a successful
download (nothing accumulates on the Volume across calls, unlike
modal_cvae.py's intentionally-cached dataset/resume-checkpoint uploads).
"""
import shutil
import tempfile
import uuid
from pathlib import Path

import modal

from pipeline.modal_cvae import image  # same deps/image definition as the CVAE offload

APP_NAME = "glioma-cnn"
VOLUME_NAME = "glioma-cnn-vol"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

VOL_MOUNT = "/vol"


@app.function(gpu="T4", volumes={VOL_MOUNT: volume}, timeout=60 * 120)
def train_cnn_remote(run_id: str, cnn_cfg: dict, aug_cfg: dict, seed: int, has_bag_ids: bool):
    """GPU worker: download this call's uploaded arrays, train the CNN, upload
    the resulting .keras file + accuracy back to the Volume."""
    import json
    import os
    import time

    import numpy as np
    import tensorflow as tf

    os.chdir("/root")
    from pipeline._common import train_cnn_only

    # Without this, TF's BFC allocator aggressively probes for very large
    # candidate blocks on this T4 (observed: repeated "ran out of memory
    # trying to allocate 24GiB" warnings even for this small model/dataset) --
    # benign per TF's own log text, but noisy and unnecessary; growth mode
    # instead allocates only what's actually used.
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    in_dir = Path(VOL_MOUNT) / "inputs" / run_id
    X_train = np.load(in_dir / "X_train.npy")
    y_train = np.load(in_dir / "y_train.npy")
    X_val = np.load(in_dir / "X_val.npy")
    y_val = np.load(in_dir / "y_val.npy")
    train_bag_ids = np.load(in_dir / "train_bag_ids.npy", allow_pickle=True) if has_bag_ids else None

    print(f"        [modal] training started: X_train={X_train.shape} X_val={X_val.shape} "
         f"epochs={cnn_cfg['epochs']} patience={cnn_cfg['early_stopping_patience']}", flush=True)
    t0 = time.time()
    out_dir = Path("/root/output")
    acc = train_cnn_only(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg,
                        out_dir, "remote", seed, train_bag_ids, progress_every=5)
    print(f"        [modal] training finished in {time.time() - t0:.0f}s, acc={acc:.4f}", flush=True)

    out_prefix = f"outputs/{run_id}"
    with volume.batch_upload() as batch:
        batch.put_file(str(out_dir / "cnn_remote.keras"), f"{out_prefix}/cnn_remote.keras")
    (out_dir / "accuracy.json").write_text(json.dumps({"accuracy": acc}))
    with volume.batch_upload() as batch:
        batch.put_file(str(out_dir / "accuracy.json"), f"{out_prefix}/accuracy.json")
    volume.commit()
    return acc


def run_cnn_on_modal(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg, model_dir, fold_name,
                     seed=42, train_bag_ids=None, attempts=3, backoff_s=120):
    """
    Local-side driver, called from pipeline/_common.py::train_four_models in
    place of train_cnn_only when models.cnn.backend == 'modal'. Same return
    contract (float accuracy) and the same saved-file layout
    (model_dir/cnn_{fold_name}.keras) as the local path.

    Retries the whole call (fresh run_id each time) on any exception: a transient
    network failure (e.g. DNS ``getaddrinfo`` errors after the machine wakes from
    sleep) used to fail a fold's CNN outright, and on Windows the temp-dir cleanup
    then raised ``WinError 32`` that masked the real error. The underlying error
    is now printed on every failed attempt.
    """
    import time

    last = None
    for attempt in range(1, attempts + 1):
        try:
            return _run_cnn_on_modal_once(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg,
                                          model_dir, fold_name, seed, train_bag_ids)
        except Exception as e:  # noqa: BLE001 -- retry on anything, surface the cause
            last = e
            print(f"        [modal] CNN attempt {attempt}/{attempts} failed: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt < attempts:
                time.sleep(backoff_s * attempt)
    raise last


def submit_cnn_on_modal(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg, seed=42,
                        train_bag_ids=None):
    """
    Upload one fold's arrays and START its remote training WITHOUT waiting
    (``Function.spawn``). Must be called inside a ``with app.run():`` block that
    stays open until ``collect_cnn_from_modal`` has been called on the handle.

    Lets several folds train on Modal GPUs concurrently from one local process:
    the caller prepares fold k, submits it, frees the arrays, and moves on to
    fold k+1, so local memory holds one fold at a time (the dataset is large and
    each process that loads it costs ~2 GB), while the GPU jobs overlap.
    """
    import numpy as np

    run_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        np.save(tmp / "X_train.npy", X_train)
        np.save(tmp / "y_train.npy", y_train)
        np.save(tmp / "X_val.npy", X_val)
        np.save(tmp / "y_val.npy", y_val)
        has_bag_ids = train_bag_ids is not None
        if has_bag_ids:
            np.save(tmp / "train_bag_ids.npy", np.asarray(train_bag_ids))
        remote_prefix = f"inputs/{run_id}"
        with volume.batch_upload() as batch:
            for f in tmp.glob("*.npy"):
                batch.put_file(str(f), f"{remote_prefix}/{f.name}")

    call = train_cnn_remote.spawn(run_id, cnn_cfg, aug_cfg, seed, has_bag_ids)
    return {'run_id': run_id, 'has_bag_ids': has_bag_ids, 'call': call, 'acc': None,
            'cnn_cfg': cnn_cfg, 'aug_cfg': aug_cfg, 'seed': seed}


def collect_cnn_from_modal(handle, model_dir, fold_name, attempts=3, backoff_s=120):
    """
    Wait for a submitted job, download ``cnn_{fold_name}.keras`` into ``model_dir``,
    and clean this call's inputs/outputs off the Volume. On any failure the real
    error is printed and the step is retried: if the remote job had not finished,
    it is re-spawned from the SAME uploaded inputs (still on the Volume until
    success); if it had, only the download is retried.
    """
    import time

    run_id = handle['run_id']
    out_prefix = f"outputs/{run_id}"
    remote_prefix = f"inputs/{run_id}"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / f"cnn_{fold_name}.keras"

    for attempt in range(1, attempts + 1):
        try:
            if handle['acc'] is None:
                handle['acc'] = float(handle['call'].get())
            with open(dest, "wb") as fh:
                for chunk in volume.read_file(f"{out_prefix}/cnn_remote.keras"):
                    fh.write(chunk)
            break
        except Exception as e:  # noqa: BLE001 -- retry on anything, surface the cause
            print(f"        [modal] {fold_name} collect attempt {attempt}/{attempts} failed: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(backoff_s * attempt)
            if handle['acc'] is None:
                handle['call'] = train_cnn_remote.spawn(
                    run_id, handle['cnn_cfg'], handle['aug_cfg'], handle['seed'],
                    handle['has_bag_ids'])

    volume.remove_file(remote_prefix, recursive=True)
    volume.remove_file(out_prefix, recursive=True)
    print(f"        [modal] CNN trained remotely -> {dest}  (acc={handle['acc']:.4f})", flush=True)
    return handle['acc']


def _run_cnn_on_modal_once(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg, model_dir, fold_name,
                           seed=42, train_bag_ids=None):
    with app.run():
        handle = submit_cnn_on_modal(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg, seed,
                                     train_bag_ids)
        return collect_cnn_from_modal(handle, model_dir, fold_name, attempts=1)
