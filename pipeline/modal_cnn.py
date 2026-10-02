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
                     seed=42, train_bag_ids=None):
    """
    Local-side driver, called from pipeline/_common.py::train_four_models in
    place of train_cnn_only when models.cnn.backend == 'modal'. Same return
    contract (float accuracy) and the same saved-file layout
    (model_dir/cnn_{fold_name}.keras) as the local path.
    """
    import json

    import numpy as np

    run_id = uuid.uuid4().hex
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
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

        with app.run():
            acc = train_cnn_remote.remote(run_id, cnn_cfg, aug_cfg, seed, has_bag_ids)

            out_prefix = f"outputs/{run_id}"
            dest = model_dir / f"cnn_{fold_name}.keras"
            with open(dest, "wb") as fh:
                for chunk in volume.read_file(f"{out_prefix}/cnn_remote.keras"):
                    fh.write(chunk)

        # Clean up this call's uploaded inputs/outputs -- nothing should
        # accumulate on the Volume across repeated calls. (No explicit
        # volume.commit() here: that's only valid on the in-container mounted
        # handle, which is what train_cnn_remote's own commit() call -- after
        # writing outputs -- already is. batch_upload()/remove_file() from
        # this local/client-side handle persist on their own.)
        volume.remove_file(remote_prefix, recursive=True)
        volume.remove_file(out_prefix, recursive=True)

    print(f"        [modal] CNN trained remotely -> {dest}  (acc={acc:.4f})")
    return float(acc)
