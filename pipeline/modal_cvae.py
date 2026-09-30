"""
Modal GPU offload for the CVAE stage (local-first, CVAE-only).

Only CVAE *training + generation* runs remotely on a GPU. The resulting decoder
weights, encoder weights, training history and synthetic pools are written to a
Modal Volume and then downloaded back into the local ``results/{exp}/...`` dirs,
so every other stage (TRTR / TSTR / TSRTR / evaluation) runs locally exactly as
if the CVAE had been trained locally.

Usage
-----
  # via the suite / cvae stage
  python -m pipeline.run_suite --configs exp_201 --stages cvae --cvae-backend modal
  python -m pipeline.cvae_stage --config exp_201 --backend modal

  # direct (advisory CLI)
  modal run pipeline/modal_cvae.py::train_cvae_remote --config-name exp_201 --epochs 2

Notes
-----
* Targets the modern Modal API (App + image.add_local_dir + Volume). If your
  installed ``modal`` version differs, adjust the image/mount calls accordingly.
* TensorFlow's pip wheel bundles the CUDA runtime, so a plain pip install works
  on Modal's GPU workers.
* The large HDF5 dataset is uploaded to the Volume once (cached across runs);
  small code/config/split files are baked into the image.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "glioma-cvae"
VOLUME_NAME = "glioma-cvae-vol"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML / data
        "tensorflow[and-cuda]==2.20.0",
        "numpy", "pandas", "scipy", "h5py", "PyYAML",
        # sklearn ecosystem
        "scikit-learn", "joblib",
        # Classifiers
        "xgboost",
        # Raman / signal processing
        "ramanspy", "natsort",
        # Utilities
        "tqdm", "matplotlib", "seaborn",
        "statsmodels", "shap", "hdbscan",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(ROOT / "pipeline"), remote_path="/root/pipeline")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
    .add_local_file(str(ROOT / "data" / "splits" / "manual_splits.json"),
                    remote_path="/root/data/splits/manual_splits.json")
    .add_local_file(str(ROOT / "data" / "RamanFreqTable.csv"),
                    remote_path="/root/data/RamanFreqTable.csv")
)

app = modal.App(APP_NAME, image=image)

VOL_MOUNT = "/vol"
INPUTS_PREFIX = "inputs"      # /vol/inputs/<h5 filename>
OUTPUTS_PREFIX = "outputs"    # /vol/outputs/<relative results path>


def _effective_output_dirs(config, store_suffix):
    """
    The real training run passes ``store_suffix=None`` and gets the config's
    own paths, unchanged. A shuffled-label control run passes a suffix (e.g.
    'shuffled') so its weights land in a sibling directory instead of
    overwriting the real CVAE weights that Tier 3's frozen trunk and Tier 4's
    CbR both depend on -- used identically on the remote (write) side and the
    local (download) side so the two never disagree about where a run's
    files live.
    """
    models_dir = config["paths"]["models_cvae_dir"]
    synthetic_dir = config["paths"]["synthetic_dir"]
    if store_suffix:
        models_dir = f"{models_dir}_{store_suffix}"
        synthetic_dir = f"{synthetic_dir}_{store_suffix}"
    return models_dir, synthetic_dir


@app.function(gpu="T4", volumes={VOL_MOUNT: volume}, timeout=60 * 60 * 6)
def train_cvae_remote(config_name: str, epochs=None, h5_name: str = None,
                      permutation_seed: int = None, store_suffix: str = None):
    """GPU worker: train the CVAE for `config_name` and stage outputs to the Volume.

    ``permutation_seed``/``store_suffix``: see cvae_stage.run_local and
    _effective_output_dirs -- None/None for a real run (unchanged behaviour);
    both set together for Tier 4's shuffled-bag-label control."""
    import os
    import shutil

    os.chdir("/root")
    from src.utils.config_manager import ConfigManager
    from pipeline.cvae_stage import run_local

    config = ConfigManager.load_config(config_name)

    # Place the uploaded dataset where the loader expects it (config processed_file)
    processed_file = Path(config["paths"]["processed_file"])
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    src_h5 = Path(VOL_MOUNT) / INPUTS_PREFIX / (h5_name or processed_file.name)
    if not processed_file.exists():
        shutil.copy(src_h5, processed_file)

    models_dir, synthetic_dir = _effective_output_dirs(config, store_suffix)

    # Train + generate locally inside the container
    run_local(config, epochs=epochs, permutation_seed=permutation_seed,
             model_out_dir=models_dir, synthetic_out_dir=synthetic_dir)

    # Stage outputs onto the volume, preserving their relative paths
    out_root = Path(VOL_MOUNT) / OUTPUTS_PREFIX
    for rel in (models_dir, synthetic_dir):
        rel = Path(rel)
        if rel.exists():
            dst = out_root / rel
            dst.mkdir(parents=True, exist_ok=True)
            for f in rel.rglob("*"):
                if f.is_file():
                    target = dst / f.relative_to(rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, target)
    volume.commit()
    return {"models_cvae_dir": models_dir, "synthetic_dir": synthetic_dir}


def _upload_inputs(config):
    """Upload the experiment's HDF5 dataset to the Volume (idempotent)."""
    processed_file = Path(config["paths"]["processed_file"])
    remote = f"{INPUTS_PREFIX}/{processed_file.name}"
    existing = {e.path for e in volume.listdir(INPUTS_PREFIX)} if _safe_listdir(INPUTS_PREFIX) else set()
    if remote in existing:
        print(f"  [modal] input already on volume: {remote}")
        return processed_file.name
    print(f"  [modal] uploading {processed_file} -> {VOLUME_NAME}:{remote}")
    with volume.batch_upload() as batch:
        batch.put_file(str(processed_file), remote)
    return processed_file.name


def _safe_listdir(prefix):
    try:
        return list(volume.listdir(prefix))
    except Exception:
        return []


def _download_outputs(config, store_suffix=None):
    """Pull /vol/outputs/<rel> back into the local results dirs."""
    from pathlib import PurePosixPath
    for rel in _effective_output_dirs(config, store_suffix):
        # Normalise to forward-slash so it matches Modal volume paths (Linux)
        remote_prefix = f"{OUTPUTS_PREFIX}/{rel.replace(chr(92), '/')}"
        entries = _safe_listdir(remote_prefix)
        if not entries:
            print(f"  [modal] no outputs found at {remote_prefix} (skipping)")
            continue
        Path(rel).mkdir(parents=True, exist_ok=True)
        stack = list(entries)
        while stack:
            entry = stack.pop()
            etype = getattr(entry, "type", None)
            if str(etype).endswith("DIRECTORY") or etype == 2:
                stack.extend(_safe_listdir(entry.path))
                continue
            # entry.path is a POSIX path from the Modal volume (forward slashes).
            # Compute the path relative to the remote prefix, then map onto local rel.
            rel_part = PurePosixPath(entry.path).relative_to(remote_prefix)
            local_path = Path(rel) / Path(*rel_part.parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as fh:
                for chunk in volume.read_file(entry.path):
                    fh.write(chunk)
        print(f"  [modal] downloaded {remote_prefix} -> {rel}")


def run_on_modal(config_name, epochs=None, permutation_seed=None, store_suffix=None):
    """
    Driver invoked by cvae_stage(backend='modal'): upload dataset, run CVAE on a
    GPU, download weights + synthetic pools back to the local results dirs.

    ``permutation_seed``/``store_suffix``: both None for a real run (default,
    unchanged behaviour). Both set together to run Tier 4's shuffled-bag-label
    control on Modal GPU -- e.g. permutation_seed=123, store_suffix='shuffled'
    (matching the convention scripts/stage6_shuffled_label_control.py already
    uses for Tiers 1-3) writes to results/{exp}/models/cvae_shuffled/ and
    .../data/synthetic_shuffled/, never touching the real weights Tier 3's
    frozen trunk and Tier 4's CbR depend on.
    """
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    print(f"\n[modal] CVAE on GPU for {config_name}"
         + (f"  [shuffled-label control, seed={permutation_seed}]" if permutation_seed else ""))
    with app.run():
        h5_name = _upload_inputs(config)
        train_cvae_remote.remote(config_name, epochs=epochs, h5_name=h5_name,
                                 permutation_seed=permutation_seed, store_suffix=store_suffix)
        _download_outputs(config, store_suffix=store_suffix)
    print(f"[modal] CVAE artifacts synced locally for {config_name}")
    models_dir, synthetic_dir = _effective_output_dirs(config, store_suffix)
    return {"models_cvae_dir": models_dir, "synthetic_dir": synthetic_dir}
