"""
Modal offload for classifier training stages (trtr / tstr / tsrtr).

Shares the same Modal Volume as modal_cvae.py so CVAE outputs (decoder weights,
synthetic pool) are available to classifier stages without re-uploading if the
CVAE stage already ran on Modal.

Primary entry point used by run_suite:
  run_stages_parallel(stage, [config_A, config_B, ...])
    -> uploads inputs for all configs, spawns all on Modal in parallel,
       waits for all to finish, downloads outputs.

Single-config path (for debugging or non-parallel use):
  run_stage_on_modal(config, stage)
"""

from pathlib import Path, PurePosixPath

import modal

ROOT           = Path(__file__).resolve().parents[1]
VOLUME_NAME    = "glioma-cvae-vol"   # shared with modal_cvae.py
APP_NAME       = "glioma-classifiers"
VOL_MOUNT      = "/vol"
INPUTS_PREFIX  = "inputs"
OUTPUTS_PREFIX = "outputs"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow[and-cuda]==2.20.0",
        "numpy", "pandas", "scipy", "h5py", "PyYAML",
        "scikit-learn", "joblib",
        "xgboost",
        "ramanspy", "natsort",
        "tqdm", "matplotlib", "seaborn",
        "statsmodels", "shap", "hdbscan",
    )
    .add_local_dir(str(ROOT / "src"),      remote_path="/root/src")
    .add_local_dir(str(ROOT / "pipeline"), remote_path="/root/pipeline")
    .add_local_dir(str(ROOT / "configs"),  remote_path="/root/configs")
    .add_local_file(str(ROOT / "data" / "splits" / "manual_splits.json"),
                    remote_path="/root/data/splits/manual_splits.json")
    .add_local_file(str(ROOT / "data" / "RamanFreqTable.csv"),
                    remote_path="/root/data/RamanFreqTable.csv")
)

app = modal.App(APP_NAME, image=image)


# ── volume helpers ─────────────────────────────────────────────────────────────

def _safe_listdir(prefix: str) -> list:
    try:
        return list(volume.listdir(prefix))
    except Exception:
        return []


def _vol_has_content(remote_prefix: str) -> bool:
    return bool(_safe_listdir(remote_prefix))


def _upload_dir_to_vol(local_dir: str, remote_prefix: str):
    """Upload every file under local_dir to remote_prefix on the volume."""
    local = Path(local_dir)
    if not local.exists():
        return
    with volume.batch_upload() as batch:
        for f in local.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local).as_posix()
                batch.put_file(str(f), f"{remote_prefix}/{rel}")
    print(f"  [modal] uploaded {local_dir} -> {VOLUME_NAME}:{remote_prefix}")


def _posix_remote(local_rel: str) -> str:
    """Convert a local (possibly Windows backslash) path to a POSIX volume key."""
    return local_rel.replace("\\", "/")


def _upload_h5(config) -> str:
    """Upload the experiment HDF5 to the volume if not already there."""
    h5 = Path(config["paths"]["processed_file"])
    remote = f"{INPUTS_PREFIX}/{h5.name}"
    existing = {e.path for e in _safe_listdir(INPUTS_PREFIX)}
    if remote not in existing:
        print(f"  [modal] uploading {h5.name} -> {VOLUME_NAME}:{remote}")
        with volume.batch_upload() as batch:
            batch.put_file(str(h5), remote)
    else:
        print(f"  [modal] h5 already on volume: {h5.name}")
    return h5.name


def _upload_stage_inputs(config, stage: str) -> str:
    """Upload everything `stage` needs and return the h5 filename."""
    h5_name = _upload_h5(config)

    if stage == "tstr":
        synth_dir = config["paths"]["synthetic_dir"]
        remote = f"{OUTPUTS_PREFIX}/{_posix_remote(synth_dir)}"
        if not _vol_has_content(remote):
            print(f"  [modal] uploading synthetic pool from {synth_dir}")
            _upload_dir_to_vol(synth_dir, remote)
        else:
            print(f"  [modal] synthetic pool already on volume")

    if stage == "tsrtr":
        cvae_dir = (config["paths"].get("models_cvae_dir")
                    or config.get("tsrtr", {}).get("cvae_weights_dir"))
        if cvae_dir:
            remote = f"{OUTPUTS_PREFIX}/{_posix_remote(cvae_dir)}"
            if not _vol_has_content(remote):
                print(f"  [modal] uploading CVAE weights from {cvae_dir}")
                _upload_dir_to_vol(cvae_dir, remote)
            else:
                print(f"  [modal] CVAE weights already on volume")

    return h5_name


def _download_stage_outputs(config, stage: str):
    """Pull model files from the volume back into local results dirs."""
    paths = config["paths"]
    output_rels = {
        "trtr":  [paths.get("models_baseline_dir") or paths.get("models_trtr_dir")],
        "tstr":  [paths.get("models_tstr_dir")],
        "tsrtr": [paths.get("models_tsrtr_dir")],
    }.get(stage, [])

    for rel in output_rels:
        if not rel:
            continue
        remote_prefix = f"{OUTPUTS_PREFIX}/{_posix_remote(rel)}"
        entries = _safe_listdir(remote_prefix)
        if not entries:
            print(f"  [modal] no outputs at {remote_prefix} (skipping)")
            continue
        Path(rel).mkdir(parents=True, exist_ok=True)
        stack = list(entries)
        while stack:
            entry = stack.pop()
            etype = getattr(entry, "type", None)
            if str(etype).endswith("DIRECTORY") or etype == 2:
                stack.extend(_safe_listdir(entry.path))
                continue
            rel_part = PurePosixPath(entry.path).relative_to(remote_prefix)
            local_path = Path(rel) / Path(*rel_part.parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as fh:
                for chunk in volume.read_file(entry.path):
                    fh.write(chunk)
        print(f"  [modal] downloaded {remote_prefix} -> {rel}")


# ── remote function ────────────────────────────────────────────────────────────

@app.function(cpu=4.0, memory=8192, volumes={VOL_MOUNT: volume}, timeout=60 * 60 * 4)
def train_stage_remote(config_name: str, stage: str, h5_name: str = None):
    """Run one classifier stage (trtr / tstr / tsrtr) for one experiment on Modal."""
    import os
    import shutil
    os.chdir("/root")

    from src.utils.config_manager import ConfigManager
    config = ConfigManager.load_config(config_name)
    paths  = config["paths"]
    vol_out = f"{VOL_MOUNT}/{OUTPUTS_PREFIX}"

    # ── copy HDF5 into place ──────────────────────────────────────────────────
    processed_file = f"{paths['processed_file']}"
    import pathlib
    pf = pathlib.Path(processed_file)
    pf.parent.mkdir(parents=True, exist_ok=True)
    src_h5 = pathlib.Path(VOL_MOUNT) / INPUTS_PREFIX / (h5_name or pf.name)
    if not pf.exists():
        shutil.copy(str(src_h5), str(pf))

    # ── copy stage prerequisites from volume ──────────────────────────────────
    if stage == "tstr":
        synth_dir = pathlib.Path(paths["synthetic_dir"])
        vol_synth  = pathlib.Path(vol_out) / synth_dir.as_posix()
        if vol_synth.exists() and not synth_dir.exists():
            shutil.copytree(str(vol_synth), str(synth_dir))

    if stage == "tsrtr":
        cvae_dir = (paths.get("models_cvae_dir")
                    or config.get("tsrtr", {}).get("cvae_weights_dir"))
        if cvae_dir:
            cvae_path = pathlib.Path(cvae_dir)
            vol_cvae   = pathlib.Path(vol_out) / cvae_path.as_posix()
            if vol_cvae.exists() and not cvae_path.exists():
                shutil.copytree(str(vol_cvae), str(cvae_path))

    # ── run the stage ─────────────────────────────────────────────────────────
    print(f"\n[modal:{config_name}] running {stage}")
    if stage == "trtr":
        from pipeline.trtr_stage import trtr_stage
        trtr_stage(config, None)
    elif stage == "tstr":
        from pipeline.tstr_stage import tstr_stage
        tstr_stage(config, None)
    elif stage == "tsrtr":
        from pipeline.tsrtr_stage import tsrtr_stage
        tsrtr_stage(config, None)

    # ── stage outputs to volume ───────────────────────────────────────────────
    output_rels = {
        "trtr":  [paths.get("models_baseline_dir") or paths.get("models_trtr_dir")],
        "tstr":  [paths.get("models_tstr_dir")],
        "tsrtr": [paths.get("models_tsrtr_dir")],
    }.get(stage, [])

    vol_out_path = pathlib.Path(vol_out)
    for rel in output_rels:
        if not rel:
            continue
        src = pathlib.Path(rel)
        if not src.exists():
            continue
        dst = vol_out_path / src.as_posix()
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                t = dst / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(f), str(t))
    volume.commit()
    print(f"[modal:{config_name}:{stage}] done, outputs staged to volume")


# ── driver functions (called from run_suite) ───────────────────────────────────

def run_stages_parallel(stage: str, configs: list):
    """
    Dispatch `stage` for all configs in parallel on Modal, then download outputs.

    Upload phase runs locally (before app.run), so volume I/O is serial but fast.
    Execution phase spawns one container per config — they run concurrently.
    """
    print(f"\n[modal] uploading inputs for {stage}: {[c['name'] for c in configs]}")
    h5_names = {c["name"]: _upload_stage_inputs(c, stage) for c in configs}

    print(f"[modal] spawning {len(configs)} parallel {stage} container(s)...")
    with app.run():
        handles = {
            c["name"]: train_stage_remote.spawn(
                c["name"], stage, h5_name=h5_names[c["name"]]
            )
            for c in configs
        }
        for cfg_name, handle in handles.items():
            print(f"  [modal] waiting for {cfg_name}:{stage}...")
            handle.get()
            print(f"  [modal] {cfg_name}:{stage} complete")

    print(f"[modal] downloading outputs for {stage}...")
    for c in configs:
        _download_stage_outputs(c, stage)
    print(f"[modal] all {stage} stages synced locally")


def run_stage_on_modal(config, stage: str):
    """Single-config convenience wrapper (upload → run → download)."""
    print(f"\n[modal] {config['name']}:{stage}")
    h5_name = _upload_stage_inputs(config, stage)
    with app.run():
        train_stage_remote.remote(config["name"], stage, h5_name=h5_name)
    _download_stage_outputs(config, stage)
    print(f"[modal] {config['name']}:{stage} artifacts synced locally")
