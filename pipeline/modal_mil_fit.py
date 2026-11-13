"""
Modal offload for the MIL fit stage (mi-SVM) -- parallel across FOLDS,
not across configs.

Unlike modal_classifiers.py (which parallelizes the SAME stage across several
experiment configs), each (method, fold) pair here is an independent unit of
work for ONE config: mi-SVM fits fold f's training bags only and never shares
state across folds. This machine's free RAM does not cover 5 folds' worth of
independent data copies running at once locally (the RAM check that
motivated this file) -- Modal gives each fold its own
isolated container instead, at the same CPU-only cost profile as
modal_classifiers.py (this is sklearn/RF/SVM work throughout; no GPU need).

Primary entry point used by mil_fit_stage(backend='modal'):
  run_mil_fit_parallel(config, methods=('misvm',), ...)
    -> uploads the H5 once, spawns one container per (method, fold) pair,
       waits for all, writes every result into the LOCAL latent-label store
       (never written to remotely -- HDF5 concurrent writes across
       containers would be its own hazard; each container only returns its
       posterior arrays over Modal's RPC, small enough to be cheap: a
       (n_instances, K+1) float32 array is a few MB).
"""

from pathlib import Path

import modal

ROOT           = Path(__file__).resolve().parents[1]
VOLUME_NAME    = "glioma-cvae-vol"   # shared with modal_cvae.py / modal_classifiers.py
APP_NAME       = "glioma-mil-fit"
VOL_MOUNT      = "/vol"
INPUTS_PREFIX  = "inputs"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # mil_fit_stage.py itself only needs numpy/scipy/h5py/sklearn, but
        # importing anything under pipeline.* eagerly runs pipeline/__init__.py,
        # which unconditionally imports every stage module (preprocessing,
        # trtr, cvae, tstr, tsrtr, evaluation) -- so this needs their combined
        # transitive dependencies too. Matched to modal_classifiers.py's
        # already-working list rather than trimmed, after a first attempt
        # with a shorter list failed on a missing 'tqdm' inside
        # src/data/preprocessing.py, pulled in via preprocessing_stage.
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


def _safe_listdir(prefix: str) -> list:
    try:
        return list(volume.listdir(prefix))
    except Exception:
        return []


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


# Each fold's fit is CPU-only sklearn/RF/SVM work (Apollo's RF+SVM inside
# mi-SVM's iterations) -- 3 cores per container
# leaves headroom to run several folds concurrently without the containers
# fighting each other for a shared pool the way local multiprocessing would
# on this machine's 16 logical processors.
@app.function(cpu=3.0, memory=4096, volumes={VOL_MOUNT: volume}, timeout=60 * 60 * 3)
def fit_fold_remote(config_name: str, method: str, fold_name: str, h5_name: str,
                    pi_min: float, max_iters: int, seed: int,
                    svm_subsample_per_bag: int = 500, label_map: dict | None = None):
    """Fit one MIL formulation on one fold, remotely. Returns
    (fold_name, variant, posterior_full, in_sample_full) -- no store writes
    happen here; the local driver collects every result and writes them.

    ``label_map`` (optional): {bag_id: permuted_label}, computed ONCE by the
    caller (src/mil/shuffle_control.py::permuted_bag_label_map) and passed
    in explicitly rather than recomputed remotely -- avoids any risk of
    local/remote RNG divergence producing a different permutation per
    container. When given, substitutes the loader's own y_class before fold
    splitting, exactly mirroring scripts/shuffled_label_control.py's
    local path (the only "shuffled" intervention; everything downstream is
    identical to a real fit) -- this is what lets the shuffled-label
    control run fold-parallel on Modal instead of ~4h sequential locally."""
    import os
    import pathlib
    import shutil
    os.chdir("/root")

    from src.utils.config_manager import ConfigManager
    from pipeline._common import make_loader, num_classes_of
    from pipeline.mil_fit_stage import _fit_one_fold

    config = ConfigManager.load_config(config_name)
    paths = config["paths"]

    processed_file = pathlib.Path(paths["processed_file"])
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    src_h5 = pathlib.Path(VOL_MOUNT) / INPUTS_PREFIX / (h5_name or processed_file.name)
    if not processed_file.exists():
        shutil.copy(str(src_h5), str(processed_file))

    n_classes = num_classes_of(config)
    loader = make_loader(config)
    full_bag_ids = loader.groups
    full_coords = loader.coords

    if label_map is not None:
        from src.mil.shuffle_control import apply_permuted_labels
        loader.y_class = apply_permuted_labels(loader.groups, label_map)

    fold_data = None
    for fd in loader:
        if fd["fold_name"] == fold_name:
            fold_data = fd
            break
    if fold_data is None:
        raise ValueError(f"fold {fold_name!r} not found for {config_name}")

    tag = "shuffled" if label_map is not None else "real"
    print(f"[modal:{config_name}:{method}:{fold_name}:{tag}] fitting "
          f"(svm_subsample_per_bag={svm_subsample_per_bag})...")
    result = _fit_one_fold(fold_data, config, method, n_classes, pi_min, max_iters, seed,
                           full_bag_ids, full_coords,
                           svm_subsample_per_bag=svm_subsample_per_bag)
    print(f"[modal:{config_name}:{method}:{fold_name}:{tag}] done")
    return result


def run_mil_fit_parallel(config, methods=('misvm',), pi_min=None, max_iters=10, seed=None,
                         svm_subsample_per_bag=None, label_map=None, store_path=None):
    """
    Fit every (method, fold) pair for one config in parallel on Modal, then
    write every result into the local latent-label store.

    ``label_map`` / ``store_path`` (optional, both or neither): the same
    shuffled-label-control override scripts/shuffled_label_control.py
    uses locally (see fit_fold_remote's docstring) -- lets the shuffled
    mi-SVM control run fold-parallel on Modal instead of ~4h sequential
    locally. ``store_path`` defaults to the real store
    (results/<name>/latent/latent_labels.h5) when not given; pass
    results/<name>/latent/latent_labels_<suffix>.h5 together with a
    label_map to write a shuffled run to its own separate store, never the
    real one.
    """
    from pipeline._common import make_loader, set_seeds
    from src.mil.store import config_hash, write_posteriors

    seed = seed if seed is not None else set_seeds(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    svm_subsample_per_bag = (svm_subsample_per_bag if svm_subsample_per_bag is not None
                             else mil_cfg.get('svm_subsample_per_bag', 500))
    n_folds = int(config.get('cv_strategy', {}).get('n_folds', 5))
    fold_names = [f"fold_{i}" for i in range(n_folds)]

    if store_path is None:
        store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
            / 'latent' / 'latent_labels.h5'
    else:
        store_path = Path(store_path)

    tag = "shuffled" if label_map is not None else "real"
    print(f"\n{'='*80}\nMIL FIT STAGE (modal, fold-parallel, {tag}): {config['name']}\n{'='*80}")
    print(f"  methods: {methods}  folds: {fold_names}  pi_min={pi_min}  "
          f"svm_subsample_per_bag={svm_subsample_per_bag}")
    print(f"  store -> {store_path}")

    h5_name = _upload_h5(config)
    cfg_hash = config_hash(config)

    # sample_ids/coords for write_posteriors' alignment checksum -- built
    # locally once (data loading only, no fitting) rather than returned by
    # every remote container, which would repeat ~5MB of string data 2x5=10
    # times over the wire for no benefit.
    local_loader = make_loader(config)
    full_bag_ids = local_loader.groups
    full_coords = local_loader.coords

    jobs = [(method, fold_name) for method in methods for fold_name in fold_names]
    print(f"[modal] spawning {len(jobs)} parallel (method, fold) container(s)...")
    with app.run():
        handles = {
            (method, fold_name): fit_fold_remote.spawn(
                config['name'], method, fold_name, h5_name, pi_min, max_iters, seed,
                svm_subsample_per_bag=svm_subsample_per_bag, label_map=label_map)
            for method, fold_name in jobs
        }
        for (method, fold_name), handle in handles.items():
            print(f"  [modal] waiting for {method}:{fold_name}...")
            result_fold_name, variant, posterior_full, in_sample_full = handle.get()
            write_posteriors(
                store_path, method=method, variant=variant, fold_name=result_fold_name,
                posterior=posterior_full, in_sample=in_sample_full,
                sample_ids=full_bag_ids, coords=full_coords, config_hash=cfg_hash,
            )
            print(f"  [modal] {method}:{fold_name} complete -> written to store")

    print(f"\n  MIL fit stage complete (modal) -> {store_path}")
    return {'store_path': str(store_path), 'methods': list(methods)}
