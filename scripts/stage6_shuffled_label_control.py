"""
Stage 6 backlog: the shuffled-bag-label control (docs/MIL_REFRAME_PLAN.md
§6, backlog §13). Permutes which label goes with which bag (exact
permutation of the true label multiset -- src/mil/shuffle_control.py),
holds everything else fixed, refits a MIL formulation under the permuted
labels, and reports the same spatial statistics Stage 5 computes for the
real fit -- so the two can be compared directly. If lag>=3 coherence
survives this, the real result cannot be attributed to the true label.

Reuses each formulation's own per-fold fitting code UNCHANGED (the
permutation happens once, in the loader's own y_class array, before any
fold splitting -- everything downstream of that is identical to a real
fit): pipeline/mil_fit_stage.py's _fit_one_formulation for misvm,
pipeline/mil_attention_stage.py's mil_attention_stage for attention (its
frozen CVAE trunk is untouched by this -- only the attention head refits,
same as any other real attention fit), pipeline/modal_cvae.py's
run_on_modal for Tier 4's cvae (a full CVAE retrain, GPU, since Tier 4
conditions on the CVAE's own weights rather than a separate frozen-trunk
model -- the one tier whose control needs new training, not just a
downstream head). Writes to a SEPARATE store path (and, for
attention/cvae, separate bag-level/model output paths too) so a shuffled
run can never collide with or corrupt the real latent-label store or
artifacts that Tier 3/4's REAL results depend on.

Two steps for Tier 4, not one: first --method cvae (retrains the CVAE
itself under permuted labels, on Modal GPU), THEN --method cbr (scores
those already-shuffle-trained weights into 'reconstruction'/'encoder_head'
posteriors -- cheap, local, pure inference, identical to how the real Tier
4 fit works). --method cbr's own label permutation is a harmless no-op
(CbR's scoring never reads labels at all, only features and CVAE weights --
see the comment at its call site) -- what makes its result "shuffled" is
that --method cvae already trained the weights it loads.

Usage:
    python scripts/stage6_shuffled_label_control.py --config exp_501 --method misvm
    python scripts/stage6_shuffled_label_control.py --config exp_501 --method attention
    python scripts/stage6_shuffled_label_control.py --config exp_501 --method cvae [--epochs N]
    python scripts/stage6_shuffled_label_control.py --config exp_501 --method cbr
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import make_loader, num_classes_of, set_seeds
from pipeline.mil_attention_stage import mil_attention_stage
from pipeline.mil_fit_stage import _fit_one_formulation
from src.mil.shuffle_control import apply_permuted_labels, permuted_bag_label_map


def run(config_name, method, pi_min=None, max_iters=10, svm_subsample_per_bag=500,
       seed=42, permutation_seed=123, store_suffix='shuffled',
       attn_dim=32, dropout_rate=0.3, l2=1e-3, center_by_bag=True,
       sub_bag_size=None, n_per_bag_per_epoch=50, max_epochs=200, patience=15,
       fold_names=None, cvae_epochs=None, backend='local'):
    """``backend='modal'`` (misvm only): fold-parallel on Modal instead
    of ~4h sequential locally -- pipeline/modal_mil_fit.py::run_mil_fit_parallel's
    label_map/store_path override, the shuffled-control analogue of the real
    fit's own --backend modal path (pipeline/mil_fit_stage.py). attention's
    own training loop and cvae's Modal GPU retrain are unaffected by this
    flag -- they already run as they always have."""
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)

    if method == 'cvae':
        # Unlike misvm/attention, the permutation happens INSIDE the
        # remote container (cvae_stage.run_local, given just the seed) --
        # there is no local loader to substitute and hand off, since Modal
        # containers build their own loader from the config alone. See
        # pipeline/modal_cvae.py::run_on_modal's docstring.
        from pipeline.modal_cvae import run_on_modal
        print(f"\n{'='*80}\nSHUFFLED-LABEL CONTROL: cvae (Modal GPU)  ({config_name})\n{'='*80}")
        result = run_on_modal(config_name, epochs=cvae_epochs,
                              permutation_seed=permutation_seed, store_suffix=store_suffix)
        print(f"\n  Shuffled-label CVAE retrain complete -> {result['models_cvae_dir']}")
        return result

    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    svm_subsample_per_bag = svm_subsample_per_bag or mil_cfg.get('svm_subsample_per_bag', 500)

    store_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / f'latent_labels_{store_suffix}.h5'

    print(f"\n{'='*80}\nSHUFFLED-LABEL CONTROL: {method}  ({config_name})\n{'='*80}")
    print(f"  permutation_seed={permutation_seed}  store -> {store_path}")

    loader = make_loader(config)

    label_map = permuted_bag_label_map(loader.groups, loader.y_class, seed=permutation_seed)
    n_moved = sum(1 for b in label_map if label_map[b] != loader.y_class[loader.groups == b][0])
    print(f"  {n_moved}/{len(label_map)} bags received a label different from their true one")

    # The one, surgical intervention: substitute the loader's own y_class in
    # place, BEFORE any fold splitting (__iter__ reads self.y_class fresh
    # per fold). Everything else -- features, bag membership, the CV split
    # itself -- is untouched.
    loader.y_class = apply_permuted_labels(loader.groups, label_map)

    seed = seed if seed is not None else set_seeds(config)

    if method == 'attention':
        bag_level_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'latent' / f'attention_bag_level_{store_suffix}.json'
        mil_attention_stage(
            config, sub_bag_size=sub_bag_size, n_per_bag_per_epoch=n_per_bag_per_epoch,
            max_epochs=max_epochs, patience=patience, pi_min=pi_min, seed=seed,
            fold_names=fold_names, attn_dim=attn_dim, dropout_rate=dropout_rate, l2=l2,
            center_by_bag=center_by_bag, loader=loader, store_path=store_path,
            bag_level_path=bag_level_path,
        )
        print(f"\n  Shuffled-label control fit complete -> {store_path}")
        print(f"  Shuffled-label bag-level results -> {bag_level_path}")
        return {'store_path': str(store_path), 'bag_level_path': str(bag_level_path),
               'label_map': label_map}

    cbr_kwargs = {}
    if method == 'cbr':
        # Unlike misvm, CbR's own posterior computation
        # (_score_one_fold_cbr) never reads fd.y_train/y_val at all -- it is
        # a pure function of which CVAE weights it loads and which features
        # it scores (docs/mil_tiers_methodology.md §7). So the label
        # permutation above is a harmless no-op for this method specifically
        # -- what actually makes this a "shuffled" result is that the CVAE
        # weights themselves were trained under permuted labels (--method
        # cvae, above). Point at that run's weights, not the real ones.
        from pipeline.modal_cvae import _effective_output_dirs
        cvae_weights_dir, _ = _effective_output_dirs(config, store_suffix)
        mil_cfg = config.get('mil', {})
        cbr_kwargs = {'cvae_weights_dir': cvae_weights_dir,
                      'cbr_temperature': mil_cfg.get('cbr_temperature', 1.0)}
        print(f"  cvae_weights_dir -> {cvae_weights_dir}  "
              f"cbr_temperature={cbr_kwargs['cbr_temperature']}")

    if backend == 'modal' and method == 'misvm':
        from pipeline.modal_mil_fit import run_mil_fit_parallel
        run_mil_fit_parallel(
            config, methods=(method,), pi_min=pi_min, max_iters=max_iters, seed=seed,
            svm_subsample_per_bag=svm_subsample_per_bag, label_map=label_map, store_path=store_path,
        )
        print(f"\n  Shuffled-label control fit complete (modal) -> {store_path}")
        return {'store_path': str(store_path), 'label_map': label_map}

    _fit_one_formulation(
        method, loader, config, n_classes, store_path, pi_min, max_iters, seed,
        svm_subsample_per_bag=svm_subsample_per_bag, **cbr_kwargs,
    )
    print(f"\n  Shuffled-label control fit complete -> {store_path}")
    return {'store_path': str(store_path), 'label_map': label_map}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shuffled-bag-label control")
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True,
                        choices=['misvm', 'attention', 'cvae', 'cbr'])
    parser.add_argument('--epochs', type=int, default=None,
                        help="cvae only: override epoch count (smoke-test use)")
    parser.add_argument('--pi-min', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--svm-subsample-per-bag', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutation-seed', type=int, default=123)
    parser.add_argument('--store-suffix', default='shuffled',
                        help="shuffled store is latent_labels_<suffix>.h5 (default 'shuffled'); use a "
                             "distinct suffix to run two controls concurrently without two writers "
                             "on one HDF5 file")
    parser.add_argument('--attn-dim', type=int, default=32)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--no-center-by-bag', action='store_true')
    parser.add_argument('--sub-bag-size', type=int, default=None)
    parser.add_argument('--n-per-bag-per-epoch', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--folds', nargs='+', default=None,
                        help="attention only: subset of fold names, e.g. fold_0 (smoke-test use)")
    parser.add_argument('--backend', default='local', choices=['local', 'modal'],
                        help="misvm only: 'modal' fits every fold in its own parallel "
                             "CPU container instead of ~4h sequential locally (mirrors the "
                             "real fit's own --backend modal). No effect on attention/cvae/cbr.")
    args = parser.parse_args()
    run(args.config, args.method, pi_min=args.pi_min, max_iters=args.max_iters,
       svm_subsample_per_bag=args.svm_subsample_per_bag, seed=args.seed,
       permutation_seed=args.permutation_seed, attn_dim=args.attn_dim,
       dropout_rate=args.dropout_rate, l2=args.l2, center_by_bag=not args.no_center_by_bag,
       sub_bag_size=args.sub_bag_size, n_per_bag_per_epoch=args.n_per_bag_per_epoch,
       max_epochs=args.max_epochs, patience=args.patience, fold_names=args.folds,
       cvae_epochs=args.epochs, backend=args.backend, store_suffix=args.store_suffix)
