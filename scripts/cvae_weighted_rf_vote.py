"""
The key-instance utility test (docs/mil_validation_test_suite.md §B): keep
Tier 0's existing RF (raw 571-band spectrum, unchanged, the same trained
model everywhere else in this project) but replace its naive "1 instance = 1
vote" majority-vote pooling with alternatives informed by a given MIL tier's
own per-instance posterior (any (method, variant) stored in a config's
latent_labels.h5 -- originally built against exp_607's CbR 'reconstruction'
posterior only; --method/--variant now make this reusable for any tier/config,
e.g. attention's posterior, or CbR at a different beta).

Three variants. The first two are pure post-hoc re-aggregation of two
ALREADY-COMPUTED artifacts -- the shared TRTR RF's rf_fold_N.joblib and
exp_607's stored CbR posterior -- no new training:

- 'weighted': each instance's RF class-1 probability is weighted by the
  CVAE's own P(class=1) for that instance (from CbR's 2-way diagnostic
  posterior, renormalised) before averaging to a bag score. An instance the
  CVAE considers more "class-1-like" pulls the bag's aggregate score harder
  toward 1, and vice versa for class 0 -- weight = agreement, not confidence
  in some direction.
- 'key_instance_vote': the CVAE's non-diagnostic mass (CbR posterior column 0)
  is binarised via a median split (see docstring on
  ``_key_instance_vote_bag_decision`` for why not an absolute 0.5 threshold).
  Only key instances' RF votes enter a plain majority vote; a bag with zero
  key instances falls back to all of its instances (flagged, not silently
  dropped). This RF is the SAME already-trained Tier 0 model -- key-instance
  selection only changes which of its existing votes get counted.

A third, genuinely different variant -- a NEW classifier, not a
re-aggregation of an old one:

- 'key_instance_train': per fold, the median-split threshold is fit from
  TRAINING instances' own non-diagnostic mass only (never from validation --
  that would leak), then a FRESH RandomForestClassifier (same hyperparameters
  as Tier 0's RF) is fit on ONLY the training instances the CVAE calls "key"
  under that threshold. At inference, the same threshold decides which
  validation instances are "key"; only those are ever shown to this new RF.
  This is the two-stage pipeline the user described: the CVAE decides which
  instances matter, then a classifier trained only on what it considers
  informative does the classifying -- not Tier 0's RF with a different vote,
  a different RF with different training data. Honest confound to flag: this
  RF sees roughly half the training instances Tier 0's RF does, so any
  difference conflates "does key-instance filtering help" with "does less
  training data hurt."

A fourth variant, the confound-(i) control for 'key_instance_train':

- 'random_subsample': per fold, a fresh RandomForestClassifier fit on a
  RANDOM subset of training instances the SAME SIZE as that fold's real
  CVAE-selected key set (median split), scored on the identical
  CVAE-selected key_val as 'key_instance_train' -- the only difference from
  'key_instance_train' is which mechanism chose the training subset. If
  'key_instance_train' does no better than this, the apparent effect is
  attributable to training-set size, not CVAE selection.

All four are compared against the plain majority-vote RF baseline (Tier 0's
own rule, same already-trained model) via the same bag-level bootstrap used
everywhere else in this project.

Pooling instances across folds directly (ignoring fold boundaries) is valid
for 'weighted'/'key_instance_vote' for a duplicated bag because both
statistics computed are linear (weighted mean) or the project's own
already-proven majority-vote invariance
(src/utils/statistical_evaluation.py::bag_decisions's docstring) -- a
duplicated bag has the same instance count in every fold it appears in, so
pooling-then-aggregating equals per-fold-then-averaging for both.
'key_instance_train' instead averages each duplicated bag's PER-FOLD scores
(it cannot pool instances across folds at all -- each fold fits its own,
different RF), the same "per-fold then average" convention
aggregate_bag_scores_across_folds uses for its own 'mean' rule.

Usage:
    python scripts/cvae_weighted_rf_vote.py --cvae-config exp_607
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import make_loader, num_classes_of, prepare_fold
from src.mil.store import read_posteriors
from src.utils.statistical_evaluation import (
    bag_decisions, bag_level_bootstrap, find_model_path, load_and_predict,
    load_and_predict_proba, summarize_bootstrap_diffs,
)


def _collect_per_fold(config, cvae_config_name, method='cbr', variant='reconstruction'):
    """One pass over every fold: RF's per-instance P(class=1) (the shared
    TRTR RF, unchanged) AND its own ``.predict()`` hard label, the CVAE's
    per-instance posterior for the SAME out-of-sample instances (same order
    -- see module docstring/commit message for the alignment argument), and
    the true bag id/label. Returns concatenated 1-D arrays (pooled across
    folds).

    Both ``rf_p1`` (continuous) and ``rf_pred_hard`` (``.predict()``) are
    returned deliberately, not just the former thresholded at 0.5: confirmed
    directly on real data that sklearn's ``RandomForestClassifier.predict()``
    and ``(predict_proba()[:, 1] >= 0.5)`` can disagree on a real,
    non-trivial fraction of instances (69/3686 for one bag, enough to flip
    that bag's majority vote) -- a genuine floating-point-accumulation
    quirk, not a tie-breaking edge case. The project's established
    convention everywhere else (``load_and_predict``, used by
    `scripts/stage6_bag_eval.py`/`pipeline/evaluation_stage.py`) is
    ``.predict()``; any hard decision here must match it, or this script's
    'baseline' silently stops being the same baseline every other number in
    the project is compared against."""
    from src.utils.config_manager import ConfigManager

    cvae_config = ConfigManager.load_config(cvae_config_name).to_dict()
    store_path = Path(cvae_config['paths']['experiment_root']) / 'latent' / 'latent_labels.h5'
    rf_dir = config['paths']['models_baseline_dir']

    loader = make_loader(config)
    full_bag_ids = loader.groups

    all_rf_p1, all_rf_pred_hard, all_cvae_p1, all_non_diag, all_bag_ids, all_y = [], [], [], [], [], []

    for fold_idx, fold_data in enumerate(loader):
        fd = prepare_fold(fold_data, config, cap_train=False)
        rf_path = find_model_path(rf_dir, 'rf', '.joblib', fold_idx)
        rf_proba = load_and_predict_proba(rf_path, 'sklearn', fd.X_val)
        rf_p1 = rf_proba[:, 1]
        rf_pred_hard_fold = load_and_predict(rf_path, 'sklearn', fd.X_val)

        data = read_posteriors(store_path, method, variant, fold_data['fold_name'])
        posterior, in_sample = data['posterior'], data['in_sample']
        val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
        post_val = posterior[val_mask_full]
        assert post_val.shape[0] == fd.X_val.shape[0], (
            f"fold {fold_data['fold_name']}: CbR posterior ({post_val.shape[0]} rows) and "
            f"RF val set ({fd.X_val.shape[0]} rows) misaligned -- the row-order equivalence "
            f"this script depends on (module docstring) does not hold for this data.")

        non_diag = post_val[:, 0]
        diag = post_val[:, 1:]
        diag_sum = diag.sum(axis=1, keepdims=True)
        diag_norm = diag / np.where(diag_sum > 0, diag_sum, 1.0)
        cvae_p1 = diag_norm[:, 1]

        all_rf_p1.append(rf_p1)
        all_rf_pred_hard.append(rf_pred_hard_fold)
        all_cvae_p1.append(cvae_p1)
        all_non_diag.append(non_diag)
        all_bag_ids.append(fd.val_bag_ids if fd.val_bag_ids is not None else fold_data['val_sample_ids'])
        all_y.append(fd.y_val)
        print(f"  fold_{fold_idx}: {len(rf_p1)} val instances")

    return (np.concatenate(all_rf_p1), np.concatenate(all_rf_pred_hard), np.concatenate(all_cvae_p1),
           np.concatenate(all_non_diag), np.concatenate(all_bag_ids), np.concatenate(all_y))


def _collect_train_and_val_per_fold(config, cvae_config_name, method='cbr', variant='reconstruction'):
    """Like _collect_per_fold, but keeps each fold SEPARATE (not pooled) and
    returns both TRAIN and val raw spectra + non-diagnostic mass -- what
    'key_instance_train' needs to fit a per-fold RF on train's own key
    instances, decided by a threshold fit from train alone (never val)."""
    from src.utils.config_manager import ConfigManager

    cvae_config = ConfigManager.load_config(cvae_config_name).to_dict()
    store_path = Path(cvae_config['paths']['experiment_root']) / 'latent' / 'latent_labels.h5'

    loader = make_loader(config)
    full_bag_ids = loader.groups

    per_fold = []
    for fold_data in loader:
        fd = prepare_fold(fold_data, config, cap_train=False)
        data = read_posteriors(store_path, method, variant, fold_data['fold_name'])
        posterior = data['posterior']

        train_mask_full = np.isin(full_bag_ids, fold_data['train_sample_ids'])
        val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
        train_post = posterior[train_mask_full]
        val_post = posterior[val_mask_full]
        assert train_post.shape[0] == fd.X_train.shape[0]
        assert val_post.shape[0] == fd.X_val.shape[0]

        per_fold.append({
            'fold_name': fold_data['fold_name'],
            'X_train': fd.X_train, 'y_train': fd.y_train, 'train_non_diag': train_post[:, 0],
            'X_val': fd.X_val, 'y_val': fd.y_val, 'val_non_diag': val_post[:, 0],
            'val_bag_ids': fd.val_bag_ids if fd.val_bag_ids is not None else fold_data['val_sample_ids'],
        })
        print(f"  {fold_data['fold_name']}: train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")
    return per_fold


def _fit_and_score_random_subsample_rf(per_fold, rf_kwargs, seed=123):
    """Confound-(i) control for 'key_instance_train' (docs/mil_validation_test_suite.md
    §B): per fold, train a fresh RF on a RANDOM subset of training instances
    the SAME SIZE as that fold's real CVAE-selected key set, but score
    validation using the identical CVAE-selected key_val as
    'key_instance_train' -- the only thing that differs between this and
    'key_instance_train' is which mechanism chose the training subset
    (random vs. CVAE-selected), size held fixed. If 'key_instance_train'
    and this control perform similarly, the apparent effect of key-instance
    training is attributable to training-set SIZE, not CVAE selection; if
    'key_instance_train' clearly beats this control, the CVAE's selection is
    doing real work beyond just subsampling."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    per_bag_fold_scores = {}
    bag_true = {}
    n_fallback = 0
    for pf in per_fold:
        threshold = float(np.median(pf['train_non_diag']))
        key_train = pf['train_non_diag'] < threshold
        key_val = pf['val_non_diag'] < threshold  # same val selection as key_instance_train

        n_train, n_key = len(key_train), int(key_train.sum())
        random_train = np.zeros(n_train, dtype=bool)
        random_train[rng.choice(n_train, size=n_key, replace=False)] = True

        rf = RandomForestClassifier(
            n_estimators=rf_kwargs.get('n_estimators', 100),
            class_weight=rf_kwargs.get('class_weight', 'balanced'),
            random_state=rf_kwargs.get('random_state', seed),
            n_jobs=rf_kwargs.get('n_jobs', -1),
        )
        rf.fit(pf['X_train'][random_train], pf['y_train'][random_train])
        print(f"  {pf['fold_name']}: random_train={random_train.sum()}/{n_train} "
             f"(size-matched to key_train={n_key})  "
             f"key_val={key_val.sum()}/{len(key_val)} ({key_val.mean():.1%})")

        bag_ids_val, y_val, X_val = pf['val_bag_ids'], pf['y_val'], pf['X_val']
        for b in np.unique(bag_ids_val):
            m = bag_ids_val == b
            m_key = m & key_val
            if not m_key.any():
                m_key = m
                n_fallback += 1
            pred_b = rf.predict(X_val[m_key])
            score = float((pred_b == 1).mean())
            per_bag_fold_scores.setdefault(b, []).append(score)
            bag_true[b] = int(y_val[m][0])

    bags = np.array(sorted(per_bag_fold_scores))
    bag_scores = np.array([np.mean(per_bag_fold_scores[b]) for b in bags])
    bag_pred = (bag_scores >= 0.5).astype(int)
    bag_true_arr = np.array([bag_true[b] for b in bags])
    return bags, bag_pred, bag_true_arr, n_fallback


def _fit_and_score_key_instance_train_rf(per_fold, rf_kwargs, seed=42):
    """The two-stage pipeline: per fold, fit a median-split 'key' threshold
    from TRAIN's own non-diagnostic mass (never val -- that would leak), fit
    a FRESH RandomForestClassifier on only the training instances below it,
    then score ONLY the val instances below that SAME threshold. Returns
    per-(bag, fold) class-1 fractions so a bag duplicated across folds gets
    its per-fold scores averaged (aggregate_bag_scores_across_folds's own
    'mean'-rule convention), not pooled instances (impossible here -- each
    fold's RF is a different model)."""
    from sklearn.ensemble import RandomForestClassifier

    per_bag_fold_scores = {}  # bag_id -> list of per-fold class1 fractions
    bag_true = {}
    n_fallback = 0
    for pf in per_fold:
        threshold = float(np.median(pf['train_non_diag']))
        key_train = pf['train_non_diag'] < threshold
        key_val = pf['val_non_diag'] < threshold

        rf = RandomForestClassifier(
            n_estimators=rf_kwargs.get('n_estimators', 100),
            class_weight=rf_kwargs.get('class_weight', 'balanced'),
            random_state=rf_kwargs.get('random_state', seed),
            n_jobs=rf_kwargs.get('n_jobs', -1),
        )
        rf.fit(pf['X_train'][key_train], pf['y_train'][key_train])
        print(f"  {pf['fold_name']}: threshold={threshold:.4f}  "
             f"key_train={key_train.sum()}/{len(key_train)} ({key_train.mean():.1%})  "
             f"key_val={key_val.sum()}/{len(key_val)} ({key_val.mean():.1%})")

        bag_ids_val, y_val, X_val = pf['val_bag_ids'], pf['y_val'], pf['X_val']
        for b in np.unique(bag_ids_val):
            m = bag_ids_val == b
            m_key = m & key_val
            if not m_key.any():
                m_key = m
                n_fallback += 1
            pred_b = rf.predict(X_val[m_key])
            score = float((pred_b == 1).mean())
            per_bag_fold_scores.setdefault(b, []).append(score)
            bag_true[b] = int(y_val[m][0])

    bags = np.array(sorted(per_bag_fold_scores))
    bag_scores = np.array([np.mean(per_bag_fold_scores[b]) for b in bags])
    bag_pred = (bag_scores >= 0.5).astype(int)
    bag_true_arr = np.array([bag_true[b] for b in bags])
    return bags, bag_pred, bag_true_arr, n_fallback


def _weighted_vote_bag_decision(rf_p1, cvae_p1, bag_ids):
    """Per bag: weighted mean of RF's P(class=1), weights = CVAE's P(class=1).
    Decision = 1 if weighted mean >= 0.5."""
    unique_bags = np.unique(bag_ids)
    bag_pred = np.zeros(len(unique_bags), dtype=int)
    for i, b in enumerate(unique_bags):
        m = bag_ids == b
        w = cvae_p1[m]
        w_sum = w.sum()
        score = (rf_p1[m] * w).sum() / w_sum if w_sum > 0 else rf_p1[m].mean()
        bag_pred[i] = int(score >= 0.5)
    return unique_bags, bag_pred


def _key_instance_vote_bag_decision(rf_pred_hard, non_diag_mass, bag_ids):
    """Per bag: plain majority vote of RF's hard prediction (``.predict()``,
    NOT a manual ``predict_proba()>=0.5`` threshold -- see
    ``_collect_per_fold``'s docstring for why these can genuinely disagree),
    restricted to 'key' instances. 'Key' is a median split on non-diagnostic
    mass, NOT an absolute 0.5 threshold: for this K=2 joint-softmax
    posterior, non-diagnostic mass is provably bounded at (K-1)/(2K-1)=1/3
    (docs/mil_tiers_methodology.md §7) -- confirmed directly on this data
    (observed max=0.3333) -- so a 0.5 threshold is structurally unreachable
    and would make every instance 'key', a silent no-op. The median split
    gives a real, data-driven half/half separation instead. Falls back to
    all instances if a bag somehow has none (flagged, not silently
    dropped) -- shouldn't trigger under a median split but kept as a guard."""
    rf_pred = rf_pred_hard
    key_threshold = float(np.median(non_diag_mass))
    key = non_diag_mass < key_threshold
    print(f"  key-instance threshold (median non-diagnostic mass): {key_threshold:.4f} "
         f"({key.mean():.1%} of instances are 'key')")
    unique_bags = np.unique(bag_ids)
    bag_pred = np.zeros(len(unique_bags), dtype=int)
    n_fallback = 0
    for i, b in enumerate(unique_bags):
        m = bag_ids == b
        m_key = m & key
        if not m_key.any():
            m_key = m
            n_fallback += 1
        votes = rf_pred[m_key]
        bag_pred[i] = int(np.bincount(votes).argmax())
    return unique_bags, bag_pred, n_fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='exp_607', help="config whose RF/data to use (shares exp_501's TRTR)")
    parser.add_argument('--cvae-config', default='exp_607', help="config whose posterior to use as the weighting signal")
    parser.add_argument('--method', default='cbr', help="posterior source stored in latent_labels.h5, e.g. 'cbr', 'attention'")
    parser.add_argument('--variant', default='reconstruction',
                        help="e.g. 'reconstruction'/'encoder_head' for cbr, 'attention' for attention")
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random-subsample-seed', type=int, default=123)
    parser.add_argument('--skip-train-variant', action='store_true',
                        help="skip 'key_instance_train'/'random_subsample' (fits a fresh RF per fold -- slower)")
    args = parser.parse_args()

    from src.utils.config_manager import ConfigManager
    from sklearn.metrics import accuracy_score

    config = ConfigManager.load_config(args.config).to_dict()
    rf_kwargs = config['models'].get('random_forest', {})

    print(f"\n{'='*80}\nCVAE-WEIGHTED RF VOTE: RF from {config['paths']['models_baseline_dir']}, "
         f"{args.method}/{args.variant} from {args.cvae_config}\n{'='*80}")
    rf_p1, rf_pred_hard, cvae_p1, non_diag, bag_ids, y = _collect_per_fold(
        config, args.cvae_config, method=args.method, variant=args.variant)
    print(f"  pooled: {len(rf_p1)} out-of-sample instances, {len(np.unique(bag_ids))} unique bags")

    # Baseline: plain majority vote (Tier 0's own rule), same already-trained RF.
    # Uses rf_pred_hard (.predict()), NOT (rf_p1 >= 0.5) -- see _collect_per_fold's
    # docstring: these can genuinely disagree, and only .predict() matches the
    # project-wide convention (load_and_predict) this baseline must stay comparable to.
    baseline_bags, baseline_pred, baseline_true = bag_decisions(rf_pred_hard, bag_ids, y, rule='majority')
    baseline_true_by_bag = dict(zip(baseline_bags, baseline_true))
    baseline_pred_by_bag = dict(zip(baseline_bags, baseline_pred))

    weighted_bags, weighted_pred = _weighted_vote_bag_decision(rf_p1, cvae_p1, bag_ids)
    key_bags, key_pred, n_fallback = _key_instance_vote_bag_decision(rf_pred_hard, non_diag, bag_ids)
    assert np.array_equal(baseline_bags, weighted_bags) and np.array_equal(baseline_bags, key_bags)

    baseline_acc = accuracy_score(baseline_true, baseline_pred)
    weighted_acc = accuracy_score(baseline_true, weighted_pred)
    key_acc = accuracy_score(baseline_true, key_pred)
    n_bags = len(baseline_bags)

    print(f"\n  plain majority vote (baseline): bag_acc={baseline_acc:.4f}  (n={n_bags} bags)")

    results = [
        ('weighted vote (CVAE P1 as weight, same RF)', baseline_pred, weighted_pred, baseline_true, weighted_acc),
        (f'key-instance vote (median split, same RF, {n_fallback} bags fell back)',
         baseline_pred, key_pred, baseline_true, key_acc),
    ]

    if not args.skip_train_variant:
        print("\n  Fitting key_instance_train (a NEW RF per fold on key-only training instances)...")
        per_fold = _collect_train_and_val_per_fold(config, args.cvae_config, method=args.method, variant=args.variant)
        train_bags, train_pred, train_true, train_n_fallback = \
            _fit_and_score_key_instance_train_rf(per_fold, rf_kwargs, seed=args.seed)
        # Realign to baseline's bag order/dict (bag_level_bootstrap needs paired arrays).
        aligned_baseline_pred = np.array([baseline_pred_by_bag[b] for b in train_bags])
        aligned_baseline_true = np.array([baseline_true_by_bag[b] for b in train_bags])
        assert np.array_equal(aligned_baseline_true, train_true), \
            "bag true-label mismatch between baseline and key_instance_train -- alignment bug"
        train_acc = accuracy_score(train_true, train_pred)
        results.append((
            f'key_instance_train (NEW RF, median split, {train_n_fallback} bags fell back)',
            aligned_baseline_pred, train_pred, aligned_baseline_true, train_acc,
        ))

        print("\n  Fitting random_subsample (confound-(i) control, size-matched random training set)...")
        rs_bags, rs_pred, rs_true, rs_n_fallback = \
            _fit_and_score_random_subsample_rf(per_fold, rf_kwargs, seed=args.random_subsample_seed)
        aligned_rs_baseline_pred = np.array([baseline_pred_by_bag[b] for b in rs_bags])
        aligned_rs_baseline_true = np.array([baseline_true_by_bag[b] for b in rs_bags])
        assert np.array_equal(aligned_rs_baseline_true, rs_true), \
            "bag true-label mismatch between baseline and random_subsample -- alignment bug"
        rs_acc = accuracy_score(rs_true, rs_pred)
        results.append((
            f'random_subsample (confound-(i) control, size-matched, {rs_n_fallback} bags fell back)',
            aligned_rs_baseline_pred, rs_pred, aligned_rs_baseline_true, rs_acc,
        ))
        print(f"\n  [confound check] key_instance_train vs random_subsample: "
             f"{train_acc:.4f} vs {rs_acc:.4f}  "
             f"({'selection beats size-matched random' if train_acc > rs_acc else 'no better than random subsampling of the same size' if train_acc <= rs_acc else ''})")

        # Per-slide predictions of the three arms, for the paired bootstrap test
        # (scripts/key_instance_paired_bootstrap.py).
        rs_pred_by_bag = dict(zip(rs_bags, rs_pred))
        cvae_root = Path(ConfigManager.load_config(args.cvae_config).to_dict()['paths']['experiment_root'])
        per_slide_path = (cvae_root / 'mil_eval' / 'key_instance_per_slide'
                          / f'{args.method}_{args.variant}.json')
        per_slide_path.parent.mkdir(parents=True, exist_ok=True)
        per_slide_path.write_text(json.dumps({
            'config': args.config, 'cvae_config': args.cvae_config,
            'method': args.method, 'variant': args.variant,
            'seed': args.seed, 'random_subsample_seed': args.random_subsample_seed,
            'slides': [str(b) for b in train_bags],
            'y_true': [int(v) for v in train_true],
            'naive': [int(v) for v in aligned_baseline_pred],
            'selected': [int(v) for v in train_pred],
            'random': [int(rs_pred_by_bag[b]) for b in train_bags],
            'n_slides_fell_back': int(train_n_fallback),
        }, indent=1), encoding='utf-8')
        print(f"  per-slide predictions -> {per_slide_path}")

    for label, b_pred, cand_pred, b_true, cand_acc in results:
        _, _, diffs = bag_level_bootstrap(b_pred, cand_pred, b_true,
                                          n_boot=args.n_boot, seed=args.seed)
        boot = summarize_bootstrap_diffs(diffs)
        print(f"  {label}: bag_acc={cand_acc:.4f}  delta={cand_acc - baseline_acc:+.4f}  "
             f"95%CI[{boot['ci_diff_lo']:+.4f},{boot['ci_diff_hi']:+.4f}]  "
             f"p={boot['p_bootstrap']:.4f}  significant={boot['p_bootstrap'] < 0.05}")


if __name__ == '__main__':
    main()
