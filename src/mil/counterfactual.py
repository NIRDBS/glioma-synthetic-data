"""
CVAE counterfactual probe -- a free-standing analysis, not a Stage-N MIL tier
(it doesn't produce a competing (K+1)-way instance posterior and isn't
subject to the bag-accuracy/spatial-coherence machinery every Stage 0-6 tier
was evaluated by). See docs/counterfactual_probe.md for the full writeup and
docs/MIL_REFRAME_PLAN.md's plan file / approved plan for why this exists.

Core idea: encode a real spectrum to z_mean (the SAME z_mean-only convention
used everywhere else in this project -- src/mil/generative.py:98,
pipeline/evaluation_stage.py:132), then decode that same z_mean under BOTH
class one-hot labels. Delta(z) = D(z, mutant) - D(z, wildtype) is a learned,
per-instance "IDH effect" spectrum.

Class-index convention (verified from three independent sources this
session: src/evaluation/synthetic_realism.py's CLASS_NAMES/binary mapping,
pipeline/_common.py::prepare_fold's "y<3 -> 0 Mutant, else 1 Wildtype"
docstring, paper/main.tex's LGm-to-IDH correspondence): class 0 = IDH-Mutant,
class 1 = IDH-Wildtype. Every signed Delta value and direction string in this
module follows this convention -- see test_class_index_constants_match_project_convention.

Architectural caveat this module exists partly to MEASURE, not assume away
(a reviewer's concern, confirmed directly against src/models/cvae.py:78,79,91):
the encoder never sees the label, but z_mean/z_log_var/the auxiliary
class_pred head are three separate Dense layers reading the SAME shared
`features` tensor. Training can push class information into z_mean, which
would make the decoder partially ignore the label at decode time -- Delta
would then underestimate the true effect. fit_z_class_probe/score_z_class_probe
below measure exactly how much of this leakage exists, independent of
anything else in this module.
"""
import numpy as np

IDH_MUTANT_CLASS_IDX = 0
IDH_WILDTYPE_CLASS_IDX = 1


def compute_counterfactual_reconstructions(X, encoder, decoder, num_classes, batch_size=256):
    """
    One encoder forward pass -> z_mean, then ``num_classes`` decoder passes,
    one per one-hot class condition, on that SAME z_mean. Reuses the exact
    decode-loop pattern in src/mil/generative.py::instance_posteriors_from_cvae
    (its ``mse`` computation, generalised to return the raw reconstructions
    themselves rather than an MSE-derived posterior -- this probe needs
    D(z,k) directly, not a classification score).

    encoder/decoder: same .predict(...) interface
    pipeline._cvae_gen.load_encoder_decoder returns (deliberately not
    type-checked, so tests use lightweight fakes, no real CVAE needed).

    Returns {'z_mean': (n, latent_dim) float32, 'recon': (n, num_classes, n_features) float32}.
    Call this PER FOLD on that fold's own out-of-sample split only -- a full
    fold's (n, K, 571) float32 recon array is a genuine ~130-265MB transient;
    do not accumulate all 5 folds' raw recon arrays simultaneously.
    """
    import tensorflow as tf

    n_features = X.shape[1]
    X_exp = X.reshape(-1, n_features, 1)
    z_mean, _, _, _ = encoder.predict(X_exp, batch_size=batch_size, verbose=0)

    recon = np.zeros((len(X), num_classes, n_features), dtype=np.float32)
    for k in range(num_classes):
        label = tf.keras.utils.to_categorical(
            [k] * len(X), num_classes=num_classes).astype('float32')
        rec = decoder.predict([z_mean, label], batch_size=batch_size, verbose=0)
        if rec.ndim == 3:                       # (n, n_features, 1) -> (n, n_features)
            rec = rec.reshape(rec.shape[0], -1)
        recon[:, k, :] = rec

    return {'z_mean': z_mean.astype(np.float32), 'recon': recon}


def compute_delta(recon, mutant_idx=IDH_MUTANT_CLASS_IDX, wildtype_idx=IDH_WILDTYPE_CLASS_IDX):
    """
    Delta(z) = D(z, mutant) - D(z, wildtype), per instance, per feature.
    recon: (n, K, n_features) from compute_counterfactual_reconstructions.
    Returns (n, n_features) float32.

    K=2-only by construction (a scalar difference has no natural K>2
    generalisation, and this project's exp_501 is K=2 throughout --
    docs/mil_tiers_methodology.md §1).

    Units: the CVAE decoder's own normalized-intensity scale (a sigmoid
    output, the same nominal [0,1] range as the minmax-per-spectrum-
    normalized input) -- NOT raw Raman counts.
    """
    recon = np.asarray(recon)
    return (recon[:, mutant_idx, :] - recon[:, wildtype_idx, :]).astype(np.float32)


def merge_out_of_sample_arrays(per_fold_results, full_bag_ids):
    """
    Generic analogue of src/mil/store.py::merge_out_of_sample_posterior for
    an ARBITRARY-shaped per-instance array (Delta, z_mean, a probe's proba --
    anything (n_val, *shape)), rather than only a (N,K+1) posterior. The
    store's own function is not reused because it hard-codes a
    renormalise-to-1 step that does not apply to a reconstructed spectrum or
    a latent vector. Same cross-fold duplicate-bag averaging convention as
    everywhere else in this project (docs/MIL_REFRAME_PLAN.md §7.1's 4 bags
    spanning two CV test folds: each contributing fold's out-of-sample
    estimate is averaged, never picked arbitrarily).

    per_fold_results: list of (val_mask_full (bool, N,), array (n_val, *shape))
    pairs, one per fold -- val_mask_full marks which of the FULL dataset's N
    rows this fold's array covers (in the same row order as `array`).

    Returns (merged (N, *shape) float32, NaN where no fold covered that row;
    n_folds_covering (N,) int).
    """
    n_full = len(full_bag_ids)
    sums = None
    counts = np.zeros(n_full, dtype=np.int64)

    for val_mask_full, array in per_fold_results:
        array = np.asarray(array, dtype=np.float64)
        if sums is None:
            sums = np.zeros((n_full,) + array.shape[1:], dtype=np.float64)
        sums[val_mask_full] += array
        counts[val_mask_full] += 1

    merged = np.full_like(sums, np.nan)
    covered = counts > 0
    merged[covered] = sums[covered] / counts[covered].reshape(
        (-1,) + (1,) * (sums.ndim - 1))
    return merged.astype(np.float32), counts


def mean_delta_per_bag(delta, bag_ids):
    """
    Group-by-bag mean of an already-merged, leak-free (N, n_features) Delta
    array. Returns (bag_ids_sorted, per_bag_mean (n_bags, n_features),
    grand_mean (n_features,)).

    grand_mean is the UNWEIGHTED mean of the per-bag means (a "bag-of-bags"
    mean), NOT a naive pooled-instance mean -- bag size varies ~8x across
    this dataset (docs/mil_tiers_methodology.md §3), exactly the
    large-bag-dominance problem this project already guards against
    elsewhere (topq's fractional-not-count design, every
    aggregate_bag_scores_across_folds pooling rule). Every "mean Delta"
    reported anywhere in this analysis -- global, per-fold, real, or
    shuffled -- must use this same convention.

    Rows that are entirely NaN (a bag with zero covered instances) are
    excluded from both the per-bag output and the grand mean, not silently
    treated as zero.
    """
    bag_ids = np.asarray(bag_ids)
    delta = np.asarray(delta)
    bags_sorted = sorted(set(bag_ids.tolist()))
    per_bag = []
    kept_bags = []
    for b in bags_sorted:
        idx = np.where(bag_ids == b)[0]
        rows = delta[idx]
        valid = ~np.isnan(rows).any(axis=1)
        if not valid.any():
            continue
        kept_bags.append(b)
        per_bag.append(rows[valid].mean(axis=0))
    per_bag = np.array(per_bag, dtype=np.float32)
    grand_mean = per_bag.mean(axis=0).astype(np.float32)
    return np.array(kept_bags, dtype=object), per_bag, grand_mean


def cosine_to_reference(vectors, reference):
    """
    Cosine similarity of each row in ``vectors`` (n,d) against one fixed
    ``reference`` (d,) vector. Scale-free companion to
    normalized_distance_to_reference -- deliberately paired the same way this
    project pairs Moran's I (scale-free) with the variogram (scale-sensitive,
    src/evaluation/spatial.py), since a scale-free statistic alone can read
    as large on an otherwise-tiny effect (docs/mil_tiers_methodology.md §10).

    Returns (n,) float, NaN for any all-zero row or a zero reference.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(reference)
    row_norms = np.linalg.norm(vectors, axis=1)
    dots = vectors @ reference
    denom = row_norms * ref_norm
    out = np.full(len(vectors), np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = dots[valid] / denom[valid]
    return out


def normalized_distance_to_reference(vectors, reference):
    """
    ||vectors[i] - reference|| / ||reference||, per row. The scale-SENSITIVE
    companion to cosine_to_reference -- see that function's docstring for why
    both are reported together.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(reference)
    diffs = np.linalg.norm(vectors - reference, axis=1)
    if ref_norm <= 0:
        return np.full(len(vectors), np.nan, dtype=np.float64)
    return diffs / ref_norm


def pairwise_cosine_stability(vectors):
    """
    All-pairs cosine similarity + effective-count diagnostic. SAME
    construction as src/mil/common.py::null_component_collapse_diagnostic
    (normalise rows -> cosine Gram matrix -> eigenvalue participation ratio
    effective_n = (sum eigvals)^2 / sum(eigvals^2)), reused here for a
    DIFFERENT question with the OPPOSITE reading: there (3 null components),
    near-1.0 pairwise cosine = collapse = bad. HERE (m folds' mean Delta
    vectors, or a {real, shuffled} pair), near-1.0 pairwise cosine = the
    folds independently agree on the same "IDH effect" direction = a real,
    reproducible signal; effective_n near 1 (out of max m) = stable; near m =
    no cross-fold agreement, i.e. fold-specific noise.

    vectors: (m, d). Returns {'pairwise_cosine': (m,m) float, 'effective_n_components': float,
    'mean_offdiag_cosine': float}.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    m = vectors.shape[0]
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.where(norm > 0, norm, 1.0)
    cos_sim = normed @ normed.T

    eigvals = np.linalg.eigvalsh(cos_sim)
    eigvals = np.clip(eigvals, 0, None)
    effective_n = float((eigvals.sum() ** 2) / max((eigvals ** 2).sum(), 1e-12))

    off_diag_mask = ~np.eye(m, dtype=bool)
    mean_offdiag = float(cos_sim[off_diag_mask].mean()) if m > 1 else float('nan')

    return {'pairwise_cosine': cos_sim, 'effective_n_components': effective_n,
            'mean_offdiag_cosine': mean_offdiag}


def random_direction_null(d, m=5, n_trials=1000, seed=42):
    """
    Monte Carlo null for pairwise_cosine_stability's mean_offdiag_cosine: m
    independent random unit vectors in R^d, repeated n_trials times -- gives
    an empirical null distribution to z/p-score real cross-fold agreement
    against, mirroring src/evaluation/spatial.py::permutation_null_by_lag's
    "a real statistic needs a null, not a bare number" convention.

    Returns {'null_mean': float, 'null_std': float}.
    """
    if m <= 1:
        return {'null_mean': float('nan'), 'null_std': float('nan')}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_trials, dtype=np.float64)
    off_diag_mask = ~np.eye(m, dtype=bool)
    for t in range(n_trials):
        vecs = rng.normal(size=(m, d))
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.where(norm > 0, norm, 1.0)
        cos_sim = normed @ normed.T
        stats[t] = cos_sim[off_diag_mask].mean()
    return {'null_mean': float(stats.mean()), 'null_std': float(stats.std())}


def within_class_reconstruction_spread(recon, y_true, mutant_idx=IDH_MUTANT_CLASS_IDX,
                                       wildtype_idx=IDH_WILDTYPE_CLASS_IDX):
    """
    Operationalises "how much the decoder uses the label" via the natural
    noise floor: for each true class k, take ONLY the FACTUAL reconstructions
    of instances whose true label is k, decoded under their OWN matching
    condition -- {recon[i, k, :] : y_true[i] == k} -- and compute the
    per-feature std across those instances.

    This isolates "how much do genuine same-class reconstructions vary from
    spectrum to spectrum" (the natural noise floor) from any effect of z
    partially leaking class information (which would only show up if
    counterfactual, cross-class decodes were mixed in here -- deliberately
    NOT done, to keep this check independent of fit_z_class_probe's
    question).

    Returns {'mutant': (n_features,) std, 'wildtype': (n_features,) std,
    'pooled': elementwise mean of the two}.
    """
    recon = np.asarray(recon)
    y_true = np.asarray(y_true).astype(int)

    mutant_mask = y_true == mutant_idx
    wildtype_mask = y_true == wildtype_idx
    mutant_std = recon[mutant_mask, mutant_idx, :].std(axis=0)
    wildtype_std = recon[wildtype_mask, wildtype_idx, :].std(axis=0)
    pooled = (mutant_std + wildtype_std) / 2.0
    return {'mutant': mutant_std.astype(np.float32),
            'wildtype': wildtype_std.astype(np.float32),
            'pooled': pooled.astype(np.float32)}


def delta_effect_size(mean_delta, spread):
    """
    |mean_delta| / spread, per feature -- a Cohen's-d-like ratio: "the IDH
    effect at this band is X times the natural within-class reconstruction
    variability." Precise, defensible operationalization of "compare the
    size of Delta with the within-class reconstruction spread" (reviewer's
    2nd validity check). Reported per named RAMAN_PEAKS band and as a global
    summary (e.g. median ratio, or ||mean_delta|| / ||spread||).

    Guards a zero/near-zero spread (would otherwise blow up to inf) by
    returning NaN at those features rather than a misleadingly huge ratio.
    """
    mean_delta = np.asarray(mean_delta, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    out = np.full_like(mean_delta, np.nan)
    valid = spread > 1e-9
    out[valid] = np.abs(mean_delta[valid]) / spread[valid]
    return out.astype(np.float32)


def fit_z_class_probe(z_mean_train, y_train, seed=42):
    """
    Fit sklearn LogisticRegression(z_mean -> y_bag) on one fold's TRAINING
    instances' z_mean (in-sample -- the same instances that fold's own CVAE
    encoder trained on; legitimate train-side use, the same naive-MIL
    discipline as every Tier 0 instance learner -- docs/mil_tiers_methodology.md
    §3). This is the "frozen trunk + fresh small classifier" pattern
    src/mil/attention.py uses (freeze, fit something new and small on top,
    evaluate out-of-sample), simplified to a single sklearn model on plain
    z_mean arrays rather than a trainable Keras attention model.

    Includes a StandardScaler fit on z_mean_train only (never validation) --
    a standard, minor addition for solver stability, not because of
    attention.py's cross-fold latent-scale problem (that mattered for ONE
    model sharing weights across folds; this probe fits a fresh model per
    fold, so no cross-fold scale assumption is made or needed here).

    Returns (scaler, model). class_weight='balanced', matching this
    project's own RF/CNN convention (configs/exp_501.yaml).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_mean_train)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
    model.fit(z_scaled, y_train)
    return scaler, model


def score_z_class_probe(scaler, model, z_mean_val):
    """
    predict_proba on a fold's VALIDATION (out-of-sample) z_mean, scaled by
    the SAME (train-fit) scaler. Returns (n_val, K) float32 -- structurally
    identical to Tier 0's own per-instance proba
    (load_and_predict_proba's output shape), deliberately NOT lifted to K+1
    -- there is no non-diagnostic-mass question here, only "how separable is
    z_mean by class," the same question Tier 0's classifiers answer for the
    raw spectrum. Feeds directly into
    aggregate_bag_scores_across_folds(rule='mean') for a bag-level number
    directly comparable to encoder_head's existing 0.7966.
    """
    z_scaled = scaler.transform(z_mean_val)
    return model.predict_proba(z_scaled).astype(np.float32)


def independent_classifier_flip_rate(X_real, y_true, recon_swap, model_path, model_type):
    """
    recon_swap: (n, n_features) = recon[i, 1-y_true[i], :] per instance --
    the class-SWAPPED counterfactual reconstruction (each instance decoded
    under the OPPOSITE of its own true label). Scores both X_real and
    recon_swap through the SAME already-trained, CVAE-independent baseline
    classifier via src.utils.statistical_evaluation.load_and_predict.

    Returns:
      pred_real (n,), pred_cf (n,)
      flip_to_target_rate: P(pred_cf == 1 - y_true) -- did editing z's label
          make the independent classifier see the OPPOSITE class (the
          reviewer's literal ask, read as "flip TO the intended target")
      flip_from_own_rate: P(pred_cf != pred_real) -- did the edit change the
          classifier's mind at all relative to its own (possibly
          already-wrong) call on the real spectrum -- robust to the
          baseline classifier's own real-data error rate; report BOTH, they
          need not agree when the baseline was already wrong on X_real.

    model_path/model_type: as returned by
    src.utils.statistical_evaluation.find_model_path -- e.g. an already
    fold-trained RF or XGBoost model (this project's strongest, most
    independent, non-CVAE-derived classifiers). Deliberately not defaulted
    to the CNN (unlike the bag-eval scripts' --reference-model default):
    this check needs the most trustworthy independent read on class
    identity available, and the CNN is this project's weakest classifier.
    """
    from src.utils.statistical_evaluation import load_and_predict

    y_true = np.asarray(y_true).astype(int)
    pred_real = load_and_predict(model_path, model_type, X_real)
    pred_cf = load_and_predict(model_path, model_type, recon_swap)
    target = 1 - y_true
    flip_to_target_rate = float((pred_cf == target).mean())
    flip_from_own_rate = float((pred_cf != pred_real).mean())
    return {'pred_real': pred_real, 'pred_cf': pred_cf,
            'flip_to_target_rate': flip_to_target_rate,
            'flip_from_own_rate': flip_from_own_rate}


def bootstrap_ci_mean(values, n_boot=1000, seed=42):
    """
    Plain percentile bootstrap CI on the mean of ``values`` (e.g. 59 per-bag
    ||Delta|| scalars) -- resample n=len(values) indices with replacement
    n_boot times, recompute the mean, take the 2.5/97.5 percentiles. Simpler
    than stratified_slide_bootstrap (which resamples PAIRED model-A/model-B
    decisions): this is a single-arm CI on a magnitude, not a two-model
    comparison.

    Returns {'mean': float, 'ci_lo': float, 'ci_hi': float}.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    return {'mean': float(values.mean()),
            'ci_lo': float(np.percentile(boot_means, 2.5)),
            'ci_hi': float(np.percentile(boot_means, 97.5))}


def band_assignment_report(mean_delta, wavenumbers, per_fold_mean_delta=None, spread=None,
                           peaks=None):
    """
    Maps the average Delta spectrum onto named Raman bands. Imports
    ``RAMAN_PEAKS`` directly from src.evaluation.synthetic_realism (not
    copied) unless a caller overrides ``peaks`` -- Phe 1003, Phosphodiester
    1087, Amide III 1250, CH2-twist/Lipid 1301, CH2/CH3 def. 1450, Amide II
    1550, Amide I 1655, all inside the 700-1800 cm^-1 crop this project's
    models actually see, already used on this exact 571-band representation
    and already cited in the paper (synthetic_realism.py's own comment:
    "paper §1 & §3"). Do NOT substitute
    src.preprocessing.baseline.RAMAN_REFERENCE_BANDS -- it includes
    ch_stretch_2930, OUTSIDE the 700-1800 crop, physically impossible for
    this model's Delta to say anything about.

    Reuses synthetic_realism.py's own nearest-single-index lookup idiom
    exactly (``np.argmin(np.abs(wn - shift))``, synthetic_realism.py:455) --
    no window/tolerance averaging: there is no existing precedent in this
    codebase for a window width, and introducing one would add a free
    parameter with nothing to justify its value.

    per_fold_mean_delta: optional (n_folds, n_features) array -- if given,
    also reports each named band's per-fold sign-agreement fraction (how
    many of the folds' own mean Delta agree in sign at that band), not just
    a pooled number.
    spread: optional (n_features,) within_class_reconstruction_spread
    (typically its 'pooled' field) -- if given, also reports the
    Delta-to-spread ratio at each named band.

    Returns {peak_name: {'wavenumber_matched': float, 'delta': float,
    'direction': str, 'effect_size': float or None,
    'fold_sign_agreement_frac': float or None}}.
    """
    if peaks is None:
        from src.evaluation.synthetic_realism import RAMAN_PEAKS
        peaks = RAMAN_PEAKS

    mean_delta = np.asarray(mean_delta)
    wavenumbers = np.asarray(wavenumbers)
    report = {}
    for name, target_wn in peaks.items():
        idx = int(np.argmin(np.abs(wavenumbers - target_wn)))
        delta_val = float(mean_delta[idx])
        direction = ("higher in IDH-Mutant" if delta_val > 0
                    else "higher in IDH-Wildtype" if delta_val < 0 else "no difference")

        effect_size = None
        if spread is not None:
            s = float(np.asarray(spread)[idx])
            effect_size = abs(delta_val) / s if s > 1e-9 else None

        fold_sign_agreement_frac = None
        if per_fold_mean_delta is not None:
            per_fold_mean_delta = np.asarray(per_fold_mean_delta)
            fold_vals = per_fold_mean_delta[:, idx]
            agree = np.sign(fold_vals) == np.sign(delta_val)
            fold_sign_agreement_frac = float(agree.mean())

        report[name] = {
            'wavenumber_matched': float(wavenumbers[idx]),
            'delta': delta_val,
            'direction': direction,
            'effect_size': effect_size,
            'fold_sign_agreement_frac': fold_sign_agreement_frac,
        }
    return report
