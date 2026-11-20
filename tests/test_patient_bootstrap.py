"""Patient-level (cluster) stratified bootstrap: regression anchor against the
published slide-level bootstrap, plus the cluster/stratum invariants.

The anchor: with singleton clusters in positional order the cluster bootstrap
must reproduce ``bag_level_bootstrap`` / ``bag_level_bootstrap_cbr``
bit-for-bit (same RNG call order, exact 0/1 sums), so the patient-level code
path is the slide-level one with only the resampling unit changed.
"""
import numpy as np
import pytest

from src.utils.statistical_evaluation import (
    bag_level_bootstrap,
    bag_level_bootstrap_cbr,
    bag_level_patient_bootstrap,
    bag_level_patient_bootstrap_cbr,
    paired_cluster_test,
    patient_id_from_slide,
    patient_ids_from_slides,
    stratified_cluster_bootstrap,
)


def _cohort(seed=0):
    """59 sorted slide ids over 46 patients (multi-slide pattern of the real
    cohort: 37 x 1, 6 x 2, 2 x 3, 1 x 4), one binary label per patient."""
    rng = np.random.default_rng(seed)
    sizes = [1] * 37 + [2] * 6 + [3] * 2 + [4]
    slides, labels = [], []
    for p, k in enumerate(sizes):
        y = int(rng.integers(0, 2))
        for j in range(k):
            slides.append(f"HF-{100 + p}_S{j}_1.h5_{j}")
            labels.append(y)
    order = np.argsort(slides)
    slides = np.array(slides)[order]
    y = np.array(labels)[order]
    pred_A = np.where(rng.random(len(y)) < 0.75, y, 1 - y)
    pred_B = np.where(rng.random(len(y)) < 0.80, y, 1 - y)
    return slides, y, pred_A, pred_B


def test_patient_id_rule():
    assert patient_id_from_slide("HF-1002_V2AT_1.h5_1") == "HF-1002"
    assert patient_id_from_slide("HF-2849_VIT2_1.h5_4") == "HF-2849"
    assert list(patient_ids_from_slides(["HF-1_a", "HF-10_b"])) == ["HF-1", "HF-10"]


def test_singleton_clusters_bit_identical_to_bag_level_bootstrap():
    slides, y, pa, pb = _cohort()
    old = bag_level_bootstrap(pa, pb, y, n_boot=1000, seed=42)
    new = bag_level_patient_bootstrap(pa, pb, y, slides, n_boot=1000, seed=42, cluster_unit='slide')
    for o, n in zip(old, new):
        assert np.array_equal(o, n)


def test_singleton_clusters_bit_identical_to_cbr():
    slides, y, pa, _ = _cohort(1)
    old = bag_level_bootstrap_cbr(pa, y, n_boot=1000, seed=42)
    new = bag_level_patient_bootstrap_cbr(pa, y, slides, n_boot=1000, seed=42, cluster_unit='slide')
    assert np.array_equal(old, new)


def test_patients_never_split_and_class_counts_fixed():
    # one unit-valued indicator per patient: theta * |draw| must be an integer
    # multiple pattern consistent with whole patients only
    slides, y, _, _ = _cohort(2)
    pats = patient_ids_from_slides(slides)
    # value 1 on exactly one slide of the 4-slide patient, 0 elsewhere
    big = [p for p in np.unique(pats) if np.sum(pats == p) == 4][0]
    idx = np.where(pats == big)[0]
    v = np.zeros(len(slides))
    v[idx[0]] = 1.0
    size = np.ones(len(slides))
    cnt, _, _ = stratified_cluster_bootstrap(size, None, y, pats, n_boot=300, seed=3)
    assert np.all(cnt == 1.0)
    # the drawn-slide total is a sum of whole patients' slide counts and the
    # number of drawn patients per class is fixed: check via patient counts
    per_patient = np.zeros(len(slides))
    first = {}
    for i, p in enumerate(pats):
        if p not in first:
            first[p] = i
            per_patient[i] = 1.0
    theta_pat, _, _ = stratified_cluster_bootstrap(per_patient, None, y, pats, n_boot=300, seed=3)
    theta_v, _, _ = stratified_cluster_bootstrap(v, None, y, pats, n_boot=300, seed=3)
    # patients drawn / slides drawn; patients drawn is constant = 46
    n_pat = len(first)
    slides_drawn = n_pat / theta_pat
    assert np.allclose(slides_drawn, np.round(slides_drawn))
    # v counts copies of the big patient; its slides are never split, so
    # copies * 4 <= slides drawn and copies is an integer
    copies = theta_v * slides_drawn
    assert np.allclose(copies, np.round(copies))


def test_cluster_spanning_two_strata_raises():
    with pytest.raises(ValueError):
        stratified_cluster_bootstrap([1, 0, 1], None, [0, 1, 1], ["A", "A", "B"], n_boot=5)


def test_deterministic_and_paired_diff():
    slides, y, pa, pb = _cohort(4)
    a1, b1, d1 = bag_level_patient_bootstrap(pa, pb, y, slides, n_boot=200, seed=7)
    a2, b2, d2 = bag_level_patient_bootstrap(pa, pb, y, slides, n_boot=200, seed=7)
    assert np.array_equal(d1, d2)
    assert np.allclose(d1, b1 - a1)


def test_patient_bootstrap_wider_when_patients_correlated():
    # all slides of a patient share one outcome -> patient resampling must not
    # be narrower than slide resampling (on average it is wider)
    slides, y, _, _ = _cohort(5)
    pats = patient_ids_from_slides(slides)
    rng = np.random.default_rng(0)
    pat_ok = {p: rng.random() < 0.7 for p in np.unique(pats)}
    pred = np.array([y[i] if pat_ok[p] else 1 - y[i] for i, p in enumerate(pats)])
    s = bag_level_patient_bootstrap_cbr(pred, y, slides, n_boot=4000, seed=1, cluster_unit='slide')
    p = bag_level_patient_bootstrap_cbr(pred, y, slides, n_boot=4000, seed=1, cluster_unit='patient')
    assert np.std(p) >= np.std(s) * 0.98


def test_paired_cluster_test_delta_and_counts():
    slides, y, _, _ = _cohort(6)
    d = np.random.default_rng(1).normal(0.01, 0.02, len(slides))
    r = paired_cluster_test(d, slides, y, n_boot=500, seed=42)
    assert r['delta'] == pytest.approx(d.mean())
    assert r['n_units'] == 59 and r['n_clusters'] == 46
    assert sum(r['n_clusters_per_class'].values()) == 46
    assert r['ci_lo'] <= r['delta'] <= r['ci_hi']
    # matched subset: strata/clusters built only from the slides passed
    keep = np.arange(len(slides)) % 5 != 0
    r2 = paired_cluster_test(d[keep], slides[keep], y[keep], n_boot=200, seed=42)
    assert r2['n_units'] == keep.sum()
    assert r2['n_clusters'] == len(set(patient_ids_from_slides(slides[keep])))


def test_signflip_is_per_patient():
    # a patient whose slides have opposite d cancels exactly under a per-patient
    # flip: two patients with D_p = 0 give T == 0 for every sign draw
    slides = np.array(["HF-1_a", "HF-1_b", "HF-2_a", "HF-2_b"])
    y = np.array([0, 0, 1, 1])
    d = np.array([0.5, -0.5, 0.3, -0.3])
    r = paired_cluster_test(d, slides, y, n_boot=200, seed=0)
    assert r['delta'] == 0.0
    assert r['p_signflip'] == 1.0
    assert r['ci_lo'] == r['ci_hi'] == 0.0
