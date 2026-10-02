"""
Decomposes REAL data's within-class variance into a within-slide component
("noise"-like: local/intra-tumor heterogeneity between pixels of the SAME
patient's slide, the closest thing this dataset has to a replicate measurement)
and a between-slide component ("signal"-like: genuine inter-patient biological
difference) -- to give the synth/real std ratio (compute_synth_std_ratio.py) a
reference point: is the CVAE's shortfall (ratio well under 1.0) because it's
failing to reproduce mostly-irrelevant intra-slide noise, or because it's
failing to reproduce genuine inter-patient variation the synthetic pool would
need for realistic downstream (TSTR/TSRTR) use?

Important framing: this is NOT an instrument-noise measurement (no literal
repeated measurements of the same physical point exist in this dataset) --
within-slide spread still contains real intra-tumor cellular/spatial
heterogeneity, so it's an UPPER BOUND on "pure noise," not a clean estimate
of it. Still a meaningful reference: the CVAE conditions only on class label
(not patient/slide identity), so it structurally has no way to reproduce
between-slide (patient-specific) structure at generation time regardless of
training quality -- the within-slide component is the closest thing to an
architecture-appropriate ceiling for what this specific model COULD capture.

Per-feature decomposition (law of total variance, plug-in sample moments --
a simplification consistent with every other diagnostic ratio used this
session, not a bias-corrected ANOVA estimator):
    Var_total(feature) = mean over samples of (x - class_mean)^2
    Var_within(feature) = weighted mean over slides of each slide's own
                           variance (weighted by slide's sample count)
    Var_between(feature) = Var_total - Var_within   (clipped at 0)

Usage:
    python scripts/compute_within_vs_between_slide_variance.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_manager import ConfigManager
from pipeline._common import make_loader, num_classes_of

CONFIG_NAME = "exp_601"  # any Suite 600 config works -- identical real data/preprocessing


def _within_between_variance(X, groups):
    """Per-feature (Var_within, Var_between), pooled over all slides in X."""
    n_total, n_features = X.shape
    overall_mean = X.mean(axis=0)
    var_total = ((X - overall_mean) ** 2).mean(axis=0)

    weighted_within_sum = np.zeros(n_features)
    for slide_id in np.unique(groups):
        idx = np.where(groups == slide_id)[0]
        if len(idx) < 2:
            continue  # a single-pixel slide has no within-slide variance to contribute
        slide_var = X[idx].var(axis=0)  # population variance within this slide
        weighted_within_sum += len(idx) * slide_var
    var_within = weighted_within_sum / n_total
    var_between = np.clip(var_total - var_within, 0, None)
    return var_total, var_within, var_between


def main():
    config = ConfigManager.load_config(CONFIG_NAME).to_dict()
    nc = num_classes_of(config)
    loader = make_loader(config)

    per_fold_rows = []
    for fold_data in loader:
        X_train, y_train = fold_data['train']
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)
        groups = fold_data['train_groups']
        if y_train.size and np.min(y_train) == 1:
            y_train = y_train - 1
        if nc == 2:
            y_train = np.where(y_train < 3, 0, 1)

        for c in sorted(np.unique(y_train).tolist()):
            mask = y_train == c
            var_total, var_within, var_between = _within_between_variance(
                X_train[mask], groups[mask]
            )
            std_total = np.sqrt(var_total).mean()
            std_within = np.sqrt(var_within).mean()
            std_between = np.sqrt(var_between).mean()
            per_fold_rows.append({
                'fold': fold_data['fold_name'], 'class': c,
                'std_total': std_total, 'std_within': std_within, 'std_between': std_between,
                'within_frac': std_within / std_total, 'between_frac': std_between / std_total,
            })
            print(f"  {fold_data['fold_name']} class {c}: "
                 f"std_total={std_total:.4f}  std_within(noise-like)={std_within:.4f} "
                 f"({std_within/std_total:.1%})  std_between(signal-like)={std_between:.4f} "
                 f"({std_between/std_total:.1%})")

    within_fracs = [r['within_frac'] for r in per_fold_rows]
    between_fracs = [r['between_frac'] for r in per_fold_rows]
    print(f"\nOverall (mean over {len(per_fold_rows)} fold x class rows):")
    print(f"  within-slide (noise-like) fraction of real std:  {np.mean(within_fracs):.1%}")
    print(f"  between-slide (signal-like) fraction of real std: {np.mean(between_fracs):.1%}")


if __name__ == "__main__":
    main()
