"""
Per-feature std ratio of synthetic vs. real data, within class, for the 3
parallel CVAE continuation experiments (exp_601 constant-beta continuation,
exp_605/exp_606 windowed beta schedules) -- the same ad hoc diagnostic used
earlier this session on the Suite 600 beta sweep (exp_601-604) to distinguish
"more diverse because reconstruction genuinely improved" from "more diverse
because the decoder got noisier."

For each fold and class: ratio = mean_per_feature(std(X_synth)) /
mean_per_feature(std(X_real)), using that fold's REAL TRAINING split (the
data the CVAE was actually trained to resemble) as the reference -- not the
pairwise-distance diversity_ratio_synth_real in
src/evaluation/synthetic_realism.py::compute_diversity, which is a more
rigorous but differently-scaled statistic. A ratio of 1.0 means the
synthetic pool reproduces real within-class variance exactly; well below 1.0
is the collapse signature; a ratio near or above 1.0 that comes with
genuinely LOWER reconstruction loss (see plot_beta_window_continuation.py)
reads as real improvement rather than noise, since noise alone would raise
variance without lowering reconstruction error.

Usage:
    python scripts/compute_synth_std_ratio.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_manager import ConfigManager
from pipeline._common import make_loader, num_classes_of

EXPERIMENTS = [
    ("exp_601 (beta=1.0, constant, 200 epochs)", "exp_601", "results/exp_601/synthetic_extended"),
    ("exp_605 (beta window: 1.25/1 alt., 200 epochs)", "exp_605", "results/exp_605/synthetic"),
    ("exp_606 (beta window: 0.75/1 alt., 200 epochs)", "exp_606", "results/exp_606/synthetic"),
]


def _std_ratio_for_experiment(config_name, synthetic_dir):
    config = ConfigManager.load_config(config_name).to_dict()
    nc = num_classes_of(config)
    loader = make_loader(config)

    per_fold_per_class = []  # list of dicts: {class: ratio}
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        X_real, y_real = fold_data['train']
        X_real = np.asarray(X_real, dtype=np.float32)
        y_real = np.asarray(y_real)
        if y_real.size and np.min(y_real) == 1:
            y_real = y_real - 1
        if nc == 2:
            y_real = np.where(y_real < 3, 0, 1)

        synth_path = Path(synthetic_dir)
        X_synth = np.load(synth_path / f"{fold_name}_X_synth.npy")
        y_synth = np.load(synth_path / f"{fold_name}_y_synth.npy")

        ratios = {}
        for c in sorted(np.unique(y_real).tolist()):
            real_std = X_real[y_real == c].std(axis=0).mean()
            synth_std = X_synth[y_synth == c].std(axis=0).mean()
            ratios[c] = float(synth_std / (real_std + 1e-10))
        per_fold_per_class.append(ratios)

    return per_fold_per_class


def main():
    for label, config_name, synthetic_dir in EXPERIMENTS:
        print(f"\n{label}")
        per_fold = _std_ratio_for_experiment(config_name, synthetic_dir)
        classes = sorted(per_fold[0].keys())
        for c in classes:
            vals = [fold_ratios[c] for fold_ratios in per_fold]
            print(f"  class {c}: per-fold ratios = {[round(v, 4) for v in vals]}  "
                 f"mean = {np.mean(vals):.4f}")
        overall = np.mean([v for fold_ratios in per_fold for v in fold_ratios.values()])
        print(f"  overall mean (both classes, 5 folds): {overall:.4f}")


if __name__ == "__main__":
    main()
