"""
Pairwise bag-level bootstrap between latent dimensions (exp_701: d=16,
exp_608: d=32, exp_702: d=64; beta=0.5, everything else identical).

For each regimen (TS/TR, TSR/TR) and each of the four benchmark models, tests
d=16 vs d=32 and d=32 vs d=64: one majority-vote decision per bag (slide) from
each experiment's own fold models, then the project's class-stratified
bag-level bootstrap on the PAIRED bag decisions
(src/utils/statistical_evaluation.py::bag_level_bootstrap, 1000 iterations,
seed 42 -- the same function, settings and Delta convention as
pipeline/evaluation_stage.py; Delta = accuracy(larger d) - accuracy(smaller d)).
4 models x 2 regimens x 2 comparisons = 16 tests, no multiplicity correction
(consistent with the paper's stated stance).

Validation spectra are identical across the three experiments (same processed
file and CV splits), so the bag sets align exactly; this is asserted. Each
recomputed bag accuracy is also checked against the accuracy already reported
by that experiment's stats-eval JSON.

Usage:
    python scripts/latent_dim_pairwise_bootstrap.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation_stage import (  # noqa: E402
    MODEL_LABELS, MODEL_SPECS, _build_fold_val_sets, _gather_predictions,
)
from src.utils.config_manager import ConfigManager  # noqa: E402
from src.utils.statistical_evaluation import (  # noqa: E402
    bag_decisions, bag_level_bootstrap, summarize_bootstrap_diffs,
)

EXPS = [('exp_701', 16), ('exp_608', 32), ('exp_702', 64)]
REGIMENS = [('TSTR', 'models_tstr_dir', 'TS/TR'), ('TSRTR', 'models_tsrtr_dir', 'TSR/TR')]
N_BOOT = 1000
SEED = 42
OUT = PROJECT_ROOT / 'results' / 'latent_dim_bootstrap' / 'pairwise_latent_dim_bootstrap.json'


def _reported_bag_acc(exp, regimen, model_label):
    pat = str(PROJECT_ROOT / 'results' / exp / 'statistical_eval'
              / f'statistical_evaluation_TRTR_vs_{regimen}_*.json')
    files = sorted(glob.glob(pat), key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    res = json.load(open(files[-1]))
    for r in res['results']:
        if r['model'] == model_label:
            return r[f'bag_acc_{regimen}']
    raise KeyError((exp, regimen, model_label))


def main():
    cfgs = {e: ConfigManager.load_config(e).to_dict() for e, _ in EXPS}
    assert len({c['paths']['processed_file'] for c in cfgs.values()}) == 1
    assert len({c['paths']['splits_file'] for c in cfgs.values()}) == 1
    folds = _build_fold_val_sets(cfgs['exp_608'])

    # bag decisions per (regimen, model, exp)
    dec = {}
    ref_ids = ref_true = None
    for regimen, path_key, _ in REGIMENS:
        for prefix, ext, model_type in MODEL_SPECS:
            for exp, _d in EXPS:
                model_dir = cfgs[exp]['paths'][path_key]
                preds, y_true, slide_ids = _gather_predictions(folds, model_dir, prefix, ext, model_type)
                bag_ids, bag_pred, bag_true = bag_decisions(preds, slide_ids, y_true)
                if ref_ids is None:
                    ref_ids, ref_true = np.asarray(bag_ids), np.asarray(bag_true)
                assert np.array_equal(np.asarray(bag_ids), ref_ids), 'bag sets differ across experiments'
                assert np.array_equal(np.asarray(bag_true), ref_true)
                dec[(regimen, prefix, exp)] = np.asarray(bag_pred)
                acc = float((np.asarray(bag_pred) == ref_true).mean())
                rep = _reported_bag_acc(exp, regimen, MODEL_LABELS[prefix])
                assert abs(acc - rep) < 1e-9, f'{exp} {regimen} {prefix}: recomputed {acc} != reported {rep}'
                print(f'  {regimen:5s} {prefix:7s} {exp}: bag_acc={acc:.4f} (matches reported)', flush=True)

    rows = []
    for regimen, _, reg_label in REGIMENS:
        for prefix, _, _ in MODEL_SPECS:
            for (e_a, d_a), (e_b, d_b) in zip(EXPS[:-1], EXPS[1:]):
                pa, pb = dec[(regimen, prefix, e_a)], dec[(regimen, prefix, e_b)]
                _, _, diffs = bag_level_bootstrap(pa, pb, ref_true, n_boot=N_BOOT, seed=SEED)
                s = summarize_bootstrap_diffs(diffs)
                acc_a, acc_b = float((pa == ref_true).mean()), float((pb == ref_true).mean())
                rows.append({
                    'regimen': reg_label, 'model': MODEL_LABELS[prefix],
                    'comparison': f'd={d_a} vs d={d_b}', 'baseline_exp': e_a, 'candidate_exp': e_b,
                    'acc_baseline': acc_a, 'acc_candidate': acc_b, 'delta': acc_b - acc_a,
                    'ci_lo': s['ci_diff_lo'], 'ci_hi': s['ci_diff_hi'],
                    'p_bootstrap': s['p_bootstrap'], 'significant_95': s['significant_95'],
                    'n_bags': int(len(ref_true)), 'n_boot': N_BOOT, 'seed': SEED,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'tests': rows,
                               'bag_ids': ref_ids.tolist(), 'bag_true': ref_true.tolist(),
                               'bag_pred': {f'{k[0]}|{k[1]}|{k[2]}': v.tolist() for k, v in dec.items()}},
                              indent=1), encoding='utf-8')
    print(f'\nWrote {OUT}\n')
    print(f"{'regimen':7s}{'model':13s}{'comparison':16s}{'acc_a':>7s}{'acc_b':>7s}{'delta':>8s}  "
          f"{'95% CI':>18s}{'p':>8s}  sig")
    for r in rows:
        print(f"{r['regimen']:7s}{r['model']:13s}{r['comparison']:16s}{r['acc_baseline']:7.3f}"
              f"{r['acc_candidate']:7.3f}{r['delta']:+8.3f}  "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]".ljust(66)
              + f"{r['p_bootstrap']:8.3f}  {'*' if r['significant_95'] else ''}")


if __name__ == '__main__':
    main()
