"""
Direct bag-level bootstrap between two TS/TR runs that differ only in the size of the
synthetic training pool: exp_608 (20,000 spectra per class, 40,000 per fold) vs.
exp_708 (40,000 per class, 80,000 per fold). Same CVAE decoders, same models and
hyperparameters, same real validation folds.

Method identical to scripts/latent_dim_pairwise_bootstrap.py: one majority-vote
decision per bag from each experiment's own fold models, then the project's
class-stratified bag-level bootstrap on the PAIRED decisions
(src/utils/statistical_evaluation.py::bag_level_bootstrap, 1000 iterations, seed 42);
Delta = accuracy(exp_708) - accuracy(exp_608). 4 models = 4 tests, no multiplicity
correction. Recomputed bag accuracies are asserted against each experiment's
stats-eval JSON.

Usage:
    python scripts/synth_pool_pairwise_bootstrap.py
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

BASE, CAND = ('exp_608', '40k'), ('exp_708', '80k')
N_BOOT, SEED = 1000, 42
OUT = PROJECT_ROOT / 'results' / 'synth_pool_bootstrap' / 'pairwise_pool_size_bootstrap.json'


def _reported_bag_acc(exp, model_label):
    pat = str(PROJECT_ROOT / 'results' / exp / 'statistical_eval'
              / 'statistical_evaluation_TRTR_vs_TSTR_*.json')
    files = sorted(glob.glob(pat), key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    for r in json.load(open(files[-1]))['results']:
        if r['model'] == model_label:
            return r['bag_acc_TSTR']
    raise KeyError((exp, model_label))


def main():
    cfgs = {e: ConfigManager.load_config(e).to_dict() for e, _ in (BASE, CAND)}
    assert cfgs[BASE[0]]['paths']['processed_file'] == cfgs[CAND[0]]['paths']['processed_file']
    folds = _build_fold_val_sets(cfgs[BASE[0]])

    rows, store = [], {}
    for prefix, ext, model_type in MODEL_SPECS:
        dec = {}
        for exp, _ in (BASE, CAND):
            preds, y_true, slide_ids = _gather_predictions(
                folds, cfgs[exp]['paths']['models_tstr_dir'], prefix, ext, model_type)
            bag_ids, bag_pred, bag_true = bag_decisions(preds, slide_ids, y_true)
            dec[exp] = (np.asarray(bag_ids), np.asarray(bag_pred), np.asarray(bag_true))
            acc = float((dec[exp][1] == dec[exp][2]).mean())
            rep = _reported_bag_acc(exp, MODEL_LABELS[prefix])
            assert abs(acc - rep) < 1e-9, f'{exp} {prefix}: recomputed {acc} != reported {rep}'
            print(f'  {prefix:7s} {exp}: bag_acc={acc:.4f} (matches reported)', flush=True)
        assert np.array_equal(dec[BASE[0]][0], dec[CAND[0]][0])
        assert np.array_equal(dec[BASE[0]][2], dec[CAND[0]][2])
        pa, pb, truth = dec[BASE[0]][1], dec[CAND[0]][1], dec[BASE[0]][2]
        _, _, diffs = bag_level_bootstrap(pa, pb, truth, n_boot=N_BOOT, seed=SEED)
        s = summarize_bootstrap_diffs(diffs)
        rows.append({
            'regimen': 'TS/TR', 'model': MODEL_LABELS[prefix],
            'comparison': f'{BASE[1]} vs {CAND[1]}', 'baseline_exp': BASE[0], 'candidate_exp': CAND[0],
            'acc_baseline': float((pa == truth).mean()), 'acc_candidate': float((pb == truth).mean()),
            'delta': float((pb == truth).mean() - (pa == truth).mean()),
            'ci_lo': s['ci_diff_lo'], 'ci_hi': s['ci_diff_hi'], 'p_bootstrap': s['p_bootstrap'],
            'significant_95': s['significant_95'], 'n_bags': int(len(truth)),
            'n_boot': N_BOOT, 'seed': SEED})
        store[prefix] = {BASE[0]: pa.tolist(), CAND[0]: pb.tolist()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'tests': rows, 'bag_pred': store}, indent=1), encoding='utf-8')
    print(f'\nWrote {OUT}\n')
    for r in rows:
        print(f"{r['model']:13s} {r['comparison']:11s} {r['acc_baseline']:.3f} -> {r['acc_candidate']:.3f}  "
              f"delta={r['delta']:+.3f}  CI[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_bootstrap']:.3f}"
              f"{'  *' if r['significant_95'] else ''}")


if __name__ == '__main__':
    main()
