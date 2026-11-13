"""
Patch the bag-level AUROC of Classification by Reconstruction (CbR) into the
existing statistical-evaluation JSONs.

CbR is a binary classifier whose per-spectrum decision is the argmin of the class-
conditional reconstruction MSE. The continuous score behind that decision is the
normalised margin s = (mse_0 - mse_1) / (mse_0 + mse_1), positive when class 1
reconstructs better. The bag score is the mean of s over the bag's spectra (the
analogue of the mean-pooled class-1 probability of the other classifiers) and the
AUROC ranks the n=59 bag scores against the bag labels. The AUROC of the vote
share (fraction of a bag's spectra predicted class 1) is stored as a sensitivity
check: the majority vote is its 0.5 threshold.

Predictions come from ``pipeline.evaluation_stage._cbr_predictions``, the same code
path the pipeline uses; the recomputed bag accuracy is asserted against the
``cbr_bag_accuracy`` already stored in each JSON before anything is written.

Usage
-----
    python scripts/patch_cbr_auroc.py --experiments exp_608 exp_501 ...
    python scripts/patch_cbr_auroc.py --dry-run
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / 'results'
EXPERIMENTS = ['exp_501', 'exp_601', 'exp_607', 'exp_608', 'exp_609',
               'exp_701', 'exp_702', 'exp_703', 'exp_704',
               'exp_705', 'exp_706', 'exp_707', 'exp_708',
               'exp_2.607', 'exp_2.608', 'exp_2.609', 'exp_4.608']
AUROC_KEYS = ('bag_auroc', 'bag_auroc_vote_share')
ACC_TOL = 1e-9


def _latest_json(exp, candidate):
    pat = str(RESULTS / exp / 'statistical_eval'
              / f'statistical_evaluation_TRTR_vs_{candidate}_*.json')
    files = sorted(glob.glob(pat), key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiments', nargs='+', default=EXPERIMENTS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='recompute even if bag_auroc is present')
    args = ap.parse_args()

    from src.utils.experiment_utils import setup_experiment
    from pipeline.evaluation_stage import _build_fold_val_sets, _cbr_predictions, _cbr_bag_auroc
    from src.utils.statistical_evaluation import compute_full_metrics

    cache = {}   # weights_dir -> (auroc, auroc_vote, bag_accuracy)
    summary = []
    for exp in args.experiments:
        jsons = {c: _latest_json(exp, c) for c in ('TSTR', 'TSRTR')}
        if not any(jsons.values()):
            summary.append(f'  {exp}: SKIP (no stats-eval JSON)')
            continue
        folds = cfg = None
        for cand, jpath in jsons.items():
            if not jpath:
                continue
            payload = json.load(open(jpath, encoding='utf-8'))
            cbr = payload.get('cbr')
            if not cbr:
                summary.append(f'  {exp}/{cand}: no cbr block')
                continue
            if 'bag_auroc' in cbr and not args.force:
                summary.append(f'  {exp}/{cand}: already patched (AUC={cbr["bag_auroc"]:.3f})')
                continue
            wd = payload.get('cvae_weights_dir')
            if wd not in cache:
                if folds is None:
                    cfg, _ = setup_experiment(exp)
                    folds = _build_fold_val_sets(cfg)
                print(f'\n[{exp}/{cand}] CbR scores (weights: {wd})', flush=True)
                out = _cbr_predictions(folds, cfg, folds[0]['X'].shape[1], wd, return_scores=True)
                if out is None:
                    summary.append(f'  {exp}/{cand}: CbR weights missing -> skipped')
                    continue
                preds, y, sid, s = out
                bag_acc = compute_full_metrics(y, preds, slide_ids=sid)['bag_accuracy']
                cache[wd] = (*_cbr_bag_auroc(s, y, sid), bag_acc)
            auc, auc_vote, bag_acc = cache[wd]
            stored = cbr.get('cbr_bag_accuracy')
            if stored is not None and abs(bag_acc - stored) > ACC_TOL:
                summary.append(f'  {exp}/{cand}: !! bag-acc mismatch stored={stored:.4f} '
                               f'recomputed={bag_acc:.4f} -> NOT written')
                continue
            summary.append(f'  {exp}/{cand}: AUC={auc:.3f} (vote share {auc_vote:.3f}) '
                           f'bag_acc={bag_acc:.4f}{" [DRY-RUN]" if args.dry_run else ""}')
            if args.dry_run:
                continue
            entries = [cbr] + [r for r in payload.get('results', [])
                               if r.get('model') == 'Classification_by_Reconstruction']
            for e in entries:
                e['bag_auroc'], e['bag_auroc_vote_share'] = auc, auc_vote
            json.dump(payload, open(jpath, 'w', encoding='utf-8'), indent=2)

    print('\n=== CbR AUROC patch summary ===')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
