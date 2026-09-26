r"""
Generate the extended per-experiment results tables for paper/appendix.tex.

Reads the latest statistical_eval JSONs (schema:
statistical_evaluation_TRTR_vs_{TSTR,TSRTR}_<ts>.json) for each experiment and
emits one LaTeX table per experiment, written to
paper/appendix_extended_results.tex, which paper/appendix.tex \input{}s in
place -- this script does NOT own the \section{} placement or the hand-written
appendices (A: baseline correction, C: CV schema) around it, only the
generated middle section, and it always writes to the same file so a re-run
updates the one file the build actually uses (previously this wrote to a
disconnected "appendix (5).tex" that \input{appendix} never read -- a stale
generator whose output was never wired into the document).

Fixed columns: Model | Regimen | Accuracy | Delta | 95% CI
Configurable extra columns: defined by METRIC_COLUMNS below.

Bag-level only (docs/MIL_REFRAME_PLAN.md §7): every column in these tables is
now a genuine per-bag decision statistic -- one majority-vote decision per
bag, standard metrics over the n_bags decisions, single unit of analysis
throughout a row. This replaces the pre-Stage-2 tables, which mixed two units
in one row: bag-level sensitivity/specificity for multiclass experiments were
derived from the slide-*count* confusion matrix while the neighbouring F1
column was a within-bag instance average -- gen_appendix_tables.py:53-81's old
_derive_sens_spec (docs/MIL_REFRAME_PLAN.md §7's third bug). There is nothing
left to derive: bag_sensitivity/bag_specificity are computed directly (binary)
or via the (now-always-bag-level) confusion matrix (multiclass), by the same
function either way.

To change the columns re-run the script with a different METRIC_COLUMNS list:
    python paper/gen_appendix_tables.py

Re-run this after stats_eval (and mil_eval, for the Tier-0 comparison) finish
to refresh the tables. Running it against stale, pre-Stage-2 JSONs is safe --
they carry no bag_* keys, so every cell reads NA ('--') rather than a wrong
number; this is what the "unavailable" summary line at the end reports.
"""
import json
import glob
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REGISTRY = os.path.join(ROOT, 'configs', 'EXPERIMENT_REGISTRY.json')
OUT = os.path.join(ROOT, 'paper', 'appendix_extended_results.tex')

# Experiments to include.
EXPERIMENTS = ['exp_201', 'exp_202', 'exp_203', 'exp_204', 'exp_205', 'exp_206',
               'exp_302', 'exp_303', 'exp_401', 'exp_402', 'exp_403']

MODEL_ORDER = ['Apollo', 'RF_Standard', 'XGBoost', '1D_CNN']
MODEL_LABEL = {'Apollo': 'RF + SVM', 'RF_Standard': 'RF',
               'XGBoost': 'XGBoost', '1D_CNN': '1D-CNN'}

# ── Configurable metric columns ───────────────────────────────────────────────
# Each entry is (json_key, latex_header). json_key is looked up in
# full_metrics_<label> for the 4 models and at the top level of the cbr dict
# for the CbR row. 'bag_accuracy' is already the fixed primary column; don't
# repeat it here. Available bag-level keys: bag_accuracy, bag_balanced_accuracy,
# bag_f1_macro, bag_precision_macro, bag_recall_macro, bag_kappa,
# bag_sensitivity, bag_specificity (binary only), bag_auroc, bag_auprc,
# confusion_matrix (not suitable as a column, always bag-level now).
METRIC_COLUMNS = [
    ('bag_f1_macro',    r'Macro F1'),
    ('bag_sensitivity', r'Sensitivity'),
    ('bag_specificity', r'Specificity'),
]
# ─────────────────────────────────────────────────────────────────────────────

NA = '--'


def _derive_bag_sens_spec(fm):
    """Derive bag_sensitivity/bag_specificity from the bag-level confusion
    matrix, for multiclass experiments where _slide_level_metrics only writes
    them directly for binary tasks. Safe post-Stage-2 specifically because
    ``confusion_matrix`` is unconditionally bag-level now (majority-vote
    decisions, n_bags total) -- there is no other unit of analysis in this
    dict to accidentally mix it with.
        bag_sensitivity_c = bag_recall_c = CM[c,c] / row_sum[c] -> macro mean
            equals bag_recall_macro, already computed directly; this fills in
            the per-class breakdown that's otherwise binary-only.
        bag_specificity_c = TN_c / (TN_c + FP_c), one-vs-rest decomposition.
    """
    cm = fm.get('confusion_matrix')
    if not cm:
        return
    cm = np.array(cm, dtype=float)
    n = cm.sum()
    n_classes = len(cm)
    sens_list, spec_list = [], []
    for c in range(n_classes):
        row_sum = cm[c].sum()
        col_sum = cm[:, c].sum()
        tp = cm[c, c]
        if row_sum > 0:
            sens_list.append(tp / row_sum)
        tn = n - row_sum - col_sum + tp
        fp = col_sum - tp
        if (tn + fp) > 0:
            spec_list.append(tn / (tn + fp))
    if fm.get('bag_sensitivity') is None and sens_list:
        fm['bag_sensitivity'] = float(np.mean(sens_list))
    if fm.get('bag_specificity') is None and spec_list:
        fm['bag_specificity'] = float(np.mean(spec_list))


def latest(exp, candidate):
    """Latest statistical_evaluation_TRTR_vs_<candidate>_*.json for an experiment."""
    pat = os.path.join(RESULTS, exp, 'statistical_eval',
                       f'statistical_evaluation_TRTR_vs_{candidate}_*.json')
    files = glob.glob(pat)
    if not files:
        return None
    files.sort(key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    return json.load(open(files[-1]))


def fnum(x, signed=False):
    if x is None or x == '':
        return NA
    s = f'{x:+.3f}' if signed else f'{x:.3f}'
    return f'${s}$'


def fci(lo, hi):
    if lo is None or hi is None:
        return NA
    return f'$[{lo:+.3f}, {hi:+.3f}]$'


def acc_of(r, label):
    """Bag-level accuracy under `label`."""
    if r is None:
        return None
    return r.get(f'bag_acc_{label}')


def metrics_of(r, label):
    """Return list of formatted metric values from full_metrics_<label>."""
    fm = dict((r or {}).get(f'full_metrics_{label}') or {})
    _derive_bag_sens_spec(fm)
    return [fnum(fm.get(key)) for key, _ in METRIC_COLUMNS]


def cbr_metrics_of(cbr):
    """Return list of formatted metric values from the flat cbr dict.

    bag_sensitivity/bag_specificity are derived from
    full_metrics.confusion_matrix when not present as flat keys (always the
    case for current JSONs).
    """
    fm = dict((cbr.get('full_metrics') or {}))
    _derive_bag_sens_spec(fm)
    flat = dict(cbr)
    for key in ('bag_sensitivity', 'bag_specificity'):
        if flat.get(key) is None and fm.get(key) is not None:
            flat[key] = fm[key]
    return [fnum(flat.get(key)) for key, _ in METRIC_COLUMNS]


def rows_for_model(m, tstr_res, tsrtr_res):
    """Return three regimen rows: (regimen, acc, delta, ci, *metric_cols)."""
    def find(res, model):
        if not res:
            return None
        for r in res['results']:
            if r['model'] == model:
                return r
        return None

    rt = find(tstr_res, m)
    rr = find(tsrtr_res, m)

    def star(r):
        return '*' if (r and r.get('bag_significant_95')) else ''

    n_extra = len(METRIC_COLUMNS)
    blank = [NA] * n_extra

    out = []
    # TR/TR (baseline: no delta / CI); prefer TSRTR file, fall back to TSTR
    acc_trtr = acc_of(rr, 'TRTR') if rr else None
    if acc_trtr is None:
        acc_trtr = acc_of(rt, 'TRTR')
    m_trtr = metrics_of(rr, 'TRTR') if rr else metrics_of(rt, 'TRTR')
    out.append(('TR/TR', fnum(acc_trtr), NA, NA, *m_trtr))

    # TS/TR
    if rt:
        out.append(('TS/TR', fnum(acc_of(rt, 'TSTR')),
                    fnum(rt.get('bag_delta'), signed=True) + star(rt),
                    fci(rt.get('bag_ci_diff_lo'), rt.get('bag_ci_diff_hi')),
                    *metrics_of(rt, 'TSTR')))
    else:
        out.append(('TS/TR', NA, NA, NA, *blank))

    # TSR/TR
    if rr:
        out.append(('TSR/TR', fnum(acc_of(rr, 'TSRTR')),
                    fnum(rr.get('bag_delta'), signed=True) + star(rr),
                    fci(rr.get('bag_ci_diff_lo'), rr.get('bag_ci_diff_hi')),
                    *metrics_of(rr, 'TSRTR')))
    else:
        out.append(('TSR/TR', NA, NA, NA, *blank))

    return out


def cbr_row(tstr_res, tsrtr_res):
    cbr = None
    for res in (tsrtr_res, tstr_res):
        if res and res.get('cbr'):
            cbr = res['cbr']
            break
    blank = [NA] * len(METRIC_COLUMNS)
    if not cbr:
        return ('CbR', NA, NA, NA, *blank)
    return ('CbR', fnum(cbr.get('cbr_bag_accuracy')), NA,
            fci(cbr.get('cbr_bag_ci_lo'), cbr.get('cbr_bag_ci_hi')),
            *cbr_metrics_of(cbr))


def latex_table(exp, desc, tstr_res, tsrtr_res):
    n_extra = len(METRIC_COLUMNS)
    col_spec = 'llcc' + 'c' * (1 + n_extra)  # model, regimen, acc, delta, ci, extras
    header_extra = ' & '.join(rf'\emph{{{h}}}' for _, h in METRIC_COLUMNS)

    L = []
    L.append(r'\begin{table}[H]')
    L.append(r'\centering')
    L.append(r'\scriptsize')
    cap = desc.replace('&', r'\&').replace('_', r'\_') if desc else ''
    L.append(rf'\caption{{Extended results for \texttt{{{exp.replace("_", r"\_")}}}'
             + (f': {cap}.' if cap else '.') + '}')
    L.append(rf'\label{{tab:appendix_{exp}}}')
    L.append(rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{col_spec}}}')
    L.append(r'\toprule')
    L.append(r'\emph{Model} & \emph{Regimen} & \emph{Accuracy} & \emph{$\Delta$} & '
             rf'\emph{{95\% CI}} & {header_extra}\\')
    L.append(r'\midrule')

    for mi, m in enumerate(MODEL_ORDER):
        rows = rows_for_model(m, tstr_res, tsrtr_res)
        label = MODEL_LABEL[m]
        for ri, row in enumerate(rows):
            reg, acc, dl, ci, *extras = row
            first = rf'\multirow{{{len(rows)}}}{{*}}{{{label}}}' if ri == 0 else ''
            extra_cols = ' & '.join(extras)
            L.append(f'{first} & {reg} & {acc} & {dl} & {ci} & {extra_cols}\\\\')
        L.append(r'\midrule' if mi < len(MODEL_ORDER) - 1 else r'\midrule')

    reg, acc, dl, ci, *extras = cbr_row(tstr_res, tsrtr_res)
    extra_cols = ' & '.join(extras)
    L.append(f'CbR & {NA} & {acc} & {dl} & {ci} & {extra_cols}\\\\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular*}')

    metric_names = ', '.join(h for _, h in METRIC_COLUMNS)
    L.append(r'\note{Regimens: TR/TR (real-only baseline), TS/TR (synthetic-only), '
             r'TSR/TR (synthetic + real), CbR (Classification by Reconstruction, '
             r'model-independent). Accuracy and all extra columns are bag-level: '
             r'one majority-vote decision per bag (biopsy slide), standard '
             r'metrics over the $n$ bag decisions -- see Appendix~\ref{app:baseline_correction}'
             r"'s sibling methods discussion for the full derivation. "
             r'$\Delta$ and 95\% CI are the bag-level bootstrap '
             r'difference vs.\ TR/TR (class-stratified resampling of bag decisions, '
             r'1000 iterations); (*) marks a 95\% CI excluding zero. '
             r'The CbR row reports its own bag-level accuracy 95\% CI '
             r'(not a difference vs.\ TR/TR), so its $\Delta$ column is left blank.}')
    L.append(r'\end{table}')
    L.append('')
    return '\n'.join(L)


def main():
    reg = json.load(open(REGISTRY))
    metric_names = ', '.join(h for _, h in METRIC_COLUMNS)
    parts = [r'\section{Extended Experiment Results}',
             r'\label{app:extended_results}',
             r'\setcounter{table}{0}',
             r'\renewcommand{\thetable}{\Alph{section}.\arabic{table}}', '',
             r'This appendix reports the per-experiment, per-model results in full. '
             r'Each table lists the bag-level accuracy under each training '
             r'regimen -- one majority-vote decision per bag (biopsy slide), '
             r'standard metrics over the resulting bag decisions -- the '
             r'bootstrap difference $\Delta$ relative to the TR/TR baseline '
             rf'with its 95\% confidence interval, and the bag-level {metric_names}. '
             r'Unavailable entries are marked with a dash.', '']
    summary = []
    n_unavailable = 0
    for exp in EXPERIMENTS:
        tstr = latest(exp, 'TSTR')
        tsrtr = latest(exp, 'TSRTR')
        if not tstr and not tsrtr:
            summary.append(f'  {exp}: SKIP (no stats-eval files)')
            continue
        desc = reg.get(exp, {}).get('description', '')
        parts.append(latex_table(exp, desc, tstr, tsrtr))
        has_bag = any(
            (res and any('bag_acc_TRTR' in r or 'cbr_bag_accuracy' in r for r in res['results']))
            for res in (tstr, tsrtr) if res
        )
        if not has_bag:
            n_unavailable += 1
        summary.append(f'  {exp}: TSTR={"y" if tstr else "-"} TSRTR={"y" if tsrtr else "-"} '
                       f'CbR={"y" if (tsrtr and tsrtr.get("cbr")) or (tstr and tstr.get("cbr")) else "-"} '
                       f'{"[STALE: pre-Stage-2 JSON, bag_* absent -> all dashes]" if not has_bag else ""}')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print(f'Wrote {OUT}  [{len(METRIC_COLUMNS)} extra column(s): {metric_names}]')
    print('\n'.join(summary))
    if n_unavailable:
        print(f'\n{n_unavailable} experiment(s) have only pre-Stage-2 JSONs -- re-run '
              f'stats_eval (pipeline/run_suite.py) to populate bag-level numbers.')


if __name__ == '__main__':
    main()
