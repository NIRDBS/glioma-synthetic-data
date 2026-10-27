r"""
Generate the per-experiment downstream-classifier results tables for the
"Controlled Experiments" appendix (paper/appendix_controlled_experiments.tex).

Sibling of gen_appendix_tables.py, not a modification of it: that script owns
EXPERIMENTS = exp_2xx/3xx/4xx (the pre-MIL-reframe suite) and
paper/appendix_extended_results.tex; this one owns the current paper's
controlled-experiment family (the CVAE beta sweep, latent-dim/reconstruction-
loss/synthetic-ratio axes, and the preprocessing-variant axis -- what
docs/experiment_inventory.md section 8a had called "ablation studies": not
accurate, since nothing is removed -- each axis varies one factor against a
shared anchor, exp_608) and writes to its own output file,
paper/appendix_controlled_experiments_tables.tex. Neither script touches the
other's EXPERIMENTS list or output file.

Column design, deliberately different from gen_appendix_tables.py's extra
columns: the primary Accuracy/Delta/95% CI columns are UNCHANGED (raw bag
accuracy, exactly the bootstrap this project's pipeline already computes --
see statistical_evaluation.py). Balanced accuracy, sensitivity, specificity
and AUROC are added as four EXTRA point-estimate columns, not a replacement
for the primary column: this project's bootstrap was run on raw accuracy,
not balanced accuracy, so reporting balanced accuracy as if it carried that
same CI/significance star would overstate what was actually tested. Both
numbers are shown; only one carries a significance claim.

Usage:
    python paper/gen_appendix_controlled_experiments.py
"""
import json
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REGISTRY = os.path.join(ROOT, 'configs', 'EXPERIMENT_REGISTRY.json')
OUT = os.path.join(ROOT, 'paper', 'appendix_controlled_experiments_tables.tex')

MODEL_ORDER = ['Apollo', 'RF_Standard', 'XGBoost', '1D_CNN']
MODEL_LABEL = {'Apollo': 'RF + SVM', 'RF_Standard': 'RF',
               'XGBoost': 'XGBoost', '1D_CNN': '1D-CNN'}

METRIC_COLUMNS = [
    ('bag_balanced_accuracy', r'Bal.\ Acc'),
    ('bag_sensitivity',       r'Sensitivity'),
    ('bag_specificity',       r'Specificity'),
    ('bag_auroc',             r'AUC'),
]

# Per-experiment parameter values, read directly from each config's own
# beta/latent_dim/gamma/delta/synth_ratio/processed_file (verified against
# the actual YAML, not re-derived from the registry's free-text
# description) -- these five numbers are what every table's caption states
# directly, since they matter more to a paper reader than any file-path or
# narrative cross-reference. Threshold is None for v2-preprocessing
# experiments (exp_2.6xx): that preprocessing is adaptive, not a single
# fixed threshold -- see ADAPTIVE_THRESHOLD_NOTE.
RECON_LABEL = {
    (1.0, 0.0): 'MSE',
    (0.5, 0.5): 'MSE + Cosine (0.5/0.5)',
    (0.0, 1.0): 'Cosine',
}
EXP_PARAMS = {
    'exp_501':   dict(beta=2.0,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_601':   dict(beta=1.0,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_607':   dict(beta=0.25, latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_608':   dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_609':   dict(beta=0.75, latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_701':   dict(beta=0.5,  latent_dim=16, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_702':   dict(beta=0.5,  latent_dim=64, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),
    'exp_703':   dict(beta=0.5,  latent_dim=32, gamma=0.5, delta=0.5, ratio=1.0, thr=0.6),
    'exp_704':   dict(beta=0.5,  latent_dim=32, gamma=0.0, delta=1.0, ratio=1.0, thr=0.6),
    'exp_705':   dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=0.5, thr=0.6),
    'exp_706':   dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=2.0, thr=0.6),
    'exp_707':   dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=3.0, thr=0.6),
    'exp_708':   dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.6),   # TSTR, 2x pool
    'exp_2.607': dict(beta=0.25, latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=None),
    'exp_2.608': dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=None),
    'exp_2.609': dict(beta=0.75, latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=None),
    'exp_4.608': dict(beta=0.5,  latent_dim=32, gamma=1.0, delta=0.0, ratio=1.0, thr=0.4),
}


def _fnum(x):
    """2.0 -> '2', 0.25 -> '0.25' -- matches the user's own template exactly
    (whole-valued parameters shown without a trailing .0)."""
    return str(int(x)) if float(x) == int(x) else str(x)


def build_caption(exp, with_note_pointer=False):
    p = EXP_PARAMS[exp]
    recon = RECON_LABEL[(p['gamma'], p['delta'])]
    if p['thr'] is None:
        pointer = r"; see note below" if with_note_pointer else ""
        thr = rf"adaptive, top-90\%-by-rank per slide floored at $0.8$ (v2 preprocessing{pointer})"
    else:
        thr = _fnum(p['thr'])
    return (rf"IDH binary classification, $\beta$ = {_fnum(p['beta'])}, "
           rf"\texttt{{latent\_dim}}={p['latent_dim']}, reconstruction = {recon}, "
           rf"synthetic ratio: {_fnum(p['ratio'])}, maximum intensity threshold = {thr}")


# Axis groups: (subsection title, [exp_name, ...], force_full_set).
# exp_608 is the shared anchor for latent-dim/recon-loss/ratio; its table is
# emitted once (under the beta sweep) and cross-referenced thereafter --
# EXCEPT for the threshold axis, where force_full re-emits its own table in
# full, since that axis is presented as a self-contained study.
AXIS_GROUPS = [
    ("CVAE $\\beta$ sweep (v1 preprocessing, \\texttt{latent\\_dim}=32)",
     ['exp_501', 'exp_601', 'exp_607', 'exp_608', 'exp_609'], set()),
    ("Latent dimension ($\\beta=0.5$)",
     ['exp_701', 'exp_608', 'exp_702'], set()),
    ("Reconstruction-loss weighting ($\\beta=0.5$, \\texttt{latent\\_dim}=32)",
     ['exp_703', 'exp_704'], set()),
    ("Synthetic/real ratio for TSRTR ($\\beta=0.5$, \\texttt{latent\\_dim}=32)",
     ['exp_705', 'exp_706'], set()),
    ("Max Intensity Threshold $\\text{thr} \\in \\{0.4, 0.6, 0.8\\}$ ($\\beta=0.5$)",
     ['exp_4.608', 'exp_608', 'exp_2.608'], {'exp_608'}),
    ("v2-preprocessing's own $\\beta$ sweep (cross-check)",
     ['exp_2.607', 'exp_2.608', 'exp_2.609'], set()),
]

THRESHOLD_NOTE = (
    r"\emph{Note.} \texttt{exp\_2.608}'s preprocessing is not a pure fixed "
    r"threshold despite the $0.8$ figure above: it keeps each slide's own "
    r"top-90\% of pixels by max-intensity rank (a per-slide 10th-percentile "
    r"cutoff), with $\text{thr}>0.8$ applied only as an absolute floor "
    r"within that selection (rarely binding). \texttt{exp\_608} and "
    r"\texttt{exp\_4.608} use a single fixed threshold directly on "
    r"max-intensity, with no rank-based component."
)

NA = '--'


def _derive_bag_sens_spec(fm):
    """See gen_appendix_tables.py's identical helper -- duplicated rather than
    imported so this script has no dependency on the sibling's internals."""
    cm = fm.get('confusion_matrix')
    if not cm:
        return
    import numpy as np
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
        fm['bag_sensitivity'] = float(sum(sens_list) / len(sens_list))
    if fm.get('bag_specificity') is None and spec_list:
        fm['bag_specificity'] = float(sum(spec_list) / len(spec_list))


def latest(exp, candidate):
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
    if r is None:
        return None
    return r.get(f'bag_acc_{label}')


def metrics_of(r, label):
    fm = dict((r or {}).get(f'full_metrics_{label}') or {})
    _derive_bag_sens_spec(fm)
    return [fnum(fm.get(key)) for key, _ in METRIC_COLUMNS]


def cbr_metrics_of(cbr):
    fm = dict((cbr.get('full_metrics') or {}))
    _derive_bag_sens_spec(fm)
    flat = dict(cbr)
    for key in ('bag_sensitivity', 'bag_specificity'):
        if flat.get(key) is None and fm.get(key) is not None:
            flat[key] = fm[key]
    return [fnum(flat.get(key)) for key, _ in METRIC_COLUMNS]


def rows_for_model(m, tstr_res, tsrtr_res):
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
    acc_trtr = acc_of(rr, 'TRTR') if rr else None
    if acc_trtr is None:
        acc_trtr = acc_of(rt, 'TRTR')
    m_trtr = metrics_of(rr, 'TRTR') if rr else metrics_of(rt, 'TRTR')
    out.append(('TR/TR', fnum(acc_trtr), NA, NA, *m_trtr))

    if rt:
        out.append(('TS/TR', fnum(acc_of(rt, 'TSTR')),
                    fnum(rt.get('bag_delta'), signed=True) + star(rt),
                    fci(rt.get('bag_ci_diff_lo'), rt.get('bag_ci_diff_hi')),
                    *metrics_of(rt, 'TSTR')))
    else:
        out.append(('TS/TR', NA, NA, NA, *blank))

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


def latex_table(exp, desc, tstr_res, tsrtr_res, raw_desc=False, label_suffix=''):
    n_extra = len(METRIC_COLUMNS)
    col_spec = 'llcc' + 'c' * (1 + n_extra)
    header_extra = ' & '.join(rf'\emph{{{h}}}' for _, h in METRIC_COLUMNS)

    L = []
    L.append(r'\begin{table}[H]')
    L.append(r'\centering')
    L.append(r'\scriptsize')
    cap = desc if raw_desc else (desc.replace('&', r'\&').replace('_', r'\_') if desc else '')
    L.append(rf'\caption{{\texttt{{{exp.replace("_", r"\_")}}}'
             + (f': {cap}.' if cap else '.') + '}')
    L.append(rf'\label{{tab:ctrl_{exp}{label_suffix}}}')
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
        L.append(r'\midrule')

    reg, acc, dl, ci, *extras = cbr_row(tstr_res, tsrtr_res)
    extra_cols = ' & '.join(extras)
    L.append(f'CbR & {NA} & {acc} & {dl} & {ci} & {extra_cols}\\\\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular*}')
    L.append(r'\end{table}')
    L.append('')
    return '\n'.join(L)


BETA_TABLE_EXPS = ['exp_607', 'exp_608', 'exp_609']
BETA_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_beta_table.tex')


def beta_table_latex(exps=BETA_TABLE_EXPS):
    """One table for the beta axis (exp_607/608/609), for the main-text Results
    section: Model | Regimen | beta | metrics, each regimen's beta cell split
    into one row per experiment. Rows come from the SAME rows_for_model /
    cbr_row helpers as the per-experiment appendix tables, so every number is
    identical to the corresponding appendix cell."""
    n_extra = len(METRIC_COLUMNS)
    col_spec = 'lllccc' + 'c' * n_extra
    ncols = 3 + 3 + n_extra
    # Compact headers + tight column padding: with 10 columns the appendix-style headers
    # overflow the main-text width by ~70pt.
    short = {'Sensitivity': 'Sens.', 'Specificity': 'Spec.'}
    header_extra = ' & '.join(rf'\emph{{{short.get(h, h)}}}' for _, h in METRIC_COLUMNS)

    betas, tstrs, tsrtrs = [], {}, {}
    for e in exps:
        betas.append(EXP_PARAMS[e]['beta'])
        tstrs[e], tsrtrs[e] = latest(e, 'TSTR'), latest(e, 'TSRTR')
        if not tstrs[e] and not tsrtrs[e]:
            raise SystemExit(f'{e}: no stats-eval results found')
    beta_lbl = [rf'${_fnum(b)}$' for b in betas]
    n_b = len(exps)

    def block(label, regimen_rows):
        """regimen_rows: [(regimen_name, [(beta_cell, row), ...])]; each row is
        (acc, delta, ci, *extras). A regimen with one entry is a single line."""
        out = []
        total = sum(len(rows) for _, rows in regimen_rows)
        for ri, (reg, rows) in enumerate(regimen_rows):
            for bi, (beta_cell, row) in enumerate(rows):
                first = rf'\multirow{{{total}}}{{*}}{{{label}}}' if (ri == 0 and bi == 0) else ''
                if bi == 0:
                    reg_cell = rf'\multirow{{{len(rows)}}}{{*}}{{{reg}}}' if len(rows) > 1 else reg
                else:
                    reg_cell = ''
                out.append(f'{first} & {reg_cell} & {beta_cell} & ' + ' & '.join(row) + r'\\')
            if ri < len(regimen_rows) - 1:
                out.append(rf'\cmidrule(lr){{2-{ncols}}}')
        return out

    L = [r'\begin{table}[H]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Effect of the CVAE KL weight $\beta$ on bag-level binary IDH classification '
         r'(\texttt{exp\_607}, \texttt{exp\_608}, \texttt{exp\_609}: \texttt{latent\_dim}=32, '
         r'reconstruction = MSE, synthetic ratio: 1, maximum intensity threshold = 0.6).}',
         r'\label{tab:beta_results}',
         rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{col_spec}}}',
         r'\toprule',
         r'\emph{Model} & \emph{Regimen} & \emph{$\beta$} & \emph{Accuracy} & \emph{$\Delta$} & '
         rf'\emph{{95\% CI}} & {header_extra}\\',
         r'\midrule']

    for m in MODEL_ORDER:
        per_exp_rows = [rows_for_model(m, tstrs[e], tsrtrs[e]) for e in exps]
        regimens = [per_exp_rows[0][k][0] for k in range(3)]
        assert regimens[0] == 'TR/TR'
        # TR/TR is the shared real-only baseline: one line, beta shown as "--".
        # Guard that the three experiments really carry identical baseline cells.
        trtr = [rows[0][1:] for rows in per_exp_rows]
        assert all(t == trtr[0] for t in trtr), f'{m}: TR/TR differs across {exps}'
        regimen_rows = [('TR/TR', [(NA, trtr[0])])]
        for k in (1, 2):
            regimen_rows.append((regimens[k], [(beta_lbl[bi], per_exp_rows[bi][k][1:])
                                               for bi in range(n_b)]))
        L += block(MODEL_LABEL[m], regimen_rows)
        L.append(r'\midrule')

    cbr = [cbr_row(tstrs[e], tsrtrs[e]) for e in exps]
    L += block('CbR', [('--', [(beta_lbl[bi], row[1:]) for bi, row in enumerate(cbr)])])
    L += [r'\bottomrule', r'\end{tabular*}',
          r'\note{Regimens: TR/TR (real-only), TS/TR (synthetic-only), TSR/TR (synthetic + real), '
          r'CbR (Classification by Reconstruction). All metrics are bag-level ($n=59$ slides, '
          r'majority vote). $\Delta$ and 95\% CI: class-stratified bootstrap difference in accuracy '
          r'vs.\ TR/TR (1000 iterations); (*) = CI excluding zero. Balanced accuracy, sensitivity '
          r'(IDH-wildtype recall), specificity (IDH-mutant recall) and AUC are point estimates. '
          r'TR/TR is real-only and independent of $\beta$ (one line per model). CbR reports its own '
          r'accuracy and 95\% CI; its AUC ranks the bags by the mean over their spectra of the normalised '
          r'reconstruction-error margin $(\mathrm{MSE}_0-\mathrm{MSE}_1)/(\mathrm{MSE}_0+\mathrm{MSE}_1)$.}',
          r'\end{table}', '']
    return '\n'.join(L)


LATENT_TABLE_EXPS = ['exp_701', 'exp_608', 'exp_702']   # d = 16, 32, 64 (beta=0.5 throughout)
LATENT_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_latent_dim_table.tex')


def _accuracy_axis_table(exps, col_labels, caption, label):
    """Compact main-text table for one controlled axis: bag-level accuracy only,
    TS/TR and TSR/TR side by side, one column per axis value (col_labels, one
    per experiment, repeated under each regimen). Values come from the same
    rows_for_model helper as the appendix tables."""
    tstr = {e: latest(e, 'TSTR') for e in exps}
    tsrtr = {e: latest(e, 'TSRTR') for e in exps}
    for e in exps:
        if not tstr[e] and not tsrtr[e]:
            raise SystemExit(f'{e}: no stats-eval results found')

    n = len(exps)
    L = [r'\begin{table}[H]', r'\centering', r'\small',
         rf'\caption{{{caption}}}',
         rf'\label{{{label}}}',
         rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l{"c" * (2 * n)}}}',
         r'\toprule',
         rf'\multirow{{2}}{{*}}{{\emph{{Model}}}} & \multicolumn{{{n}}}{{c}}{{\emph{{TS/TR}}}} & '
         rf'\multicolumn{{{n}}}{{c}}{{\emph{{TSR/TR}}}} \\',
         rf'\cmidrule(lr){{2-{1 + n}}} \cmidrule(lr){{{2 + n}-{1 + 2 * n}}}',
         ' & ' + ' & '.join(col_labels + col_labels) + r' \\',
         r'\midrule']
    for m in MODEL_ORDER:
        per_exp = [rows_for_model(m, tstr[e], tsrtr[e]) for e in exps]
        ts = [rows[1][1] for rows in per_exp]      # row = (regimen, acc, ...): TS/TR
        tsr = [rows[2][1] for rows in per_exp]     # TSR/TR
        L.append(f'{MODEL_LABEL[m]} & ' + ' & '.join(ts + tsr) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(L)


_AXIS_CAPTION_HEAD = (r'Bag-level accuracy ($n=59$ slides) of the downstream classifiers trained on '
                      r'synthetic data only (TS/TR) and on synthetic and real data (TSR/TR), for ')


def latent_dim_table_latex(exps=LATENT_TABLE_EXPS):
    dims = [EXP_PARAMS[e]['latent_dim'] for e in exps]
    return _accuracy_axis_table(
        exps, [rf'$d={d}$' for d in dims],
        _AXIS_CAPTION_HEAD + rf'CVAE latent dimensions $d \in \{{{", ".join(str(d) for d in dims)}\}}$ '
        r'($\beta=0.5$, reconstruction = MSE, synthetic ratio: 1, maximum intensity threshold = 0.6).',
        'tab:latent_dim_results')


RECON_TABLE_EXPS = ['exp_608', 'exp_703', 'exp_704']   # MSE, MSE+Cos (0.5/0.5), Cos
RECON_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_recon_loss_table.tex')
RECON_COL_LABEL = {'MSE': 'MSE', 'MSE + Cosine (0.5/0.5)': 'MSE + Cos', 'Cosine': 'Cos'}


def recon_loss_table_latex(exps=RECON_TABLE_EXPS):
    labels = []
    for e in exps:
        p = EXP_PARAMS[e]
        labels.append(RECON_COL_LABEL[RECON_LABEL[(p['gamma'], p['delta'])]])
    return _accuracy_axis_table(
        exps, labels,
        _AXIS_CAPTION_HEAD + r'three CVAE reconstruction losses: mean squared error (MSE), an equal '
        r'mixture of MSE and cosine loss ($0.5/0.5$, MSE + Cos), and cosine loss (Cos) '
        r'($\beta=0.5$, \texttt{latent\_dim}=32, synthetic ratio: 1, maximum intensity threshold = 0.6).',
        'tab:recon_loss_results')


RATIO_TABLE_EXPS = ['exp_705', 'exp_608', 'exp_706', 'exp_707']   # synthetic:real = 0.5, 1, 2, 3
POOL_TSTR_EXPS = ['exp_608', 'exp_708']                           # 40k, 80k synthetic spectra (TS/TR)
RATIO_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_synth_ratio_table.tex')
POOL_BOOT_JSON = os.path.join(RESULTS, 'synth_pool_bootstrap', 'pairwise_pool_size_bootstrap.json')
LATENT_BOOT_JSON = os.path.join(RESULTS, 'latent_dim_bootstrap', 'pairwise_latent_dim_bootstrap.json')
BOOT_APPENDIX_OUT = os.path.join(ROOT, 'paper', 'appendix_bootstrap_tables.tex')
POOL_LABEL = {'exp_608': '40k', 'exp_708': '80k'}   # synthetic spectra per fold in the TS/TR pool


def synth_ratio_table_latex(exps=RATIO_TABLE_EXPS, pool_exps=POOL_TSTR_EXPS):
    """Bag-level accuracy vs. the amount of synthetic data.
    TS/TR: two columns, the synthetic pool of exp_608 (40,000 spectra per fold) and of exp_708
    (80,000), generated from the same decoders. The synthetic:real ratio only enters TSR/TR's
    mixing step, so TS/TR does not depend on it (the 40k column equals every ratio experiment's
    own TS/TR, asserted below). TSR/TR: one column per ratio. A star marks a 80k-vs-40k difference
    whose 95% bag-level bootstrap CI excludes zero (results/synth_pool_bootstrap)."""
    tstr = {e: latest(e, 'TSTR') for e in set(exps) | set(pool_exps)}
    tsrtr = {e: latest(e, 'TSRTR') for e in exps}
    for e in exps:
        if not tsrtr[e]:
            raise SystemExit(f'{e}: no TSRTR stats-eval results found')
    ratios = [EXP_PARAMS[e]['ratio'] for e in exps]

    pool_sig = {}
    if os.path.exists(POOL_BOOT_JSON):
        pool_sig = {t['model']: t['significant_95'] for t in json.load(open(POOL_BOOT_JSON))['tests']}

    L = [r'\begin{table}[H]', r'\centering', r'\small',
         r'\caption{Bag-level accuracy ($n=59$ slides) of the downstream classifiers trained on '
         r'synthetic data only (TS/TR; synthetic pools of 40,000 and 80,000 spectra) and on synthetic '
         rf'and real data (TSR/TR) at synthetic-to-real ratios {", ".join(_fnum(r) + ":1" for r in ratios[:-1])} '
         rf'and {_fnum(ratios[-1])}:1 '
         r'($\beta=0.5$, \texttt{latent\_dim}=32, reconstruction = MSE, maximum intensity threshold = 0.6). '
         + (r'$^{*}$ The 80k pool differs from the 40k pool at the 95\% level '
            r'(bag-level bootstrap; Appendix~\ref{app:controlled_experiments}).' if any(pool_sig.values()) else '')
         + '}',
         r'\label{tab:synth_ratio_results}',
         rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l{"c" * (len(pool_exps) + len(exps))}}}',
         r'\toprule',
         rf'\multirow{{2}}{{*}}{{\emph{{Model}}}} & \multicolumn{{{len(pool_exps)}}}{{c}}{{\emph{{TS/TR}}}} & '
         rf'\multicolumn{{{len(exps)}}}{{c}}{{\emph{{TSR/TR}}}} \\',
         rf'\cmidrule(lr){{2-{1 + len(pool_exps)}}} \cmidrule(lr){{{2 + len(pool_exps)}-{1 + len(pool_exps) + len(exps)}}}',
         ' & ' + ' & '.join([f'${POOL_LABEL[e][:-1]}$k' for e in pool_exps]
                            + [rf'${_fnum(r)}{{:}}1$' for r in ratios]) + r' \\',
         r'\midrule']
    for m in MODEL_ORDER:
        ts_pool = [rows_for_model(m, tstr[e], latest(e, 'TSRTR'))[1][1] for e in pool_exps]
        for e in exps:   # TS/TR must not depend on the ratio: equals the 40k pool's value
            assert rows_for_model(m, tstr[e], tsrtr[e])[1][1] == ts_pool[0], f'{m}: TS/TR differs for {e}'
        if pool_sig.get(m):
            ts_pool[-1] = ts_pool[-1][:-1] + r'^{*}$'   # $0.780$ -> $0.780^{*}$ (no adjacent $$)
        tsr = [rows_for_model(m, tstr[e], tsrtr[e])[2][1] for e in exps]
        L.append(f'{MODEL_LABEL[m]} & ' + ' & '.join(ts_pool + tsr) + r' \\')
    L += [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(L)


def _boot_delta(t):
    return f'${t["delta"]:+.3f}$' + ('*' if t['significant_95'] else '')


def _boot_ci(t):
    return f'$[{t["ci_lo"]:+.3f}, {t["ci_hi"]:+.3f}]$'


def bootstrap_appendix_tables_latex():
    """Appendix tables of the pairwise bag-level bootstrap tests: latent dimension (16 tests)
    and TS/TR synthetic-pool size (4 tests). Read from the result JSONs written by
    scripts/latent_dim_pairwise_bootstrap.py and scripts/synth_pool_pairwise_bootstrap.py."""
    lat = json.load(open(LATENT_BOOT_JSON))['tests']
    pool = json.load(open(POOL_BOOT_JSON))['tests']
    by = {(t['regimen'], t['model'], t['comparison']): t for t in lat}
    comps = [('d=16 vs d=32', r'$16 \to 32$'), ('d=32 vs d=64', r'$32 \to 64$')]

    L = [r'\begin{table}[H]', r'\centering', r'\scriptsize',
         r'\caption{Pairwise bag-level bootstrap between CVAE latent dimensions ($\beta=0.5$, '
         r'\texttt{latent\_dim} $d\in\{16,32,64\}$). $\Delta$ is the bag accuracy at the larger $d$ minus '
         r'at the smaller $d$, with its 95\% CI from the class-stratified bootstrap over the $n=59$ bag '
         r'decisions (1000 iterations, paired); (*) marks a CI excluding zero.}',
         r'\label{tab:ctrl_boot_latent}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc}',
         r'\toprule',
         r'\multirow{2}{*}{\emph{Model}} & \multirow{2}{*}{\emph{$d$}} & \multicolumn{2}{c}{\emph{TS/TR}} & '
         r'\multicolumn{2}{c}{\emph{TSR/TR}} \\',
         r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}',
         r' & & \emph{$\Delta$} & \emph{95\% CI} & \emph{$\Delta$} & \emph{95\% CI} \\',
         r'\midrule']
    for m in MODEL_ORDER:
        for ci, (key, label) in enumerate(comps):
            a, b = by[('TS/TR', m, key)], by[('TSR/TR', m, key)]
            first = rf'\multirow{{2}}{{*}}{{{MODEL_LABEL[m]}}}' if ci == 0 else ''
            L.append(f'{first} & {label} & {_boot_delta(a)} & {_boot_ci(a)} & {_boot_delta(b)} & {_boot_ci(b)} \\\\')
        L.append(r'\midrule' if m != MODEL_ORDER[-1] else r'\bottomrule')
    L += [r'\end{tabular*}',
          r'\note{16 tests (4 models $\times$ 2 comparisons $\times$ 2 regimens), no multiplicity '
          r'correction. A zero-width interval ($\Delta=0$, $[+0.000,+0.000]$) means the two models make '
          r'identical decisions on every bag, so the resampled difference is exactly zero; it is not a '
          r'degenerate bootstrap.}',
          r'\end{table}', '']

    L += [r'\begin{table}[H]', r'\centering', r'\scriptsize',
          r'\caption{Pairwise bag-level bootstrap between synthetic pools of 40,000 (\texttt{exp\_608}) and '
          r'80,000 (\texttt{exp\_708}) spectra per fold, TS/TR ($\beta=0.5$; same CVAE decoders, models and '
          r'validation folds). $\Delta$ is the bag accuracy with the 80k pool minus with the 40k pool; '
          r'95\% CI as in Table~\ref{tab:ctrl_boot_latent}; (*) marks a CI excluding zero.}',
          r'\label{tab:ctrl_boot_pool}',
          r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc}',
          r'\toprule',
          r'\emph{Model} & \emph{40k} & \emph{80k} & \emph{$\Delta$} & \emph{95\% CI} & \emph{$p$} \\',
          r'\midrule']
    pby = {t['model']: t for t in pool}
    for m in MODEL_ORDER:
        t = pby[m]
        L.append(f'{MODEL_LABEL[m]} & ${t["acc_baseline"]:.3f}$ & ${t["acc_candidate"]:.3f}$ & '
                 f'{_boot_delta(t)} & {_boot_ci(t)} & ${t["p_bootstrap"]:.3f}$ \\\\')
    L += [r'\bottomrule', r'\end{tabular*}',
          r'\note{4 tests, no multiplicity correction.}', r'\end{table}', '']
    return '\n'.join(L)


THR_TABLE_EXPS = ['exp_4.608', 'exp_608', 'exp_2.608']   # thr = 0.4, 0.6, adaptive
THR_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_threshold_table.tex')


def threshold_table_latex(exps=THR_TABLE_EXPS):
    """Bag-level accuracy vs. the strong-signal mask's max-intensity threshold.
    exp_2.608's mask is NOT a fixed threshold (EXP_PARAMS thr=None): per slide, the
    top 90% of pixels by max intensity with an absolute 0.8 floor -- labelled
    'adaptive' and defined in the caption rather than shown as a number."""
    for e in exps:
        for cand in ('TSTR', 'TSRTR'):
            res = latest(e, cand)
            assert res and all(r['n_bags'] == 59 for r in res['results']), f'{e} {cand}: n_bags != 59'
    labels = [rf'${_fnum(EXP_PARAMS[e]["thr"])}$' if EXP_PARAMS[e]['thr'] is not None else 'adaptive'
              for e in exps]
    return _accuracy_axis_table(
        exps, labels,
        _AXIS_CAPTION_HEAD + r'the maximum-intensity threshold of the strong-signal mask: fixed '
        r"thresholds of $0.4$ and $0.6$, and an adaptive rule that keeps each slide's top $90\%$ of "
        r'pixels by maximum intensity subject to an absolute floor of $0.8$ '
        r'($\beta=0.5$, \texttt{latent\_dim}=32, reconstruction = MSE, synthetic ratio: 1).',
        'tab:threshold_results')


MIL_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_mil_table.tex')
APPENDIX_CTRL = os.path.join(ROOT, 'paper', 'appendix_controlled_experiments.tex')


def mil_table_latex(exp='exp_608'):
    """Main-text MIL table for the beta=0.5 / v1 (thr=0.6) configuration: the five MIL
    tiers' rows are taken verbatim from the hand-authored per-experiment MIL table in
    the appendix (so the numbers cannot differ), preceded by a naive-MIL benchmark row:
    the Tier-0 random forest trained on real data (TR/TR), one majority-vote decision
    per bag, from the same stats-eval results as every other classifier table."""
    src = open(APPENDIX_CTRL, encoding='utf-8').read()
    key = rf'MIL tier bag-level results, \texttt{{{exp.replace("_", r"\_")}}}'
    i = src.index(key)
    body = src[src.index(r'\midrule', i) + len(r'\midrule'):src.index(r'\bottomrule', i)]
    tier_rows = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    assert len(tier_rows) == 5, tier_rows

    tstr, tsrtr = latest(exp, 'TSTR'), latest(exp, 'TSRTR')
    reg, acc, _dl, _ci, bal, sens, spec, auc = rows_for_model('RF_Standard', tstr, tsrtr)[0]
    assert reg == 'TR/TR'
    naive = rf'Na\"{{\i}}ve MIL (RF) & {acc} & {bal} & {sens} & {spec} & {auc} \\'

    L = [r'\begin{table}[H]', r'\centering', r'\small',
         r'\caption{Bag-level results ($n=59$ slides) of the multiple-instance learning (MIL) '
         r'formulations against the na\"{\i}ve MIL benchmark, a random forest trained on real '
         r"data in which every instance inherits its bag's label, with one majority-vote decision "
         r'per bag (\texttt{exp\_608}: $\beta=0.5$, \texttt{latent\_dim}=32, reconstruction = MSE, '
         r'maximum intensity threshold = 0.6). CVAE-Attention, CVAE reconstruction and CVAE encoder '
         r'head use the $\beta=0.5$ CVAE; mi-SVM and Bayesian MIL do not. Bayesian MIL is scored '
         r'blind: the bag label is never given to the scorer.}',
         r'\label{tab:mil_results}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc}',
         r'\toprule',
         r'\emph{Model} & \emph{Accuracy} & \emph{Bal.\ Acc} & \emph{Sensitivity} & '
         r'\emph{Specificity} & \emph{AUC} \\',
         r'\midrule', naive, r'\midrule'] + tier_rows + [r'\bottomrule', r'\end{tabular*}',
                                                          r'\end{table}', '']
    return '\n'.join(L)


MIL_TESTS_TABLE_OUT = os.path.join(ROOT, 'paper', 'results_mil_tests_table.tex')
# (label, stage-5 method/variant file stem, key in key_instance_utility_summary.json)
MIL_TESTS_TIERS = [
    ('mi-SVM',              'misvm_mi_svm',         'misvm'),
    ('CVAE-Attention',      'attention_attention',  'attention'),
    ('CVAE reconstruction', 'cbr_reconstruction',   'cbr_reconstruction'),
    ('CVAE encoder head',   'cbr_encoder_head',     'cbr_encoder_head'),
    ('Bayesian MIL',        'bayes_mil_gibbs',      'bayes_mil'),
]
PENDING = r'\emph{pending}'
TIE_REL = 0.03   # same relative-difference tie rule as the appendix's spatial tables


def _stage5_moran_lag3(exp, stem, shuffled):
    """Pooled Moran's I at lag 3, or None if that fit's stage-5 file is missing."""
    path = os.path.join(RESULTS, exp, 'spatial_validation',
                        f'stage5_{stem}{"_shuffled" if shuffled else ""}.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path))['pooled_morans_i_by_lag']['3']['mean']


def _std_variogram_lag3(std, stem, fit):
    """Variance-standardized lag-3 variogram gamma/sigma^2 (scripts/standardized_variogram.py)."""
    return (std.get(stem) or {}).get(fit, {}).get('gamma3_over_var')


def _more_coherent(real, shuf):
    """True when the real fit is more coherent than the shuffled one by more than the 3%
    tie tolerance, relative to the larger coherence. Coherence is Moran's I, or
    1 - gamma/sigma^2 for the standardized variogram (same scale, ~ lag-3 autocorrelation)."""
    return real - shuf > TIE_REL * max(abs(real), abs(shuf))


def mil_tests_table_latex(exp='exp_608'):
    """Status table of the MIL validation tests for one configuration. Every cell is
    read from result files (stage-5 JSONs, standardized_variogram_lag3.json,
    key_instance_utility_summary.json); a test whose result does not exist yet is shown
    as 'pending', so re-running this generator after the missing jobs finish fills the
    table in with no manual edits.

    Spatial verdict: yes only if the real fit is more coherent than the shuffled-label
    fit on BOTH Moran's I and the standardized variogram. Key-instance verdict: the
    RF trained on the selected instances vs. the naive MIL benchmark (the same RF
    trained on all instances, all of them voting), compared exactly."""
    ki = json.load(open(os.path.join(RESULTS, exp, 'mil_eval', 'key_instance_utility_summary.json')))
    std = json.load(open(os.path.join(RESULTS, exp, 'spatial_validation',
                                      'standardized_variogram_lag3.json')))
    naive = ki['baseline_bag_acc']

    rows = []
    for label, stem, ki_key in MIL_TESTS_TIERS:
        i_real, i_shuf = _stage5_moran_lag3(exp, stem, False), _stage5_moran_lag3(exp, stem, True)
        v_real, v_shuf = _std_variogram_lag3(std, stem, 'real'), _std_variogram_lag3(std, stem, 'shuffled')
        fi = lambda v: PENDING if v is None else f'${v:.4f}$'
        fv = lambda v: PENDING if v is None else f'${1 - v:.4f}$'   # 1 - gamma/sigma^2
        if None in (i_real, i_shuf, v_real, v_shuf):
            spatial_verdict = PENDING
        else:
            passes = _more_coherent(i_real, i_shuf) and _more_coherent(1 - v_real, 1 - v_shuf)
            spatial_verdict = 'yes' if passes else 'no'
        k = ki.get(ki_key)
        if k:
            sel, rnd = k['key_instance_train'], k['random_subsample']
            verdict = lambda a, b: 'tie' if a == b else ('yes' if a > b else 'no')
            ki_cells = (f'${sel:.4f}$ & ${naive:.4f}$ & ${rnd:.4f}$ & '
                        f'{verdict(sel, naive)} & {verdict(sel, rnd)}')
        else:
            ki_cells = ' & '.join([PENDING] * 5)
        rows.append(f'{label} & {fi(i_real)} & {fi(i_shuf)} & {fv(v_real)} & {fv(v_shuf)} & '
                    f'{spatial_verdict} & {ki_cells} \\\\')

    L = [r'\begin{table}[H]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{2pt}',
         r'\caption{Status of the MIL validation tests for the $\beta=0.5$ configuration '
         r"(\texttt{exp\_608}). \emph{Spatial coherence:} Moran's $I$ and the "
         r'variance-standardized variogram, reported as $1-\hat\gamma_3/\hat\sigma^2$ (on the same scale '
         r'as $I$; higher is more coherent), at Chebyshev lag 3, for the real fit and for a refit under '
         r'shuffled bag labels. The verdict is yes only if the real fit exceeds the shuffled one on both '
         r'statistics by more than 3\% of the larger value. '
         r'\emph{Key-instance utility:} bag accuracy of a random forest trained on the instances a tier '
         r'selects (median split on non-diagnostic mass), against (i) the na\"{\i}ve MIL benchmark, the '
         r'same forest trained on all instances with all of them voting, and (ii) a forest trained on a '
         r'size-matched random subset of the training instances.}',
         r'\label{tab:mil_tests_status}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccccc}',
         r'\toprule',
         r' & \multicolumn{5}{c}{\emph{Spatial coherence (lag 3)}} & '
         r'\multicolumn{5}{c}{\emph{Key-instance utility}} \\',
         r'\cmidrule(lr){2-6} \cmidrule(lr){7-11}',
         r' & \multicolumn{2}{c}{\emph{Moran\textquotesingle s $I$}} & '
         r'\multicolumn{2}{c}{\emph{$1-\hat\gamma_3/\hat\sigma^2$}} & '
         r'\multirow{2}{*}{\shortstack{\emph{Real $>$}\\\emph{shuf.?}}} & '
         r'\multicolumn{3}{c}{\emph{RF accuracy}} & '
         r'\multirow{2}{*}{\shortstack{\emph{Sel.\ $>$}\\\emph{naive?}}} & '
         r'\multirow{2}{*}{\shortstack{\emph{Sel.\ $>$}\\\emph{rand.?}}} \\',
         r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){7-9}',
         r'\emph{Model} & \emph{real} & \emph{shuf.} & \emph{real} & \emph{shuf.} & & '
         r'\emph{Selected} & \emph{Naive} & \emph{Random} & & \\',
         r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(L)


def _results_tables():
    return ((BETA_TABLE_OUT, beta_table_latex()),
            (LATENT_TABLE_OUT, latent_dim_table_latex()),
            (RECON_TABLE_OUT, recon_loss_table_latex()),
            (RATIO_TABLE_OUT, synth_ratio_table_latex()),
            (THR_TABLE_OUT, threshold_table_latex()),
            (MIL_TABLE_OUT, mil_table_latex()),
            (MIL_TESTS_TABLE_OUT, mil_tests_table_latex()),
            (BOOT_APPENDIX_OUT, bootstrap_appendix_tables_latex()))


def main(results_tables_only=False):
    if results_tables_only:
        for path, text in _results_tables():
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            print(f'Wrote {path}')
        return
    metric_names = ', '.join(h for _, h in METRIC_COLUMNS)
    parts = []
    summary = []
    emitted = set()

    for subsection, exps, force_full in AXIS_GROUPS:
        parts.append(rf'\subsection*{{{subsection}}}')
        parts.append('')
        for exp in exps:
            if exp in emitted and exp not in force_full:
                parts.append(rf'\texttt{{{exp.replace("_", r"\_")}}}: shared anchor, '
                             r'see its table above.')
                parts.append('')
                continue
            tstr = latest(exp, 'TSTR')
            tsrtr = latest(exp, 'TSRTR')
            if not tstr and not tsrtr:
                summary.append(f'  {exp}: SKIP (no stats-eval files)')
                parts.append(rf'\texttt{{{exp.replace("_", r"\_")}}}: no stats-eval '
                             r'results available.')
                parts.append('')
                continue
            is_threshold_axis = subsection.startswith('Max Intensity Threshold')
            with_note_pointer = is_threshold_axis and exp == 'exp_2.608'
            desc = build_caption(exp, with_note_pointer=with_note_pointer)
            label_suffix = '_thr' if (is_threshold_axis and exp in force_full) else ''
            parts.append(latex_table(exp, desc, tstr, tsrtr, raw_desc=True,
                                     label_suffix=label_suffix))
            emitted.add(exp)
            summary.append(f'  {exp}: TSTR={"y" if tstr else "-"} TSRTR={"y" if tsrtr else "-"}')
        if exp == 'exp_2.608' and subsection.startswith('Max Intensity Threshold'):
            parts.append(THRESHOLD_NOTE)
            parts.append('')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print(f'Wrote {OUT}  [{len(METRIC_COLUMNS)} extra column(s): {metric_names}]')
    print('\n'.join(summary))
    for path, text in _results_tables():
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        print(f'Wrote {path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-tables-only', '--beta-table-only', dest='results_tables_only',
                    action='store_true',
                    help='write only the main-text Results tables (paper/results_beta_table.tex, '
                         'paper/results_latent_dim_table.tex), leaving the appendix tables file untouched')
    main(results_tables_only=ap.parse_args().results_tables_only)
