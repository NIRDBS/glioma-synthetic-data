"""paper/gen_boot_values.py: the v2.2.1 formatter quirks and the convert ->
expand round trip (slide mode must reproduce the original sources byte for byte)."""
import importlib.util
import os
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('gen_boot_values', os.path.join(ROOT, 'paper', 'gen_boot_values.py'))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


# ------------------------------------------------------------ formatter

@pytest.mark.parametrize('x, kw, expected', [
    (-0.08474576, {}, '-0.09'),                 # '-0.085' -> half-up -> '-0.09' (direct rounding: -0.08)
    (0.0847457627, {}, '+0.09'),
    (0.06779661, {}, '+0.07'),
    (0.0, {}, '+0.00'),
    (0.0169491525, {}, '+0.02'),
    (0.678, {'signed': False}, '0.68'),
    (0.00464, {'scale': 100}, '+0.50'),         # '0.005' x 100
    (0.00464, {'scale': 100, 'signed': False}, '0.50'),
    (-0.01154, {'scale': 100}, '-1.20'),        # '-0.012' x 100
    (-0.012638532, {'scale': 100}, '-1.30'),    # '-0.013' x 100
    (0.8, {'signed': False}, '0.80'),
])
def test_fmt_num_known_cases(x, kw, expected):
    assert g.fmt_num(x, **kw) == expected


def test_ci_shapes_are_preserved():
    assert g.fmt_ci_like('$[-0.36, -0.03]$', -0.3559, -0.0339) == '$[-0.36, -0.03]$'
    assert g.fmt_ci_like('$[-0.36, -0.03]$', -0.42, 0.017) == '$[-0.42, +0.02]$'   # sign kept on a flip
    assert g.fmt_ci_like('$[-2.50,-0.40]$', -0.0248, -0.00399, scale=100) == '$[-2.50,-0.40]$'
    assert g.fmt_ci_like('$[0.68,0.88]$', 0.6779, 0.8813) == '$[0.68,0.88]$'       # unsigned prose CI
    assert g.fmt_ci_like('$[+0.00, +0.00]$', 0.0, 0.0) == '$[+0.00, +0.00]$'


def test_p_star_verdict():
    assert g.fmt_p_like('$0.80$', 0.803) == '$0.80$'
    rec = {'lo': 0.03, 'hi': 0.19, 'sig': True, 'p': 0.012}
    assert g.fmt_field('star', '*', rec) == '*'
    assert g.fmt_field('supstar', '^{*}', rec) == '^{*}'
    assert g.fmt_field('verdict', 'no', rec) == 'yes'
    assert g.fmt_field('star', '', dict(rec, sig=False)) == ''


# -------------------------------------------------------- round trip

FIXTURE = r"""\begingroup\color{red}
\begin{table}[H]
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccc}
\toprule
\emph{Model} & \emph{Regimen} & \emph{Accuracy} & \emph{$\Delta$} & \emph{95\% CI} & \emph{Bal.\ Acc}\\
\midrule
\multirow{3}{*}{RF + SVM} & TR/TR & $0.76$ & -- & -- & $0.75$\\
 & TS/TR & $0.56$ & $-0.20$* & $[-0.36, -0.03]$ & $0.59$\\
 & TSR/TR & $0.83$ & $+0.07$* & $[+0.02, +0.14]$ & $0.82$\\
\midrule
\multirow{3}{*}{1D-CNN} & TR/TR & $0.71$ & -- & -- & $0.71$\\
 & TS/TR & $0.54$ & $-0.17$ & $[-0.34, +0.03]$ & $0.53$\\
 & TSR/TR & $0.73$ & $+0.02$ & $[-0.10, +0.14]$ & $0.71$\\
\midrule
CbR & -- & $0.68$ & -- & $[+0.56, +0.80]$ & $0.68$\\
\bottomrule
\end{tabular*}
\caption{\small \texttt{exp\_501}: fixture.}
\label{tab:ctrl_exp_501}
\end{table}
\endgroup

\begin{table}[H]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}
\toprule
 & $40$k & $80$k & $0.50{:}1$ & $1{:}1$ & $2{:}1$ & $3{:}1$ \\
\midrule
RF & $0.83$ & $0.81$ & $0.85$ & $0.83$ & $0.85$ & $0.85$ \\
1D-CNN & $0.68$ & $0.78^{*}$ & $0.68$ & $0.75$ & $0.71$ & $0.78$ \\
\bottomrule
\end{tabular*}
\caption{\small Fixture (maximum intensity threshold = 0.60). \\ $^{*}$ \rev{The original unadjusted slide-level bootstrap interval for the 80k--40k accuracy difference excluded zero; this is an exploratory comparison (Appendix~\ref{app:controlled_experiments}).}}
\label{tab:synth_ratio_results}
\end{table}

\begin{table}[H]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}
\toprule
\emph{Model} & \emph{real} & \emph{shuf.} & \emph{$\Delta$ [95\% CI]} & \emph{real} & \emph{shuf.} & \emph{$\Delta$ [95\% CI]} \\
\midrule
mi-SVM & $0.50$ & $1.70$ & $-1.30$ $[-2.50,-0.40]$ & $0.90$ & $5.20$ & $-4.30$ $[-6.30,-2.40]$ \\
CVAE MIL (encoder head) & $5.60$ & $5.60$ & $0.00$ $[-2.80,+2.30]$ & $7.10$ & $6.40$ & $+0.60$* $[+0.10,+3.10]$ \\
% Commented-out row & $0.134$ & $0.116$ & $+0.018$ $[-0.012,+0.047]$ & $0.226$ & $0.264$ & $-0.038$ $[-0.097,+0.020]$ \\
\bottomrule
\end{tabular*}
\caption{\small Fixture spatial.}
\label{tab:ctrl_spatial_2607}
\end{table}
"""


def _convert(text, fname='results_synth_ratio_table.tex'):
    E = g.Entries()
    seen = []
    new = g.convert_text(fname, text, E, seen)
    return new, E.values, seen


def test_convert_then_expand_is_byte_identical():
    new, values, seen = _convert(FIXTURE)
    assert seen == ['tab:ctrl_exp_501', 'tab:synth_ratio_results', 'tab:ctrl_spatial_2607']
    assert new != FIXTURE
    assert g.expand_macros(new, values) == FIXTURE


def test_convert_cells_and_values():
    new, values, _ = _convert(FIXTURE)
    assert r' & TS/TR & $0.56$ & $-0.20$\BootStar{501/apollo/TSTR} & \BootCI{501/apollo/TSTR} & $0.59$\\' in new
    assert r'CbR & -- & $0.68$ & -- & \BootCI{501/cbr} & $0.68$\\' in new
    assert r'1D-CNN & $0.68$ & $0.78\BootSupStar{pool/cnn}$ &' in new
    assert (r'$0.00$\BootStar{sp/2.607/cbr_encoder_head/morans_i} '
            r'\BootCI{sp/2.607/cbr_encoder_head/morans_i}') in new
    assert r'0.60). \\ $\BootSupStar{pool/cnn}$ \rev{\BootText{ratio_star_note}}' in new
    assert '% Commented-out row & $0.134$' in new          # commented rows untouched
    assert values[('501/apollo/TSTR', 'star')] == '*'
    assert values[('501/cnn/TSTR', 'star')] == ''
    assert values[('501/apollo/TSTR', 'ci')] == '$[-0.36, -0.03]$'
    assert values[('pool/cnn', 'supstar')] == '^{*}'
    assert values[('pool/rf', 'supstar')] == ''
    assert values[('sp/2.607/cbr_encoder_head/one_minus_gamma_over_var', 'star')] == '*'
    assert not any(k.startswith('501/apollo/TRTR') for k, _ in values)


def test_values_file_roundtrip(tmp_path):
    _, values, _ = _convert(FIXTURE)
    p = tmp_path / 'boot_values_slide.tex'
    p.write_text(g.values_file_text('slide', values), encoding='utf-8')
    unit, parsed = g.parse_values_file(str(p))
    assert unit == 'slide' and parsed == values


def test_conflicting_duplicate_literal_is_rejected():
    E = g.Entries()
    E.add('608/rf/TSTR', 'ci', '$[-0.03, +0.09]$')
    E.add('608/rf/TSTR', 'ci', '$[-0.03, +0.09]$')
    with pytest.raises(ValueError):
        E.add('608/rf/TSTR', 'ci', '$[-0.03, +0.10]$')


def test_refuses_already_converted(tmp_path):
    for f in g.TARGET_FILES:
        (tmp_path / f).write_text(r'\BootCI{x}', encoding='utf-8')
    with pytest.raises(SystemExit):
        g.cmd_convert(str(tmp_path))


ZIP = os.path.join(ROOT, 'paper', 'versions', 'v2.2.1_pre_bootswitch.zip')


@pytest.mark.skipif(not (os.path.exists(ZIP) and os.path.exists(os.path.join(g.PAPER, g.SLIDE_VALUES))),
                    reason='v2.2.1 backup or converted sources not present')
def test_converted_paper_expands_to_backup():
    """The converted v2.2.1 sources + boot_values_slide.tex reproduce the
    pre-conversion files (main.tex: up to the inserted switch/comment lines)."""
    _, values = g.parse_values_file(os.path.join(g.PAPER, g.SLIDE_VALUES))
    with zipfile.ZipFile(ZIP) as z:
        for f in g.TARGET_FILES:
            orig = z.read(f'Raman synthetic data v2.2.1/{f}').decode('utf-8')
            cur = g.expand_macros(g.read(os.path.join(g.PAPER, f)), values)
            if f == 'main.tex':
                cur = cur.replace(g.MAIN_HEADER, '', 1)
                cur = cur.replace(g.MAIN_SWITCH, '', 1)
            assert cur == orig, f
