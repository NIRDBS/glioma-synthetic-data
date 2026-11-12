r"""
Swappable slide- vs patient-level bootstrap values for the v2.2.1 paper.

Every bootstrap-derived cell of the paper (CIs, Delta asterisks, reported p,
yes/no verdicts, the synthetic-ratio star) and every sentence whose wording
depends on the resampling unit is read through a macro defined in
``boot_switch.tex``; the values themselves live in ``boot_values_<unit>.tex``.
The switch is ``\def\bootunit{slide}`` in main.tex.

Modes
-----
--convert         one-off edit of the v2.2.1 .tex files: each unit-dependent
                  literal is replaced by its macro call and written verbatim to
                  boot_values_slide.tex (so slide mode is identical by
                  construction). Refuses to run on already-converted files.
--unit slide --check
                  re-format the SLIDE result JSONs with the paper formatter and
                  compare them to boot_values_slide.tex; print every mismatch.
--unit patient    write boot_values_patient.tex from results/patient_bootstrap/**
                  (same formatter, same literal shapes) and boot_unit_diff.md.
                  Missing patient results get a visible \textbf{??}; --strict
                  makes a missing result an error.

Formatter: the v2.2.1 tables were first rendered to 3 decimals and then
hand-rounded to 2 decimals (ROUND_HALF_UP on the 3-decimal string:
-0.08474576 -> '-0.085' -> '-0.09'); cells in 10^-2 units are the 3-decimal
string x 100 (0.00464 -> '0.005' -> '0.50').

Usage
-----
    python scripts/paper/gen_boot_values.py --convert
    python scripts/paper/gen_boot_values.py --unit slide --check
    python scripts/paper/gen_boot_values.py --unit patient [--strict]
"""
import argparse
import glob
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results')
PATIENT = os.path.join(RESULTS, 'patient_bootstrap')
PAPER = os.path.join(ROOT, 'Raman synthetic data v2.2.1')
SLIDE_VALUES = 'boot_values_slide.tex'
PATIENT_VALUES = 'boot_values_patient.tex'
SWITCH_FILE = 'boot_switch.tex'
DIFF_MD = 'boot_unit_diff.md'
MISSING = r'\textbf{??}'
N_SLIDES = 59

PREFIX_MODEL = {'apollo': 'Apollo', 'rf': 'RF_Standard', 'xgb': 'XGBoost', 'cnn': '1D_CNN'}
LABEL_PREFIX = {'RF + SVM': 'apollo', 'RF': 'rf', 'XGBoost': 'xgb', '1D-CNN': 'cnn', 'CbR': 'cbr'}
REGIMEN_CAND = {'TS/TR': 'TSTR', 'TSR/TR': 'TSRTR'}
CAND_REGIMEN = {v: k for k, v in REGIMEN_CAND.items()}
TIER_STEM = {'mi-SVM': 'misvm_mi_svm', 'Attention MIL': 'attention_attention',
             'CVAE MIL (reconstruction)': 'cbr_reconstruction', 'CVAE reconstruction': 'cbr_reconstruction',
             'CVAE MIL (encoder head)': 'cbr_encoder_head', 'CVAE encoder head': 'cbr_encoder_head'}
SPATIAL_METRICS = ('morans_i', 'one_minus_gamma_over_var')   # column order in the spatial tables
BETA_EXP = {'0.25': '607', '0.50': '608', '0.75': '609'}
LATENT_EXP = {'16': '701', '32': '608', '64': '702'}
LABEL_SHORT = {'607': '607', '608': '608', '609': '609', '2607': '2.607', '2608': '2.608', '4608': '4.608'}


def exp_name(short):
    return f'exp_{short}'


# ============================================================== formatter

def fmt_num(x, decimals=2, signed=True, scale=1):
    """3-decimal rendering, then ROUND_HALF_UP of that string to `decimals`
    (after multiplying by `scale`, e.g. 100 for 10^-2-unit cells)."""
    d = Decimal(f'{x:+.3f}') * scale
    q = d.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return f'{q:+.{decimals}f}' if signed else f'{q:.{decimals}f}'


CI_RE = re.compile(r'^(?P<pre>\$?)\[(?P<lo>[+-]?\d+\.(?P<dec>\d+))(?P<sep>,\s*)'
                   r'(?P<hi>[+-]?\d+\.\d+)\](?P<post>\$?)$')
P_RE = re.compile(r'^\$(?P<num>\d+\.(?P<dec>\d+))\$$')


def ci_shape(literal):
    m = CI_RE.match(literal)
    if not m:
        raise ValueError(f'not a CI literal: {literal!r}')
    return dict(pre=m['pre'], post=m['post'], sep=m['sep'], dec=len(m['dec']),
                signed=m['lo'][0] in '+-' or m['hi'][0] in '+-')   # any explicit sign -> signed


def fmt_ci(lo, hi, shape, scale=1):
    f = lambda v: fmt_num(v, shape['dec'], shape['signed'], scale)
    return f"{shape['pre']}[{f(lo)}{shape['sep']}{f(hi)}]{shape['post']}"


def fmt_ci_like(literal, lo, hi, scale=1):
    return fmt_ci(lo, hi, ci_shape(literal), scale)


def fmt_p_like(literal, p):
    m = P_RE.match(literal)
    if not m:
        raise ValueError(f'not a p literal: {literal!r}')
    return f'${fmt_num(p, len(m["dec"]), signed=False)}$'


STAR_SYNTAX = {'star': '*', 'supstar': '^{*}'}


def fmt_field(field, literal, rec, scale=1):
    """Format one (field) of a result record `rec` in the shape of `literal`."""
    if field == 'ci':
        return fmt_ci_like(literal, rec['lo'], rec['hi'], scale)
    if field in STAR_SYNTAX:
        return STAR_SYNTAX[field] if rec['sig'] else ''
    if field == 'p':
        return fmt_p_like(literal, rec['p'])
    if field == 'verdict':
        return 'yes' if rec['sig'] else 'no'
    raise ValueError(field)


# ============================================================ tex macros

MACRO_FIELD = {'CI': 'ci', 'Star': 'star', 'SupStar': 'supstar', 'P': 'p', 'Verdict': 'verdict'}
MACRO_RE = re.compile(r'\\Boot(CI|SupStar|Star|P|Verdict|Text)\{([^{}]*)\}')


def macro(kind, key):
    return rf'\Boot{kind}{{{key}}}'


def text_key(name):
    return f'text/{name}'


def expand_macros(text, values):
    """Replace every \\Boot...{key} call by its value (values: {(key, field): str})."""
    def rep(m):
        kind, key = m.group(1), m.group(2)
        if kind == 'Text':
            return values[(text_key(key), 'text')]
        return values[(key, MACRO_FIELD[kind])]
    return MACRO_RE.sub(rep, text)


# ========================================================= converter core

class Entries:
    """Ordered (key, field) -> literal, refusing conflicting duplicates."""

    def __init__(self):
        self.values = {}

    def add(self, key, field, literal):
        k = (key, field)
        if k in self.values and self.values[k] != literal:
            raise ValueError(f'{key}/{field}: conflicting literals {self.values[k]!r} vs {literal!r}')
        self.values.setdefault(k, literal)


TABLE_RE = re.compile(r'\\begin\{table\}.*?\\end\{table\}', re.S)
ROW_RE = re.compile(r'^(?P<body>.*?)(?P<tail>\s*\\\\\s*)$')
CELL_RE = re.compile(r'^(?P<l>\s*)(?P<core>.*?)(?P<r>\s*)$', re.S)
MULTIROW_RE = re.compile(r'^\\multirow\{\d+\}\{\*\}\{(?P<x>.*)\}$')


class Row:
    def __init__(self, line):
        m = ROW_RE.match(line)
        self.tail = m['tail']
        self.cells = []
        for c in m['body'].split('&'):
            cm = CELL_RE.match(c)
            self.cells.append([cm['l'], cm['core'], cm['r']])

    def __getitem__(self, i):
        return self.cells[i][1]

    def __setitem__(self, i, v):
        self.cells[i][1] = v

    def __len__(self):
        return len(self.cells)

    def text(self):
        return '&'.join(''.join(c) for c in self.cells) + self.tail


def is_data_row(line):
    s = line.strip()
    return ('&' in s and s.endswith('\\\\') and not s.startswith('%') and r'\emph{' not in s)


def unmulti(core):
    m = MULTIROW_RE.match(core)
    return m['x'] if m else core


def process_rows(block, handler):
    """Apply handler(row, state) to every data row of a table block; the handler
    edits the row in place. `state` carries the multirow model/regimen."""
    lines = block.split('\n')
    state = {}
    for i, line in enumerate(lines):
        if not is_data_row(line):
            continue
        row = Row(line)
        handler(row, state)
        new = row.text()
        lines[i] = new
    return '\n'.join(lines)


def carry(state, name, core):
    v = unmulti(core)
    if v:
        state[name] = v
    return state.get(name)


DELTA_STAR_RE = re.compile(r'^(?P<num>\$[+-]?\d+\.\d+\$)(?P<star>\*?)$')
SP_CELL_RE = re.compile(r'^(?P<num>\$[+-]?\d+\.\d+\$)(?P<star>\*?)(?P<sp>\s+)(?P<ci>\$\[[^\]]*\]\$)$')


def _delta_star(row, i, key, E):
    m = DELTA_STAR_RE.match(row[i])
    if not m:
        raise ValueError(f'{key}: unexpected Delta cell {row[i]!r}')
    E.add(key, 'star', m['star'])
    row[i] = m['num'] + macro('Star', key)


def _ci(row, i, key, E):
    ci_shape(row[i])   # validates
    E.add(key, 'ci', row[i])
    row[i] = macro('CI', key)


def _paired_cell(row, i, key, E, with_star=True):
    m = SP_CELL_RE.match(row[i])
    if not m:
        raise ValueError(f'{key}: unexpected Delta [CI] cell {row[i]!r}')
    E.add(key, 'ci', m['ci'])
    if with_star:
        E.add(key, 'star', m['star'])
        row[i] = m['num'] + macro('Star', key) + m['sp'] + macro('CI', key)
    else:
        if m['star']:
            raise ValueError(f'{key}: unexpected star in a star-less table')
        row[i] = m['num'] + m['sp'] + macro('CI', key)


def regimen_handler(short, E, has_beta=False):
    """Per-experiment classifier table (Model | Regimen | [beta |] Acc | Delta | CI | ...)."""
    off = 1 if has_beta else 0

    def h(row, state):
        model = carry(state, 'model', row[0])
        reg = carry(state, 'reg', row[1])
        prefix = LABEL_PREFIX[model]
        if reg == 'TR/TR':
            return
        exp = BETA_EXP[row[2].strip('$')] if has_beta else short
        if prefix == 'cbr':
            _ci(row, 4 + off, f'{exp}/cbr', E)
            return
        key = f'{exp}/{prefix}/{REGIMEN_CAND[reg]}'
        _delta_star(row, 3 + off, key, E)
        _ci(row, 4 + off, key, E)
    return h


def latent_handler(E):
    def h(row, state):
        prefix = LABEL_PREFIX[carry(state, 'model', row[0])]
        a, b = re.match(r'^\$(\d+) \\to (\d+)\$$', row[1]).groups()
        pair = f'{LATENT_EXP[a]}-{LATENT_EXP[b]}'
        for cand, (di, ci) in (('TSTR', (2, 3)), ('TSRTR', (4, 5))):
            key = f'lat/{cand}/{prefix}/{pair}'
            _delta_star(row, di, key, E)
            _ci(row, ci, key, E)
    return h


def pool_handler(E):
    def h(row, state):
        if row[0] not in LABEL_PREFIX:     # column-header row
            return
        key = f'pool/{LABEL_PREFIX[row[0]]}'
        _delta_star(row, 3, key, E)
        _ci(row, 4, key, E)
        if not P_RE.match(row[5]):
            raise ValueError(f'{key}: p cell {row[5]!r}')
        E.add(key, 'p', row[5])
        row[5] = macro('P', key)
    return h


def ratio_handler(E):
    def h(row, state):
        if row[0] not in LABEL_PREFIX:     # column-header row
            return
        key = f'pool/{LABEL_PREFIX[row[0]]}'
        m = re.match(r'^\$(?P<num>\d+\.\d+)(?P<star>(\^\{\*\})?)\$$', row[2])
        if not m:
            raise ValueError(f'{key}: 80k cell {row[2]!r}')
        E.add(key, 'supstar', m['star'])
        row[2] = '$' + m['num'] + macro('SupStar', key) + '$'
    return h


def spatial_handler(short, E, status_table=False):
    def h(row, state):
        stem = TIER_STEM[row[0]]
        for metric, i in zip(SPATIAL_METRICS, (3, 6)):
            _paired_cell(row, i, f'sp/{short}/{stem}/{metric}', E, with_star=not status_table)
        if status_table:
            key = f'sp/{short}/{stem}'
            if row[7] not in ('yes', 'no'):
                raise ValueError(f'{key}: verdict cell {row[7]!r}')
            E.add(key, 'verdict', row[7])
            row[7] = macro('Verdict', key)
    return h


def keyinst_handler(short, E):
    def h(row, state):
        stem = TIER_STEM.get(row[0])
        if stem is None:          # naive-MIL benchmark row: no test
            return
        _paired_cell(row, 2, f'ki/{short}/{stem}/vs_naive', E)
    return h


def handler_for(label, E):
    if label == 'tab:reference_results':
        return regimen_handler('608', E)
    if label == 'tab:beta_results':
        return regimen_handler(None, E, has_beta=True)
    if label.startswith('tab:ctrl_exp_'):
        return regimen_handler(label[len('tab:ctrl_exp_'):], E)
    if label == 'tab:ctrl_boot_latent':
        return latent_handler(E)
    if label == 'tab:ctrl_boot_pool':
        return pool_handler(E)
    if label == 'tab:synth_ratio_results':
        return ratio_handler(E)
    if label == 'tab:mil_tests_status':
        return spatial_handler('608', E, status_table=True)
    if label.startswith('tab:ctrl_spatial_'):
        return spatial_handler(LABEL_SHORT[label[len('tab:ctrl_spatial_'):]], E)
    if label.startswith('tab:ctrl_keyinst_'):
        return keyinst_handler(LABEL_SHORT[label[len('tab:ctrl_keyinst_'):]], E)
    return None


def convert_tables(text, E, seen_labels=None):
    def rep(m):
        block = m.group(0)
        labels = re.findall(r'\\label\{([^}]*)\}', block)
        if not labels:
            return block
        h = handler_for(labels[0], E)
        if h is None:
            return block
        if seen_labels is not None:
            seen_labels.append(labels[0])
        return process_rows(block, h)
    return TABLE_RE.sub(rep, text)


# ------------------------------------------------- unit-dependent wording
# (file, name, exact slide literal, expected count, what the patient sentence must say)
WORDING = [
    ('main.tex', 'stat_unit',
     r'The reported intervals were computed using a class-stratified paired bootstrap over slides.', 1,
     'intervals from a class-stratified paired bootstrap over patients (46 patients, 59 slides), '
     'all slides of a drawn patient kept together'),
    ('main.tex', 'stat_caveats',
     r'Asterisks identify intervals excluding zero under this original procedure. They do not establish '
     r'superiority: slides from the same patient are dependent, the intervals do not include retraining '
     r'variability, and no correction was applied for multiple comparisons.', 1,
     'asterisks mark patient-level intervals excluding zero; still exploratory: patient dependence now '
     'resampled, but no retraining variability and no multiplicity correction'),
    ('main.tex', 'ref_note_unit',
     r'using the slide-level bootstrap in Section~\ref{sec:statistical_eval}', 1,
     'name the patient-level bootstrap and point to the statistical-evaluation section'),
    ('main.tex', 'ref_note_star',
     r'An asterisk marks an interval excluding zero under the original slide-resampling procedure; it is '
     r'an exploratory flag, not a validated superiority claim. Patient dependence and retraining '
     r'variability were not included, and no multiplicity correction was applied.', 1,
     'asterisk = patient-level interval excluding zero, exploratory; retraining variability and '
     'multiplicity still not addressed'),
    ('main.tex', 'boot_over_unit', r'a bootstrap over slides', 1,
     'a class-stratified bootstrap over patients'),
    ('main.tex', 'disc_limitations',
     r'Slide-resampling intervals omit dependence within patients and model-retraining variability, '
     r'while numerous comparisons increase the risk of chance findings.', 1,
     'patient-resampling intervals account for within-patient dependence but omit retraining '
     'variability; numerous comparisons'),
    ('appendix_controlled_experiments.tex', 'app_ci_unit',
     r'its original slide-level bootstrap 95\% confidence interval', 1,
     'its patient-level bootstrap 95 percent confidence interval'),
    ('appendix_controlled_experiments.tex', 'app_star',
     r'An asterisk records an interval excluding zero under the original unadjusted procedure; these '
     r'comparisons are exploratory and do not establish superiority.', 1,
     'asterisk = unadjusted patient-level interval excluding zero; exploratory'),
    ('appendix_controlled_experiments.tex', 'app_intervals_unit',
     r'The intervals condition on the fitted models and resample slides rather than patients, so they do '
     r'not account for dependence between slides from the same patient or variability from retraining.', 1,
     'intervals condition on the fitted models and resample patients (slides of a patient together); '
     'retraining variability not included'),
    ('appendix_controlled_experiments.tex', 'app_spatial_resampling',
     r'The 95\% percentile CI used $10{,}000$ bootstrap resamples of these differences, with seed $42$.', 1,
     'percentile CI from 10,000 class-stratified resamples of patients (per-slide differences of a '
     'drawn patient kept together), seed 42'),
    ('appendix_controlled_experiments.tex', 'app_spatial_criterion',
     r'This rule used unadjusted slide-level intervals and was not a validation of diagnostic tissue '
     r'selection.', 1,
     'rule uses unadjusted patient-level intervals; not a validation of diagnostic tissue selection'),
    ('appendix_controlled_experiments_tables.tex', 'pairwise_resampling',
     r'class-stratified resampling of the $n=59$ paired bag decisions', 1,
     'class-stratified resampling of the 46 patients, carrying their paired bag decisions'),
    ('results_beta_table.tex', 'star_def_excl',
     r'(*) records an original unadjusted slide-level CI excluding zero; comparisons are exploratory and '
     r'do not establish superiority.', 1,
     '(*) = unadjusted patient-level CI excluding zero; exploratory, no superiority'),
    ('appendix_bootstrap_tables.tex', 'latent_caption_unit',
     r'the class-stratified bootstrap over the $n=59$ bag decisions', 1,
     'the class-stratified bootstrap over the 46 patients (59 bag decisions)'),
    ('appendix_bootstrap_tables.tex', 'star_def_excl_short',
     r'(*) records an original unadjusted slide-level CI excluding zero; comparisons are exploratory.', 2,
     '(*) = unadjusted patient-level CI excluding zero; exploratory'),
    ('appendix_bootstrap_tables.tex', 'pool_note_p',
     r'Reported $p$-values are retained from the original analysis and require verification of the '
     r'calculation and treatment of zero differences (Section~\ref{sec:statistical_eval}). They are not '
     r'used here to establish superiority.', 1,
     'p-values from the patient-level bootstrap; state the p definition and zero-difference treatment; '
     'not used to establish superiority'),
    ('results_synth_ratio_table.tex', 'ratio_star_note',
     r'The original unadjusted slide-level bootstrap interval for the 80k--40k accuracy difference '
     r'excluded zero; this is an exploratory comparison (Appendix~\ref{app:controlled_experiments}).', 1,
     'footnote text for the 80k-40k star (its marker follows the 1D-CNN star automatically): '
     'patient-level interval; empty the footnote if no star remains'),
    ('results_mil_appendix_tables.tex', 'sp_caption_unit', r'with its 95\% bootstrap CI over slides', 6,
     'with its 95 percent class-stratified bootstrap CI over patients'),
    ('results_mil_appendix_tables.tex', 'sp_caption_criterion',
     r'Both intervals had to be above zero to meet the descriptive coherence criterion; this unadjusted '
     r'slide-level comparison does not validate diagnostic tissue selection.', 6,
     'both patient-level intervals above zero for the descriptive criterion; does not validate '
     'diagnostic tissue selection'),
    ('results_mil_appendix_tables.tex', 'boot_over_unit', r'a bootstrap over slides', 6,
     'a class-stratified bootstrap over patients'),
    ('results_mil_appendix_tables.tex', 'ki_caption_star',
     r'(*) records an original unadjusted slide-level CI above zero; it is an exploratory marker rather '
     r'than a validated superiority claim.', 6,
     '(*) = unadjusted patient-level CI above zero; exploratory marker'),
]

# Values outside table cells: (file, key, macro kind, exact context with @@ where the literal
# goes, literal). The synthetic-ratio footnote marker follows the 1D-CNN 80k-vs-40k star, so
# it disappears together with the star in the table cell.
PROSE = [
    ('main.tex', 'prose/cbr_608', 'CI',
     r'The supporting CbR analysis reported accuracy $0.78$, with a $95\%$ CI of @@.', r'$[0.68,0.88]$'),
    ('main.tex', 'prose/rf_xgb_tstr_608', 'CI',
     r'Their reported accuracy-difference intervals, @@, included zero.', r'$[-0.03,+0.09]$'),
    ('results_synth_ratio_table.tex', 'pool/cnn', 'SupStar',
     r'maximum intensity threshold = 0.60). \\ $@@$ \rev{', r'^{*}'),
]

TARGET_FILES = ['main.tex', 'appendix_controlled_experiments.tex', 'appendix_controlled_experiments_tables.tex',
                'results_beta_table.tex', 'appendix_bootstrap_tables.tex', 'results_synth_ratio_table.tex',
                'results_mil_appendix_tables.tex', 'supporting_methods.tex']

EXPECTED_TABLES = (['tab:reference_results', 'tab:mil_tests_status', 'tab:beta_results',
                    'tab:ctrl_boot_latent', 'tab:ctrl_boot_pool', 'tab:synth_ratio_results']
                   + [f'tab:ctrl_exp_{e}' for e in ('501', '601', '701', '702', '703', '704', '705', '706',
                                                    '4.608', '2.608', '2.607', '2.609')]
                   + [f'tab:ctrl_{k}_{s}' for k in ('spatial', 'keyinst') for s in LABEL_SHORT])

MAIN_HEADER = r"""% ============================================================================
% BOOTSTRAP UNIT SWITCH. Every bootstrap-derived value (95% CIs, Delta asterisks,
% reported p-values, yes/no verdicts, the synthetic-ratio star) and every sentence
% whose wording depends on the resampling unit is read from boot_values_<unit>.tex
% through the macros of boot_switch.tex. Choose the unit with \def\bootunit{...}
% further down this preamble (search for "THE switch"):
%   slide   = original class-stratified bootstrap over the 59 slides
%   patient = class-stratified bootstrap over the 46 patients
% Values are written by scripts/paper/gen_boot_values.py (--unit slide --check / --unit patient).
% ============================================================================
"""

MAIN_SWITCH = r"""
% ---- Bootstrap resampling unit (see the header comment of this file) ----
\def\bootunit{slide}% slide | patient   <-- THE switch (the only place to change)
\input{boot_switch}
"""

MAIN_SWITCH_ANCHOR = r'\newcommand{\TODO}[1]{\ifdraft{\color{red}[TODO: #1]}\fi}' + '\n'

SWITCH_TEX = r"""% boot_switch.tex -- accessors for the slide- vs patient-level bootstrap values.
% The unit is chosen by \def\bootunit{slide|patient} in main.tex, just before
% \input{boot_switch}. The values live in boot_values_<unit>.tex, written by
% scripts/paper/gen_boot_values.py. Do not edit the values files by hand.
\makeatletter
\ifdefined\bootunit\else
  \PackageError{bootswitch}{\string\bootunit\space is not defined}%
    {Put \string\def\string\bootunit{slide} (or patient) before \string\input{boot_switch}.}%
  \def\bootunit{slide}%
\fi
% \bootset{<key>}{<field>}{<value>}: store one value (used by the values files).
\def\bootset#1#2#3{\expandafter\def\csname boot@#1@#2\endcsname{#3}}
% \bootget{<key>}{<field>}: typeset one value; an undefined value is an error.
\protected\def\bootget#1#2{%
  \ifcsname boot@#1@#2\endcsname
    \expandafter\@firstoftwo
  \else
    \expandafter\@secondoftwo
  \fi
  {\csname boot@#1@#2\endcsname}%
  {\PackageError{bootswitch}{Undefined bootstrap value #1/#2}{}}}
% Thin wrappers (protected: safe in tabular cells, \caption, \multicolumn, math).
\protected\def\BootCI#1{\bootget{#1}{ci}}
\protected\def\BootStar#1{\bootget{#1}{star}}
\protected\def\BootSupStar#1{\bootget{#1}{supstar}}
\protected\def\BootP#1{\bootget{#1}{p}}
\protected\def\BootVerdict#1{\bootget{#1}{verdict}}
\protected\def\BootText#1{\bootget{text/#1}{text}}
% In PDF strings (bookmarks) the accessors expand to nothing: do not put Boot
% macros in sectioning titles, or their values vanish from the bookmarks.
\ifdefined\pdfstringdefDisableCommands
  \pdfstringdefDisableCommands{\def\BootCI#1{}\def\BootStar#1{}\def\BootSupStar#1{}%
    \def\BootP#1{}\def\BootVerdict#1{}\def\BootText#1{}}%
\fi
\makeatother
\input{boot_values_\bootunit}
\AtBeginDocument{%
  \ifx\BootLoadedUnit\bootunit\else
    \PackageError{bootswitch}{boot_values file is for unit '\BootLoadedUnit'
      but \string\bootunit\space is '\bootunit'}{}%
  \fi}
"""


def convert_text(fname, text, E, seen_labels=None):
    """Convert one file's text (pure function of the text, for testing)."""
    new = convert_tables(text, E, seen_labels)
    for f, name, literal, count, _todo in WORDING:
        if f != fname:
            continue
        n = new.count(literal)
        if n != count:
            raise ValueError(f'{fname}: wording {name!r} found {n}x, expected {count}x')
        E.add(text_key(name), 'text', literal)
        new = new.replace(literal, macro('Text', name))
    for f, key, kind, ctx, literal in PROSE:
        if f != fname:
            continue
        old = ctx.replace('@@', literal)
        if new.count(old) != 1:
            raise ValueError(f'{fname}: prose {key} context found {new.count(old)}x')
        E.add(key, MACRO_FIELD[kind], literal)
        new = new.replace(old, ctx.replace('@@', macro(kind, key)))
    return new


def read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def write(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def values_file_text(unit, values, header=''):
    L = [f'% boot_values_{unit}.tex -- written by scripts/paper/gen_boot_values.py; do not edit by hand.',
         f'% Resampling unit: {unit}. One \\bootset{{<key>}}{{<field>}}{{<value>}} per value.']
    if header:
        L += ['% ' + h for h in header.splitlines()]
    L.append(f'\\def\\BootLoadedUnit{{{unit}}}')
    for (key, field), v in values.items():
        if '\n' in v:
            raise ValueError(f'{key}/{field}: value contains a newline')
        L.append(f'\\bootset{{{key}}}{{{field}}}{{{v}}}')
    return '\n'.join(L) + '\n'


BOOTSET_RE = re.compile(r'^\\bootset\{(?P<key>[^{}]*)\}\{(?P<field>[^{}]*)\}\{(?P<val>.*)\}$')


def parse_values_file(path):
    vals, unit = {}, None
    for line in read(path).splitlines():
        m = BOOTSET_RE.match(line)
        if m:
            vals[(m['key'], m['field'])] = m['val']
        mu = re.match(r'^\\def\\BootLoadedUnit\{(\w+)\}$', line)
        if mu:
            unit = mu.group(1)
    return unit, vals


def cmd_convert(paper=PAPER):
    if os.path.exists(os.path.join(paper, SLIDE_VALUES)):
        sys.exit(f'refusing: {SLIDE_VALUES} already exists (files already converted)')
    texts = {f: read(os.path.join(paper, f)) for f in TARGET_FILES}
    for f, t in texts.items():
        if MACRO_RE.search(t) or r'\input{boot_switch}' in t:
            sys.exit(f'refusing: {f} already contains \\Boot macros (files already converted)')
    E, seen = Entries(), []
    new = {f: convert_text(f, t, E, seen) for f, t in texts.items()}
    missing_tables = [t for t in EXPECTED_TABLES if t not in seen]
    if missing_tables:
        sys.exit(f'tables not found: {missing_tables}')
    # self-check: expanding the slide values restores every file byte for byte
    for f in TARGET_FILES:
        if expand_macros(new[f], E.values) != texts[f]:
            sys.exit(f'self-check failed for {f}')
    main = new['main.tex']
    if main.count(MAIN_SWITCH_ANCHOR) != 1:
        sys.exit('main.tex: \\TODO definition (switch anchor) not found exactly once')
    main = main.replace(MAIN_SWITCH_ANCHOR, MAIN_SWITCH_ANCHOR + MAIN_SWITCH, 1)
    first_nl = main.index('\n') + 1          # after \documentclass
    if not main.startswith(r'\documentclass'):
        sys.exit('main.tex does not start with \\documentclass')
    new['main.tex'] = main[:first_nl] + MAIN_HEADER + main[first_nl:]
    write(os.path.join(paper, SLIDE_VALUES), values_file_text('slide', E.values,
          'Literal strings of the v2.2.1 sources, replaced verbatim by --convert.'))
    write(os.path.join(paper, SWITCH_FILE), SWITCH_TEX)
    changed = []
    for f in TARGET_FILES:
        if new[f] != texts[f]:
            write(os.path.join(paper, f), new[f])
            changed.append(f)
    counts = {}
    for (key, field) in E.values:
        kind = key.split('/')[0] if not re.match(r'^\d', key) else ('cbr' if key.endswith('/cbr') else 'regimen')
        counts[f'{kind}:{field}'] = counts.get(f'{kind}:{field}', 0) + 1
    print(f'converted {len(changed)} files: {", ".join(changed)}')
    print(f'{len(E.values)} values written to {SLIDE_VALUES}')
    for k in sorted(counts):
        print(f'  {k:28s} {counts[k]}')
    print(f'tables converted: {len(seen)}')


# ======================================================== result sources

_cache = {}


def _json(path):
    if path not in _cache:
        _cache[path] = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None
    return _cache[path]


def latest_slide(exp, cand):
    """Same rule as gen_appendix_controlled_experiments.latest()."""
    files = glob.glob(os.path.join(RESULTS, exp, 'statistical_eval', f'statistical_evaluation_TRTR_vs_{cand}_*.json'))
    if not files:
        return None
    files.sort(key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    return _json(files[-1])


def regimen_json(unit, exp, cand):
    if unit == 'slide':
        return latest_slide(exp, cand)
    return _json(os.path.join(PATIENT, exp, f'regimen_TRTR_vs_{cand}.json'))


def _rec(lo, hi, sig=None, p=None, **extra):
    r = dict(lo=lo, hi=hi, sig=sig, p=p)
    r.update(extra)
    return r


def _regimen_row(unit, exp, prefix, cand):
    js = regimen_json(unit, exp, cand)
    if not js:
        return None
    for r in js['results']:
        if r['model'] == PREFIX_MODEL[prefix]:
            return r
    return None


def _cbr(unit, exp):
    for cand in ('TSRTR', 'TSTR'):   # as gen_appendix_controlled_experiments.cbr_row
        js = regimen_json(unit, exp, cand)
        if js and js.get('cbr'):
            return js['cbr']
    return None


def resolve(key, field, unit):
    """Result record for (key, field) under `unit`, or None if not available.
    Records: lo, hi (CI), sig (star/verdict), p; patient records also carry
    'strat' = the class-stratified slide-cluster CI (lo, hi) where computed."""
    parts = key.split('/')
    if re.match(r'^\d', key):
        exp = exp_name(parts[0])
        if parts[1] == 'cbr':
            c = _cbr(unit, exp)
            return c and _rec(c['cbr_bag_ci_lo'], c['cbr_bag_ci_hi'])
        r = _regimen_row(unit, exp, parts[1], parts[2])
        return r and _rec(r['bag_ci_diff_lo'], r['bag_ci_diff_hi'], r['bag_significant_95'], r['bag_p_bootstrap'])
    kind = parts[0]
    if kind == 'prose':
        if parts[1] == 'cbr_608':
            return resolve('608/cbr', field, unit)
        if parts[1] == 'rf_xgb_tstr_608':
            a, b = resolve('608/rf/TSTR', field, unit), resolve('608/xgb/TSTR', field, unit)
            if a is None or b is None:
                return None
            return dict(a, other=b)
        raise KeyError(key)
    if kind in ('lat', 'pool'):
        if kind == 'lat':
            cand, prefix, pair = parts[1], parts[2], parts[3]
            base, candexp = (exp_name(x) for x in pair.split('-'))
            path = (os.path.join(RESULTS, 'latent_dim_bootstrap', 'pairwise_latent_dim_bootstrap.json')
                    if unit == 'slide' else os.path.join(PATIENT, 'pairwise_latent_dim.json'))
            match = lambda t: (t['regimen'] == CAND_REGIMEN[cand] and t['model'] == PREFIX_MODEL[prefix]
                               and t['baseline_exp'] == base and t['candidate_exp'] == candexp)
        else:
            prefix = parts[1]
            path = (os.path.join(RESULTS, 'synth_pool_bootstrap', 'pairwise_pool_size_bootstrap.json')
                    if unit == 'slide' else os.path.join(PATIENT, 'pairwise_pool_size.json'))
            match = lambda t: t['model'] == PREFIX_MODEL[prefix]
        js = _json(path)
        if not js:
            return None
        hits = [t for t in js['tests'] if match(t)]
        if len(hits) != 1:
            return None
        t = hits[0]
        return _rec(t['ci_lo'], t['ci_hi'], t['significant_95'], t['p_bootstrap'])
    if kind == 'sp':
        exp, stem = exp_name(parts[1]), parts[2]
        path = (os.path.join(RESULTS, exp, 'spatial_validation', 'spatial_paired_real_vs_shuffled.json')
                if unit == 'slide' else os.path.join(PATIENT, exp, 'spatial_paired.json'))
        js = _json(path)
        fit = js and js['fits'].get(stem)
        if not fit:
            return None
        if len(parts) == 3:        # verdict: both statistics starred
            recs = [resolve(f'sp/{parts[1]}/{stem}/{m}', 'star', unit) for m in SPATIAL_METRICS]
            if any(r is None for r in recs):
                return None
            out = _rec(None, None, all(r['sig'] for r in recs))
            if all('strat' in r for r in recs):
                out['strat_sig'] = all(r['strat'][0] > 0 for r in recs)
            return out
        t = fit[parts[3]]
        r = _rec(t['ci_lo'], t['ci_hi'], t['significant_95_real_higher'])
        if 'slide_stratified' in t:
            r['strat'] = (t['slide_stratified']['ci_lo'], t['slide_stratified']['ci_hi'])
        return r
    if kind == 'ki':
        exp, stem, vs = exp_name(parts[1]), parts[2], parts[3]
        path = (os.path.join(RESULTS, exp, 'mil_eval', 'key_instance_paired.json')
                if unit == 'slide' else os.path.join(PATIENT, exp, 'key_instance_paired.json'))
        js = _json(path)
        tier = js and js['tiers'].get(stem)
        if not tier:
            return None
        t = tier[vs]
        r = _rec(t['ci_lo'], t['ci_hi'], t['significant_95_selected_higher'])
        if 'slide_stratified' in t:
            r['strat'] = (t['slide_stratified']['ci_lo'], t['slide_stratified']['ci_hi'])
        return r
    raise KeyError(key)


def scale_of(key):
    return 100 if key.startswith('sp/') else 1


def format_value(key, field, literal, rec):
    if key == 'prose/rf_xgb_tstr_608':
        a = fmt_ci_like(literal, rec['lo'], rec['hi'])
        b = fmt_ci_like(literal, rec['other']['lo'], rec['other']['hi'])
        if a == b:
            return a
        return (rf'RF {a}, XGBoost {b} \textcolor{{red}}{{[TODO patient-level wording: RF and XGBoost '
                rf'intervals now differ; reword the sentence]}}')
    return fmt_field(field, literal, rec, scale_of(key))


# ================================================================ check

def cmd_check(paper=PAPER):
    unit, vals = parse_values_file(os.path.join(paper, SLIDE_VALUES))
    assert unit == 'slide', unit
    mism, missing, n = [], [], 0
    for (key, field), lit in vals.items():
        if field == 'text':
            continue
        n += 1
        rec = resolve(key, field, 'slide')
        if rec is None:
            missing.append((key, field))
            continue
        got = format_value(key, field, lit, rec)
        if got != lit:
            mism.append((key, field, lit, got))
    print(f'checked {n} slide values against the slide JSONs: {len(mism)} mismatch(es), '
          f'{len(missing)} unresolved')
    for key, field, lit, got in mism:
        print(f'  MISMATCH {key}/{field}: literal {lit!r} vs formatted {got!r}')
    for key, field in missing:
        print(f'  UNRESOLVED {key}/{field}')
    return mism, missing


# ============================================================== patient

def _plain(s):
    return s.replace('$', '') if s else s


def _excl0(lo, hi):
    return lo > 0 or hi < 0


ACCURACY_KINDS = ('regimen', 'lat', 'pool', 'ki', 'prose')


def _kind(key):
    if re.match(r'^\d', key):
        return 'cbr' if key.endswith('/cbr') else 'regimen'
    return key.split('/')[0]


def cmd_patient(paper=PAPER, strict=False):
    unit, slide_vals = parse_values_file(os.path.join(paper, SLIDE_VALUES))
    assert unit == 'slide', unit
    todo = {name: desc for _f, name, _lit, _c, desc in WORDING}
    out, missing = {}, []
    for (key, field), lit in slide_vals.items():
        if field == 'text':
            name = key.split('/', 1)[1]
            out[(key, field)] = rf'\textcolor{{red}}{{[TODO patient-level wording: {todo[name]}]}}'
            continue
        rec = resolve(key, field, 'patient')
        if rec is None:
            missing.append((key, field))
            out[(key, field)] = MISSING
            continue
        out[(key, field)] = format_value(key, field, lit, rec)
    if missing and strict:
        sys.exit('missing patient results (--strict):\n' + '\n'.join(f'  {k}/{f}' for k, f in missing))
    write(os.path.join(paper, PATIENT_VALUES), values_file_text('patient', out,
          'Formatted from results/patient_bootstrap/** with the slide literals\' shapes.\n'
          f'Missing results ({len(missing)}) are shown as \\textbf{{??}}; wording keys are red [TODO ...] placeholders, '
          f'shown regardless of \\ifdraft.'))
    write_diff(paper, slide_vals, out, missing)
    print(f'wrote {PATIENT_VALUES}: {len(out)} values, {len(missing)} missing')
    for k, f in missing:
        print(f'  MISSING {k}/{f}')


def write_diff(paper, slide_vals, patient_vals, missing):
    keys = []
    for (key, field) in slide_vals:
        if field != 'text' and key not in keys:
            keys.append(key)
    miss_keys = sorted({k for k, _ in missing}, key=keys.index)
    rows, n_star, n_verd, n_zero, n_border, n_avail = [], 0, 0, 0, 0, 0
    for key in keys:
        fields = [f for (k, f) in slide_vals if k == key]
        def cell(vals, f):
            return vals.get((key, f), '')
        s_ci, p_ci = _plain(cell(slide_vals, 'ci')), _plain(cell(patient_vals, 'ci'))
        s_mark = '*' if any(cell(slide_vals, f) for f in ('star', 'supstar')) else ''
        p_mark = '*' if any(cell(patient_vals, f) for f in ('star', 'supstar')) else ''
        s_txt = ' '.join(x for x in (s_ci, s_mark and f'star {s_mark}', cell(slide_vals, 'p') and f"p={_plain(cell(slide_vals, 'p'))}",
                                     cell(slide_vals, 'verdict') and f"verdict {cell(slide_vals, 'verdict')}") if x)
        p_txt = ' '.join(x for x in (p_ci, p_mark and f'star {p_mark}', cell(patient_vals, 'p') and f"p={_plain(cell(patient_vals, 'p'))}",
                                     cell(patient_vals, 'verdict') and f"verdict {cell(patient_vals, 'verdict')}") if x)
        srec = resolve(key, fields[0], 'slide')
        prec = resolve(key, fields[0], 'patient')
        if prec is None:
            p_txt = '(missing)'
        flags = []
        strat = ''
        if prec is None:
            flags.append('MISSING')
        else:
            n_avail += 1
            if 'strat' in prec and 'ci' in fields:
                strat = _plain(fmt_ci_like(slide_vals[(key, 'ci')], *prec['strat'], scale=scale_of(key)))
            elif 'strat_sig' in prec:
                strat = f"verdict {'yes' if prec['strat_sig'] else 'no'}"
            elif _kind(key) in ('regimen', 'cbr', 'lat', 'pool', 'prose'):
                strat = '= slide'
            if any(f in fields for f in ('star', 'supstar')) and bool(srec['sig']) != bool(prec['sig']):
                flags.append('STAR CHANGED')
                n_star += 1
            if 'verdict' in fields and bool(srec['sig']) != bool(prec['sig']):
                flags.append('VERDICT CHANGED')
                n_verd += 1
            if 'ci' in fields and _kind(key) != 'cbr':
                if _excl0(srec['lo'], srec['hi']) != _excl0(prec['lo'], prec['hi']):
                    flags.append('zero-crossing changed')
                    n_zero += 1
                if _kind(key) in ACCURACY_KINDS:
                    b = [u for u, r in (('slide', srec), ('patient', prec))
                         if min(abs(r['lo']), abs(r['hi'])) <= 1 / N_SLIDES + 1e-9]
                    if b:
                        flags.append('borderline endpoint (' + ', '.join(b) + ')')
                        n_border += 1
            if key == 'prose/rf_xgb_tstr_608' and 'TODO' in patient_vals[(key, 'ci')]:
                flags.append('RF/XGBoost intervals differ')
        rows.append(f'| `{key}` | {s_txt} | {strat} | {p_txt} | {"; ".join(flags)} |')
    L = ['# Bootstrap unit: slide vs patient', '',
         'Written by `scripts/paper/gen_boot_values.py --unit patient`. Point estimates are identical under both '
         'units; only the cells below differ. Columns: slide = the value printed in the paper '
         '(class-stratified bootstrap over the 59 slides); class-stratified slide = the slide-cluster '
         'diagnostic recomputed by `scripts/patient_bootstrap.py` (`slide_stratified`; for regimen and '
         'pairwise tests the original slide bootstrap already is class-stratified, shown as "= slide"); '
         'patient = class-stratified bootstrap over the 46 patients. Spatial CIs are in 10^-2 units.', '',
         f'Flags: borderline endpoint = an accuracy-type CI endpoint at most 1/{N_SLIDES} '
         f'(one slide) from zero, compared with a 1e-9 tolerance.', '',
         '## Summary', '',
         f'- keys: {len(keys)} ({n_avail} with patient results, {len(miss_keys)} missing)',
         f'- stars changed: {n_star}',
         f'- verdicts changed: {n_verd}',
         f'- CI zero-crossing changed: {n_zero}',
         f'- keys with a borderline endpoint: {n_border}', '',
         '## Values', '',
         '| key | slide | class-stratified slide | patient | flags |',
         '|---|---|---|---|---|'] + rows + ['', '## Missing', '']
    L += [f'- `{k}`' for k in miss_keys] or ['(none)']
    write(os.path.join(paper, DIFF_MD), '\n'.join(L) + '\n')
    print(f'wrote {DIFF_MD}: {n_star} star(s) changed, {n_verd} verdict(s) changed, '
          f'{n_zero} zero-crossing change(s), {n_border} borderline, {len(miss_keys)} missing key(s)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--convert', action='store_true')
    ap.add_argument('--unit', choices=('slide', 'patient'))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--paper', default=PAPER)
    a = ap.parse_args(argv)
    if a.convert:
        cmd_convert(a.paper)
    elif a.unit == 'slide' and a.check:
        cmd_check(a.paper)
    elif a.unit == 'patient':
        cmd_patient(a.paper, strict=a.strict)
    else:
        ap.error('use --convert, --unit slide --check, or --unit patient')


if __name__ == '__main__':
    main()
