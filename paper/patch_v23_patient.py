"""Write the patient-level bootstrap values into the tables of 'Raman_synthetic_data v2.3'.

Only that folder is touched (no file under paper/ other than this script is read for
editing). The slide-level originals are the zip next to the folder.

    python paper/patch_v23_patient.py            # dry run: checks + summary, writes nothing
    python paper/patch_v23_patient.py --write    # rewrite the v2.3 tables and the change report

What is replaced: every bootstrap-derived table cell -- 95% CIs, the Delta asterisks, the
reported p-values, the yes/no verdicts, the synthetic-ratio star -- keeping the literal's shape
(decimals, signs, separators, star syntax) of the cell it replaces; plus the unit words of the
table captions/notes and of the appendix method paragraphs that say the bootstrap runs over slides.
Point estimates are unchanged. The accuracy table in main.tex (tab:beta_results) is already
patient-level and is only verified against paper/gen_beta_acc_table.py.

Guards (any failure aborts before anything is written):
  * every converted table expands back to its original text byte for byte;
  * the slide-level JSONs, formatted in the shape of each cell, reproduce the v2.3 literal
    (so a mis-mapped cell, or a folder that was already converted, is caught);
  * every key has a patient result.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_boot_values as G  # noqa: E402
import gen_beta_acc_table as B  # noqa: E402

ROOT = os.path.dirname(HERE)
V23 = os.path.join(ROOT, 'Raman_synthetic_data v2.3')
REPORT = 'patient_bootstrap_changes.md'

# v2.3 specifics relative to the v2.2.1 handlers
G.BETA_EXP['0.5'] = '608'          # v2.3 prints beta = 0.5, not 0.50
G.scale_of = lambda key: 1         # v2.3 spatial cells are in raw units, not 10^-2

TABLE_FILES = ['main.tex', 'appendix_controlled_experiments_tables.tex', 'appendix_bootstrap_tables.tex',
               'results_mil_appendix_tables.tex', 'results_beta_table.tex', 'results_synth_ratio_table.tex']
WORDING_ONLY = ['appendix_controlled_experiments.tex']
STALE = {'results_mil_tests_table.tex': 'standalone leftover, not \\input (the table is inline in main.tex)',
         'results_spatial_paired_table.tex': 'standalone leftover, not \\input'}

EXPECTED_LABELS = (['tab:mil_tests_status', 'tab:synth_ratio_results', 'tab:beta_results',
                    'tab:ctrl_boot_latent', 'tab:ctrl_boot_pool']
                   + [f'tab:ctrl_exp_{e}' for e in ('501', '601', '607', '608', '609', '608_thr', '701', '702', '703',
                                                    '704', '705', '706', '4.608', '2.608', '2.607', '2.609')]
                   + [f'tab:ctrl_{k}_{s}' for k in ('spatial', 'keyinst')
                      for s in ('607', '608', '609', '2607', '2608', '4608')])

SYNTH_FOOTNOTE = (r' \\ $^{*}$ The 80k pool differs from the 40k pool at the 95\% level '
                  r'(bag-level bootstrap; Appendix~\ref{app:controlled_experiments}).')
SLIDES_MIL = r'from a bootstrap over slides (10,000 resamples)'
# (file, slide wording, patient wording, expected count, only-if)
WORDING = [
    ('main.tex', r'from a bootstrap over slides (10000 resamples)',
     r'from a class-stratified bootstrap over patients (10000 resamples)', 1, None),
    ('main.tex', SYNTH_FOOTNOTE, '', 1, 'no_pool_star'),
    ('results_synth_ratio_table.tex', SYNTH_FOOTNOTE.replace(r' \\ $^{*}$', r' $^{*}$', 1), '', 1, 'no_pool_star'),
    ('appendix_controlled_experiments.tex',
     r'from a bootstrap over the differences between real and shuffled metrics (10,000 resamples, seed 42)',
     r'from a class-stratified bootstrap over patients of the per-slide differences between real and shuffled '
     r'metrics (10,000 resamples, seed 42)', 1, None),
    ('appendix_controlled_experiments_tables.tex',
     r'class-stratified resampling of the $n=59$ paired bag decisions, 1000 iterations',
     r'class-stratified resampling of the 46 patients, all slides of a drawn patient kept together '
     r'($n=59$ paired bag decisions), 1000 iterations', 1, None),
    ('appendix_bootstrap_tables.tex',
     r'from the class-stratified bootstrap over the $n=59$ bag decisions (1000 iterations, paired)',
     r'from the class-stratified bootstrap over the 46 patients ($n=59$ paired bag decisions; 1000 iterations, '
     r'paired)', 1, None),
    ('results_mil_appendix_tables.tex', r'with its 95\% bootstrap CI over slides',
     r'with its 95\% class-stratified bootstrap CI over patients', 6, None),
    ('results_mil_appendix_tables.tex', SLIDES_MIL,
     r'from a class-stratified bootstrap over patients (10,000 resamples)', 6, None),
    ('results_beta_table.tex', r'$\Delta$ and 95\% CI: class-stratified bootstrap difference in accuracy vs.\ TR/TR '
     r'(1000 iterations)',
     r'$\Delta$ and 95\% CI: difference in accuracy vs.\ TR/TR from a class-stratified bootstrap over patients '
     r'(1000 iterations)', 1, None),
]


def rd(path):
    with open(path, encoding='utf-8', newline='') as f:     # keep CRLF
        return f.read()


def wr(path, text):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def handler(fname, label, E):
    if fname == 'main.tex' and label == 'tab:beta_results':
        return None                                          # accuracy-only table, already patient-level
    if label == 'tab:ctrl_exp_608_thr':                      # duplicate of the exp_608 table
        return G.regimen_handler('608', E)
    return G.handler_for(label, E)


def convert(fname, text, E, seen):
    def rep(m):
        block = m.group(0)
        labels = re.findall(r'\\label\{([^}]*)\}', block)
        h = labels and handler(fname, labels[0], E)
        if not h:
            return block
        seen.append(labels[0])
        return G.process_rows(block, h)
    return G.TABLE_RE.sub(rep, text)


def check_main_accuracy_table(text):
    """Every data row of the generated patient accuracy table must occur in main.tex."""
    gen = B.build(B.load(), 'patient').splitlines()
    rows = [l for l in gen if l.rstrip().endswith('\\\\') and '&' in l and 'emph' not in l]
    flat = text.replace('\r\n', '\n')
    missing = [r for r in rows if r not in flat]
    return len(rows), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    texts = {f: rd(os.path.join(V23, f)) for f in TABLE_FILES + WORDING_ONLY}
    for f, t in texts.items():
        if G.MACRO_RE.search(t):
            sys.exit(f'abort: {f} already contains \\Boot macros')
    n_rows, missing_rows = check_main_accuracy_table(texts['main.tex'])
    print(f'main.tex accuracy table: {n_rows - len(missing_rows)}/{n_rows} generated patient rows found')
    if missing_rows:
        sys.exit('abort: main.tex tab:beta_results differs from paper/gen_beta_acc_table.py --unit patient:\n  '
                 + '\n  '.join(missing_rows))

    E_all, seen, per_file, converted = G.Entries(), [], {}, {}
    for f in TABLE_FILES:
        E = G.Entries()
        converted[f] = convert(f, texts[f], E, seen)
        for (key, field), lit in E.values.items():
            E_all.add(key, field, lit)
        per_file[f] = E.values
        if G.expand_macros(converted[f], E.values) != texts[f]:
            sys.exit(f'abort: round trip failed for {f}')
    absent = [t for t in EXPECTED_LABELS if t not in seen]
    if absent:
        sys.exit(f'abort: tables not found: {absent}')
    print(f'converted {len(seen)} tables, {len(E_all.values)} distinct (key, field) values')

    # slide guard: JSON formatted in the cell's shape == the v2.3 literal
    bad = []
    for (key, field), lit in E_all.values.items():
        rec = G.resolve(key, field, 'slide')
        if rec is None or G.format_value(key, field, lit, rec) != lit:
            bad.append((key, field, lit, None if rec is None else G.format_value(key, field, lit, rec)))
    if bad:
        for b in bad[:40]:
            print('  SLIDE MISMATCH', b)
        sys.exit(f'abort: {len(bad)} v2.3 literal(s) differ from the slide-level JSONs')
    print('slide guard: all v2.3 literals reproduced from the slide-level JSONs')

    patient, missing = {}, []
    for (key, field), lit in E_all.values.items():
        rec = G.resolve(key, field, 'patient')
        if rec is None:
            missing.append((key, field))
            continue
        patient[(key, field)] = G.format_value(key, field, lit, rec)
    if missing:
        sys.exit(f'abort: no patient result for {missing}')

    new = {f: G.expand_macros(converted[f], patient) for f in TABLE_FILES}
    new['appendix_controlled_experiments.tex'] = texts['appendix_controlled_experiments.tex']
    no_pool_star = all(not v for (k, fld), v in patient.items() if k.startswith('pool/') and fld == 'supstar')
    done = []
    for f, old, rep, count, cond in WORDING:
        if cond == 'no_pool_star' and not no_pool_star:
            continue
        n = new[f].count(old)
        if n != count:
            sys.exit(f'abort: {f}: wording found {n}x, expected {count}x: {old[:70]!r}')
        new[f] = new[f].replace(old, rep)
        done.append((f, old, rep, count))

    changed = {(k, fld): (E_all.values[(k, fld)], v) for (k, fld), v in patient.items() if v != E_all.values[(k, fld)]}
    flips = {(k, fld): lv for (k, fld), lv in changed.items() if fld in ('star', 'supstar', 'verdict')}
    print(f'{len(changed)} of {len(E_all.values)} values change; {len(flips)} asterisk/verdict flip(s)')
    for (k, fld), (o, n_) in flips.items():
        print(f'  FLIP {k}/{fld}: {o!r} -> {n_!r}')
    print(f'{len(done)} wording replacement(s)')
    for f, name in STALE.items():
        print(f'skipped {f}: {name}')

    if not args.write:
        print('dry run: nothing written (use --write)')
        return
    for f in TABLE_FILES + WORDING_ONLY:
        if new[f] != texts[f]:
            wr(os.path.join(V23, f), new[f])
            print(f'wrote {f}')
    write_report(per_file, patient, changed, flips, done)
    print(f'wrote {REPORT}')


def write_report(per_file, patient, changed, flips, done):
    L = ['# Patient-level bootstrap in v2.3: changes', '',
         'Written by `paper/patch_v23_patient.py`. The slide-level originals are `Raman_synthetic_data v2.3.zip`. '
         'Point estimates are unchanged; only bootstrap-derived cells (95% CI, asterisk, p-value, verdict) and the '
         'unit wording of captions/notes/method paragraphs were edited. Source of the new values: '
         '`results/patient_bootstrap/**` (class-stratified bootstrap over the 46 patients; 1000 resamples for the '
         'classifier tables, 10000 for the paired MIL tests).', '',
         '## Asterisk / verdict flips', '']
    if flips:
        L += ['| key | field | v2.3 (slide) | patient |', '|---|---|---|---|']
        L += [f'| `{k}` | {fld} | `{o or "(none)"}` | `{n or "(none)"}` |' for (k, fld), (o, n) in flips.items()]
    else:
        L.append('(none)')
    L += ['', '## Wording edits', '']
    for f, old, rep, count in done:
        L += [f'- `{f}` ({count}x): `{old}` -> ' + (f'`{rep}`' if rep else '*(removed)*')]
    L += ['', '## Not edited', '',
          '- `main.tex` tab:beta_results (accuracy-only table): already patient-level; verified against '
          '`paper/gen_beta_acc_table.py --unit patient`.',
          '- Key-instance verdict cells (yes/no/tie) of `main.tex` tab:mil_tests_keyinst: exact comparison of '
          'point accuracies, not a bootstrap test.']
    L += [f'- `{f}`: {why}.' for f, why in STALE.items()]
    L += ['', '## Changed values by file', '']
    for f, vals in per_file.items():
        rows = [(k, fld, lit, patient[(k, fld)]) for (k, fld), lit in vals.items() if patient[(k, fld)] != lit]
        L += [f'### {f}', '', f'{len(rows)} of {len(vals)} values changed', '']
        if rows:
            L += ['| key | field | slide (v2.3) | patient |', '|---|---|---|---|']
            L += [f'| `{k}` | {fld} | `{o}` | `{n}` |' for k, fld, o, n in rows]
            L.append('')
    wr(os.path.join(V23, REPORT), '\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
