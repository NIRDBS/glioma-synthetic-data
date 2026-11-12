"""Pad one-decimal numbers to two decimals in the live text of 'Raman_synthetic_data v2.3'
(0.5 -> 0.50, 1.0 -> 1.00, 0.5{:}1 -> 0.50{:}1): body text, captions, notes, table cells.

    python scripts/paper/pad_decimals_v23.py            # dry run: lists every replacement
    python scripts/paper/pad_decimals_v23.py --write

Left unchanged on purpose:
  * comments (lines or trailing parts after an unescaped %);
  * LaTeX layout lengths (\\arraystretch, scale=, \\tabcolsep);
  * mantissas of scientific notation (3.2\\times10^2: a rounded 10^0.5, padding would misstate it);
  * percentages (83.3\\%: a rounded measurement, padding with a zero would be wrong);
  * synthetic-to-real ratios (0.5:1, 0.5{:}1, "synthetic ratio: 0.5"), kept with one decimal.
Numbers that already have two or more decimals are untouched, so the script is idempotent.
Only files in the v2.3 folder are edited; CRLF line endings are preserved.
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V23 = os.path.join(ROOT, 'Raman_synthetic_data v2.3')
NUM = re.compile(r'(?<![\w.])(\d+)\.(\d)(?!\d)')
LAYOUT_BEFORE = ('arraystretch}{', 'scale=', 'tabcolsep}{')


def skip(code, m):
    after, before = code[m.end():m.end() + 14], code[max(0, m.start() - 20):m.start()]
    if re.match(r'\\times\s*10', after):
        return 'scientific-notation mantissa'
    if after.startswith('\\%') or after.startswith('%'):
        return 'percentage'
    if before.endswith(LAYOUT_BEFORE):
        return 'layout length'
    if after.startswith(':1') or after.startswith('{:}1') or before.endswith('ratio: '):
        return 'ratio (kept as 0.5:1)'
    return None


def process(text):
    out, done, skipped = [], [], []
    for ln, line in enumerate(text.split('\n'), 1):
        if line.lstrip().startswith('%'):
            out.append(line)
            continue
        parts = re.split(r'((?<!\\)%)', line, maxsplit=1)
        code, rest = parts[0], ''.join(parts[1:])

        def rep(m):
            why = skip(code, m)
            ctx = code[max(0, m.start() - 30):m.end() + 20].replace('\r', '')
            if why:
                skipped.append((ln, m.group(0), why, ctx))
                return m.group(0)
            done.append((ln, m.group(0), ctx))
            return m.group(0) + '0'
        out.append(NUM.sub(rep, code) + rest)
    return '\n'.join(out), done, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    tot_done = tot_skip = 0
    for path in sorted(glob.glob(os.path.join(V23, '*.tex'))):
        name = os.path.basename(path)
        with open(path, encoding='utf-8', newline='') as f:
            text = f.read()
        new, done, skipped = process(text)
        tot_done += len(done)
        tot_skip += len(skipped)
        if done or skipped:
            print(f'{name}: {len(done)} padded, {len(skipped)} kept')
        for ln, tok, why, ctx in skipped:
            print(f'    kept  l.{ln} {tok} ({why}): {ctx}')
        if args.write and new != text:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(new)
    print(f'total: {tot_done} padded, {tot_skip} kept' + ('' if args.write else '  (dry run; use --write)'))


if __name__ == '__main__':
    sys.exit(main())
