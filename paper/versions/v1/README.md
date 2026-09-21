# Paper v1 — CMPB submission (rejected)

Archived 2026-09-21, immediately before the weakly-supervised MIL reframing began.
Git tag: **`paper-v1`**.

## Contents

| path | what it is |
|---|---|
| `submitted/CMPB-D-26-05140.pdf` | The manuscript as submitted to *Computer Methods and Programs in Biomedicine*, 2026-07-14. The authoritative record of what was reviewed and rejected. |
| `source/main.tex`, `appendix.tex`, `references.bib` | LaTeX source at the freeze point (2026-09-21). **See the provenance caveat below — this is not the submitted source.** |
| `source/main.pdf` | Build of the above, 2026-08-26. |
| `source/main.bbl` | Resolved bibliography for that build. |
| `source/gen_appendix_tables.py` | Generates the appendix result tables from `statistical_evaluation_*.json`. |
| `source/figures/`, `source/img/` | All figures. These are **not** tracked in the main tree — `.gitignore` excludes `*.png` and `*.pdf` repo-wide — so this archive is the only version-controlled copy. |

## Provenance caveat — read before relying on `source/`

**The exact source that produced `CMPB-D-26-05140.pdf` was never committed and is not recoverable.**

There are no commits between `f53ea8e` (2026-06-18) and `517db47` (2026-09-21). The submission falls
inside that gap:

```
f53ea8e  2026-06-18  last commit before submission
         2026-07-14  CMPB-D-26-05140.pdf submitted   <-- source never committed
         2026-08-24  main.tex last modified
         2026-08-26  main.pdf built
517db47  2026-09-21  freeze commit (captures the working tree as it stood today)
```

So `source/` is the **closest available state**, drifted roughly six weeks past submission — not a
reconstruction of it. Where the two disagree, `submitted/CMPB-D-26-05140.pdf` is authoritative.

The same gap applies to the code: the pipeline state that generated the submitted numbers was also
never committed. `paper-v1` tags `517db47`, the nearest committed approximation, not the exact
submission-time tree.

## Known issues in v1 that the reframing addresses

Recorded here so the archive is self-explanatory, and because several affect numbers in the
submitted PDF. Full argument in [`docs/MIL_REFRAME_PLAN.md`](../../../docs/MIL_REFRAME_PLAN.md).

- **The core assumption is false.** Classifiers are trained on `(spectrum, slide_label)` pairs,
  asserting that every spectrum carries its slide's tumour grade. The mask that produced the
  modelling data is a *biological-viability* mask, not a tumour mask — `y_tumor` is a misnomer and
  is constant `1` across all 227,344 rows. There is no instance-level label in this dataset, only
  bag labels.
- **Reported metrics conflate two units of analysis.** `accuracy`, `balanced_accuracy` and
  `recall_macro` are the same float — mean within-bag instance accuracy — while the confusion matrix
  next to them counts slides under majority vote.
- **Three metrics are degenerate**: `precision_macro` is a constant ≈0.5, `kappa` is always 0, and
  `f1_macro` silently drops slides with 0% within-bag accuracy (hence `f1_macro = 0.926` beside
  `accuracy = 0.431` in `results/exp_401`).
- **The 1D-CNN early-stops on the test fold**, selecting weights on held-out data.
- **Baseline-correction λ was inherited by eye** (airPLS λ=100, from Lita et al.), and the repo's two
  preprocessing paths disagree on it by a factor of 100.
