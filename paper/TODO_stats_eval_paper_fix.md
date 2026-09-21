# TODO — remaining paper consistency fixes (audit 2026-06-23)

The earlier Part 1 / Part 2 / Part 3 audit items have been implemented in `main.tex`.
The old stale material (duplicate Statistical Evaluation section, old result tables,
Wilcoxon/Bonferroni/McNemar wording, the 90,000/15,000 synthetic counts, the
`+17.7%` narrative, placeholders) now lives only **below `\end{document}`**
(`main.tex` line 906) and therefore does **not** compile into the PDF.

## Verified DONE
- Edit 1  — Statistical Evaluation rewritten to 95% CI + bootstrap p-value, no
           Wilcoxon/Bonferroni/McNemar, no multiplicity correction (main.tex:772-783).
- Edit 2  — duplicate Statistical Evaluation section is now inert (after `\end{document}`).
- Edit 3  — fingerprint crop 700-1800 cm^-1 stated in Training Details (main.tex:764).
- Edit 4  — per-epoch mixup + spectral-shift augmentation documented (main.tex:766-768).
- Edit 5  — dynamic 500-spectra-per-slide cap documented (main.tex:765).
- Edit 6  — latent dim is `d`; KL sum upper index is `d`; d in {16,32,64}, mostly 32
           (main.tex:485, 512, 522, 525).
- Edit 7  — synthetic count task-specific: 40,000 binary / 120,000 6-class (main.tex:753).
- Edit 8  — CbR row present in active binary results table + appendix (main.tex:823).
- Edit 12 — main-text binary table now matches appendix exp_201 exactly.
- Edit 13 — significance narrative matches current CIs (gain concentrated in 1D-CNN)
           (main.tex:831, 849).

---

## OPEN

### HIGH
- [ ] **Abstract "largest gain of 10.8%" is stale** (main.tex:315). No current 6-class
      1D-CNN TSR/TR gain equals 10.8%. Appendix values: exp_204 **+2.5%**,
      exp_205 **+12.3%**, exp_206 **+6.0%**. The headline binary table is exp_201
      (MSE-only, d=32); its 6-class twin is exp_204 -> 1D-CNN **+2.5%**.
      DECISION NEEDED: feature exp_204 (+2.5%, consistent with headline config) OR
      report the suite maximum +12.3% (exp_205) and label it explicitly. Update the
      number to match whichever is chosen.

### MEDIUM
- [ ] **EXPERIMENT_REGISTRY.json mislabels exp_302/exp_303** (lines 130 & 153 still say
      "Reconstruction Loss Ablation: MSE-only (gamma=1, delta=0) for binary
      classification"). The configs are the latent-dimension sweep (exp_302 d=16,
      exp_303 d=64). The appendix captions were hand-fixed
      (appendix.tex:203, 234), but the generator copies the registry description into
      captions, so re-running `gen_appendix_tables.py` reverts them. Fix the registry
      descriptions (e.g. "Latent-dimension ablation: d=16 / d=64, MSE-only, binary").

- [ ] **appendix.tex hand-edits are not generator output** and will be lost on
      regeneration. Non-generated content: red `\subsection*` group headers, the
      latent-dim captions (above), and the entire **Appendix B - Cross-Validation
      Schema** (appendix.tex:360-449). Decide: fold these into
      `paper/gen_appendix_tables.py`, OR freeze appendix.tex and stop regenerating it.

### LOW
- [ ] State **beta = 2.0** numerically (main.tex:518 only says "scaled by beta>1");
      all configs use `beta: 2.0`.
- [ ] Label the reported metric **"slide-level balanced accuracy"** in the main-text
      tables / Statistical Evaluation section (columns still read "Accuracy"); the
      appendix intro already uses this term.
- [ ] Delete the **dead block main.tex:907-1192** and fix the malformed
      `\end{document}==============` on line 906. Inert (after `\end{document}`), but it
      is the old duplicate stat-eval + stale tables and risks accidental reactivation.
- [ ] Verify **"571 spectral bands"** (main.tex:764) for the 700-1800 cm^-1 crop against
      `data/RamanFreqTable.csv`.
- [ ] Optionally add **airPLS porder = 1** to main.tex:467 (config: `airpls_porder: 1`;
      currently only lambda=100 is stated).
- [ ] Sanity-check **RF+SVM TS/TR precision** in the appendix: 0.758 (exp_201/401/402/403)
      vs 0.258 (exp_203, appendix.tex:81) vs 0.524 (exp_303), while F1 (0.340) and recall
      (0.500) are identical across these tables. Confirm the generator pulls the correct
      precision field.

---

## Verification after edits
```
# all should hit only below \end{document} (main.tex line 906):
grep -n "ilcoxon\|McNemar\|Bonferroni\|binomial test" paper/main.tex
# abstract number should match the chosen 6-class 1D-CNN gain:
grep -n "10.8\|2.5\|12.3" paper/main.tex
# registry should no longer call 302/303 a reconstruction-loss ablation:
grep -n "302\|303" configs/EXPERIMENT_REGISTRY.json
```
