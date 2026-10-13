# CDS -> upstream retrieval figures

## `fig_upstream_specificity.{pdf,png}`  (main figure)

Four panels, *E. coli* only. One argument: which model, how much sequence, which part.

| panel | question | x-axis | n |
|---|---|---|---|
| A | which model picks the right upstream for a CDS? | C3P embedding · Evo-1 scoring · Evo-1 embedding (300 bp) | 150 genes shared by all three runs |
| B | how much upstream is needed? | 200 / 100 / 50 bp, truncated toward the start codon | 727 E. coli genes |
| C | schematic of the windows in D | — | — |
| D | which part of the upstream? | window width 50/100/150/200 bp, proximal vs distal at matched width | 409 genes with a >= 200 bp flank |

y-axis on A, B, D = **rank of the true upstream among 100 candidates**, averaged over the 20
retrieval repeats (1 = correct, chance median = 50.5), log scale.

### Headline numbers (top-1 / median rank)

| condition | top-1 | median rank |
|---|---|---|
| C3P embedding, 300 bp | 88 % | 1.0 |
| Evo-1 scoring (log P(CDS \| upstream)), 300 bp | 16 % | 13.9 |
| Evo-1 mean-pooled embedding, 300 bp | 1 % | 42.2 |
| C3P 200 / 100 / 50 bp (truncated) | 85 / 61 / 11 % | 1.0 / 1.1 / 23.1 |
| `prox50` (-50..0) | 8 % | 26.5 |
| `prox100` (-100..0) / `dist100` (-200..-100) | 50 % / 36 % | 1.7 / 5.3 |
| `prox150` (-150..0) / `dist150` (-200..-50) | 74 % / 65 % | 1.0 / 1.0 |
| `full200` (-200..0) | 81 % | 1.0 |

All six bracketed comparisons are one-sided paired Wilcoxon signed-rank, p < 1e-4
(`fig_upstream_specificity_stats.csv`); per-condition medians in
`fig_upstream_specificity_summary.csv`.

### Reading

The signal is **not** simply distal. Held at matched width the proximal window still wins
(100 bp: 50 % vs 36 %; 150 bp: 74 % vs 65 %) — but the 50 bp abutting the start codon are
near-useless alone (8 %, median rank 26), and deleting them costs only 81 -> 65 %. Marginal
gains along the nested proximal series (8 % -> 50 % -> 74 % -> 81 %) put the heavy lifting in
**-100..-50**, just behind the median TSS (5'UTR = 45 bp, from the RegulonDB overlay) and
matching the -62 peak of the independent single-6 bp occlusion run.

**Caveat for the caption:** C3P was trained on windows anchored at the start codon, so a
-200..-100 window reaches the encoder in the positional frame of a -100..0 window. Every
window in D shares that frame, so the contrasts are internally controlled — but none is the
model reading the distal region in situ.

## `fig_cds_upstream_retrieval.{pdf,png}`  (earlier 2-panel version)

Panels A and B of the above, before the window sweep existed. Superseded; kept because its
notebook is the simpler worked example.

## How to rebuild

Primary copy of the plotting code is the notebook; the script is a headless mirror.

```bash
jupyter nbconvert --to notebook --execute --inplace code/upstream_specificity_figure.ipynb
# or, with no jupyter around:
python code/make_upstream_specificity_figure.py
```

Both exist locally and on Isambard at `${ICBINB_ROOT}/`.
On Isambard use the `evo2_env` conda env (`c3p_model` has no jupyter); it renders with
DejaVu Sans instead of Arial, so the **local** build is the one that matches the other paper
figures — that is the copy mirrored into `figures/`.

## Provenance

Ranks were computed in `Contrastive_Learning_NeurIPS2026` and **copied** here, unchanged, into
`data/cds_upstream_retrieval/`; the runs that produced them are in
`code/cds_upstream_retrieval/` (`exp3_length_sweep.py`, `evo_8k_score_length_sweep.py`,
`evo_8k_length_sweep.py` and their sbatch scripts). Nothing was moved out of the contrastive
learning project.

| file | run |
|---|---|
| `c3p_length_ranks.parquet` | C3P protein↔upstream cosine retrieval, length sweep |
| `evo8k_score_ranks.parquet` | Evo-1-8k likelihood scoring (`metric == "cds"` is the readout used) |
| `evo8k_embed_ranks.parquet` | Evo-1-8k mean-pooled embedding cosine (`..._centered` = anisotropy-corrected variant, also at chance) |
| `c3p_vs_evo_stats.csv` | the original C3P-vs-Evo paired tests, both species, all four lengths |
| `c3p_window_ranks.parquet` | C3P matched-width **window** sweep (exp 4), job 5900937, both designs and both species |
| `c3p_window_summary.json` | per design x species x window: top1, top5, median, mean, n |

Exp 4 was run *for this paper* (`code/cds_upstream_retrieval/exp4_window_sweep.py` +
`run_exp4_window.slurm`, ~70 s on one GH200); everything else predates it and was copied in.
The `.run1` files on Isambard are an earlier, smaller window set, superseded by job 5900937
which reproduced them exactly.
