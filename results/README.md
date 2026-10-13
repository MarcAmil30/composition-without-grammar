# Results

Source tables and cached statistics behind the reported numbers. Rendered
figures live in `figures/`; `docs/PROVENANCE.md` maps each one back to the file
here that it reads.

```
upstream_only/            promoter-grammar track (§3, Table 1, Appendix G)
  fig2x2/                 the Figure 2 panels and their per-panel CSV sources
    panelA_tol2_bg_v2.csv     placement within +-2 bp, per model x background
    panelA_spacer.csv         distal-to-proximal spacer distribution
    panelA_summary.csv        fraction in 15-19 bp and at exactly 17 bp
    panelB_real.csv           real sigma70 per-base profile, 41 bp window
    panelB_toatg_v2.csv       generated per-base profiles, same window
    panelB_probs.csv          earlier 4-model version, used by Fig. D.1
    panelC_hamming_bg_v2.csv  bases matched at the true -10
    panelC_presence.csv       fraction with any hexamer above the PWM threshold
    panelD_context_tol2_v2.csv  placement vs prompt context length
    table1_source_v2.csv      Table 1 verbatim
    make_figure_v2.py         draws Figure 2 from the five *_v2 CSVs
  benchmark/              per-model summaries and the earlier standalone panels
  audit/                  start-codon, TSS and SD position audits of the prompts
  sd_denovo/, sd_spacing/ de novo Shine-Dalgarno PFM and spacing distributions
  pc_*.csv                position-control ablations (spacer length, GC, TSS)

cds_prompted/             CDS-conditioned generation track (§4, Fig. 3, Tables A.1, B.1)
  cds_fig_stats_testset.npz  every array Figure 3 draws
  render_fig_testset.py      draws Figure 3 from it
  fig_stats.py               rebuilds the cache from the generations (cluster)
  kl_table.md                Table A.1
  COARSE_SUMMARY.md          Table B.1 and the composition argument behind it
  panelAB_v*_reseed.npz      held-out reseeded panel A/B inputs

fig2d_long_context/       long-context placement ladder (Fig. 2d)
  summaries/*.json          per model and context length: within +-2 bp and its
                            shuffled control
  STATUS.md                 run log for the generation and scoring arrays

loss_decomposition/       final-checkpoint cross-entropy by record region (Fig. F.1)
fig1a_dataset/            orthologue-pair summary statistics (Fig. E.1)
cds_completion_profiles.npz, cds_completion_metrics.csv, cds_completion_summary.md
                          CDS midpoint completion (Figs. C.1, D.1)
cds_vs_upstream_gap.csv   position-blind reference and per-model ratios (Fig. D.1)
cds_prompted_dataset_supp_stats.npz  composition of the CDS-prompted benchmark
cds_upstream_retrieval/   earlier length and window sweeps of the retrieval task
dataset_analysis/         notebook characterising the E. coli annotation set
```

## Notes

* `cds_fig_stats_testset.npz` is the cache Figure 3 uses. Panels A and B come
  from a held-out subset (673 *E. coli*, 441 *B. subtilis*) regenerated in a
  later reseeded run; the C2 and C3 profiles in panel C come from the original
  run. The earlier full-set cache is in `_archive/` — its `C3` arrays equal `C0`
  exactly, because the substitution permutes coding sequences within a closed
  gene set, so the population-averaged profile cannot move. The paired per-gene
  KL in `kl_table.md` is the measurement that tests CDS sensitivity.
* `panelB_probs.csv` predates the two-checkpoint split of Evo 1 and carries four
  models; `panelB_toatg_v2.csv` carries five and is what Figure 2b draws.
  Figure D.1 uses the older file because only three models were run on both the
  CDS and the upstream window.
* Figure PDFs and PNGs that used to sit in this tree were moved to `_archive/`;
  the published copies are in `figures/`.
