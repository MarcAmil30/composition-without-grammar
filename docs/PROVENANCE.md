# Provenance

Every figure, table and quoted number in the paper, with the script that
produces it and the file it reads. Paths are relative to the repository root.
"cluster" means the input is a raw generation or likelihood dump that is too
large to ship; the aggregate it produces is in `results/`.

## Figures

| paper | file | drawn by | reads |
|---|---|---|---|
| Fig. 1 | `figures/Fig1_setup.pdf` | `code/figure_1_task_and_association/compose_fig1.py` | `figures/fig1_panel_a.pdf` + `figures/ICBINB_BIO_NEURIPS2026_FIGS_1.pdf` |
| Fig. 1a panel | `figures/fig1_panel_a.pdf` | `code/figure_1_task_and_association/make_fig1_panel_a.py` | `figures/fig_grammar_singleton_pairs.pdf`, `figures/fig_grammar_singleton_pairs_stats.csv` |
| Fig. 1a source | `figures/fig_grammar_singleton_pairs.pdf` | `code/figure_1_task_and_association/grammar_singleton_pairs/make_fig_grammar_singleton_pairs.py` | `$CL_ROOT/results/upstream_grammar/pairs_cds_singleton_pairs.parquet` (cluster) |
| Fig. 1b–d schematic | `figures/ICBINB_BIO_NEURIPS2026_FIGS_1.pdf` | `code/figure_1_task_and_association/make_fig1_tasks.py` | drawn from scratch |
| Fig. 2 | `figures/fig_upstream_only_v2.pdf` | `code/figure_2_promoter_element_placement/make_figure_v2.py` | `panelA_tol2_bg_v2.csv`, `panelB_real.csv`, `panelB_toatg_v2.csv`, `panelC_hamming_bg_v2.csv`, `panelD_context_tol2_v2.csv` in the same directory |
| Fig. 3 | `figures/cds_prompted_main.pdf` | `code/figure_3_cds_conditioning/render_fig_testset.py` | `results/cds_prompted/cds_fig_stats_testset.npz` |
| Fig. 4 | `figures/fig_retrieval.pdf` | `code/figure_4_retrieval/make_fig.py` | `data/retrieval/results/*/{*_ranks.parquet,*_summary.json}` |
| Fig. C.1 | `figures/supp_cds_completion_probabilities.pdf` | `code/figure_C1_cds_completion/plot_profiles.py` | `results/cds_completion_profiles.npz` |
| Fig. C.2 | `figures/fig_cds_vs_upstream_gap.pdf` | `code/figure_C2_cds_vs_upstream_gap/make_cds_vs_upstream_gap.py` | `results/cds_completion_profiles.npz`, `results/upstream_only/fig2x2/panelB_{real,probs}.csv` |
| Fig. E.1 | `figures/upstream_only_dataset_supp.pdf` | `code/figure_E1_benchmark_composition/make_upstream_only_dataset_figure.py` | `data/prompts_681_{leaked,clean}.csv`, `results/upstream_only/sd_denovo/` |
| Fig. E.2 | `figures/upstream_only_logos.pdf` | `code/figure_E2_promoter_logos/upstream_only_figure.py` | `data/prompts_681_leaked.csv` |
| Fig. E.3 | `figures/supp_spacer_histograms.pdf` | `code/figure_E3_spacer_histograms/make_supp_spacer_histograms.py` | `results/upstream_only/fig2x2/panelA_spacer.csv` |
| Fig. F.1 | `figures/supp_training_composition.pdf` | `code/figure_F1_training_corpus/noncoding_analyse_and_plot.py` | `results/training_corpus_composition/noncoding_nucleotide_composition.{csv,json}` |

Upstream inputs one level further back:

* `results/loss_decomposition/loss_decomposition.csv` ← `code/not_in_current_draft/loss_decomposition.py` (one teacher-forced pass of the fine-tuned checkpoint over 200 held-out records).
* `results/cds_completion_profiles.npz` ← `code/figure_C1_cds_completion/export_profiles.py` from the per-shard probability tensors on the cluster; those come from `build_prompts.py` + `run_shard.py`.
* `results/cds_prompted/cds_fig_stats_testset.npz` ← `code/figure_3_cds_conditioning/fig_stats.py` over `data/cds_prompted/generations/` (cluster).
* `results/upstream_only/fig2x2/panel*_v2.csv` ← `panels_ac_by_background_v2.py` and `hamming_at_true_v2.py` in the same directory, over the upstream-only generations (cluster).
* `data/retrieval/results/` ← `code/figure_4_retrieval/{c3p,evo2,lm,prok}_retrieval.py`, all scored through the shared `code/figure_4_retrieval/retrieval_eval.py`.
* `results/training_corpus_composition/noncoding_nucleotide_composition.{csv,json}` ← `code/figure_F1_training_corpus/noncoding_analyse_and_plot.py`, counting bases outside predicted CDS intervals (Pyrodigal for the microbial corpora, GENCODE v49 for hg38) over each released training corpus (cluster).
* `results/fig2d_long_context/summaries/` ← `code/figure_2_promoter_element_placement/fig2d_long_context/collect_score.py`; Fig. 2d reads `panelD_context_tol2_v2.csv`, which is assembled from those summaries.

## Tables

| paper | source |
|---|---|
| Table 1, promoter and SD placement | `results/upstream_only/fig2x2/table1_source_v2.csv` |
| Table A.1, CDS-perturbation KL | `results/cds_prompted/kl_table.md`, from `data/cds_prompted/results/kl_*_perposition.npy` |
| Table B.1, 6-mer shuffle ladder | `results/cds_prompted/COARSE_SUMMARY.md` |
| Table H.1, perturbation conditions | `data/cds_prompted/prompts_manifest.json` |

## Numbers quoted in the text

| claim | value | source |
|---|---|---|
| orthologue vs control Spearman ρ | +0.437 / +0.027 | `figures/fig_grammar_singleton_pairs_stats.csv` |
| pairs, genomes, anchors | 22,393 (finite) of 22,394 / 408 / 579 | `results/fig1a_dataset/fig1a_dataset_summary.json` |
| within-rank ρ (family / order / class) | 0.55 / 0.21 / 0.05 | same file, `within_rank` |
| ρ across k = 3,4,5 and both metrics | 0.30–0.45 | same file, `rho_grid` |
| real-promoter placement | 67.6% | `panelA_tol2_bg_v2.csv` |
| leaked→clean placement change | ≤1.6 pp, 0.3 pp for Evo 1 (8k) | same file |
| bases matched at the true −10 | 1.6–2.1 of 6 vs 3.35 for a different real box | `panelC_hamming_bg_v2.csv` |
| entropy / P(G) / P(A)+P(T) over the 41 bp window | see §3 | `panelB_real.csv`, `panelB_toatg_v2.csv` |
| Evo 2 at 8 kb over chance | 16.10% / 12.5% = 1.29× | `panelD_context_tol2_v2.csv` |
| Evo 1 (131k) at 131 kb vs shuffle | 11.76% vs 11.38% | `results/fig2d_long_context/summaries/evo1_131k_ctx131000_forced.json` |
| HyenaDNA at 449 kb vs shuffle | 12.39% vs 11.79% | `.../hyenadna_ctx449000.json` |
| 6-mer r, correct / wrong species token | 0.896, 0.938 / 0.774, 0.770 | `cds_fig_stats_testset.npz`, `A_*_{correct,wrong}_r` |
| GC spread, natural vs generated | 6.6%, 5.4% vs 2.3%, 2.4% (between-gene) | same file, `GC_*_nat`, `GC_*_vargene` |
| retrieval top-1 | 82.7 / 6.3 / 6.0 / 2.8 / 1.0% | `data/retrieval/results/*/score_cond_pmi_summary.json`, `c3p/embed_cos_summary.json` |
| retrieval, uncentred variant | 7.5 / 7.1 / 1.2 / 1.0% | `data/retrieval/results.csv` and the cluster's `score_cond_raw_summary.json` |
| loss shares (CDS / 5′UTR body) | 58.6%, 57.1% / 19.1%, 19.9% | `results/loss_decomposition/loss_decomposition.csv` |
| position-blind reference and ratios | 0.094, 0.067; 0.21, 0.88, 1.05, 1.16 | `results/cds_vs_upstream_gap.csv` |
| spacer in 15–19 bp / exactly 17 bp | 60.5% / 21.0% real, 10.7–13.5% / 2.0–3.0% models | `results/upstream_only/fig2x2/panelA_summary.csv` |
| any hexamer above PWM threshold | 93–97% generated, 95–97% shuffled | `panelC_presence.csv` |
| CDS-prompted set sizes and clean windows | 1,885 / 1,272; 781 (41%) / 406 (32%) | `data/cds_prompted/prompts_manifest.json` |

## Deliberately excluded from `figures/`

`_archive/figures/` holds superseded renders (`fig_upstream_only.pdf`,
`fig_main_2x2.pdf`, the `*_prelinear` backups), PNG previews of every published
panel, and exploratory figures that no longer appear in the paper
(`fig_multiscale_loss`, `fig_sigma70_validation`, `fig_upstream_specificity`,
`fig_cds_upstream_retrieval`, `cds_prompted_supp_cai`). `_archive/` is
`.gitignore`d and is not part of the release.

Three figures ship in `figures/` but are no longer cited by the current draft:
`supp_fig1a_dataset.pdf` (`code/not_in_current_draft/make_supp_fig1a_dataset.py`),
`fig_loss_decomposition.pdf` (`code/not_in_current_draft/make_loss_decomposition_figure.py`) and
`supp_promoter_geometry.pdf` (`code/not_in_current_draft/make_supp_promoter_geometry.py`).
`supp_element_composition.pdf` (`code/not_in_current_draft/make_supp_element_composition.py`,
reads `data/prompts_681_leaked.csv`) was drawn but not included either. Their
scripts and inputs are kept so the panels can be revived without rerunning the
benchmark.

`_archive/results/cds_prompted/cds_fig_stats.npz` is the earlier full-set cache.
Its `C3` arrays are numerically identical to `C0` because the substitution
permutes CDSs within a closed gene set; the shipped `cds_fig_stats_testset.npz`
is what Fig. 3 uses. See the caveats section of the top-level README.
