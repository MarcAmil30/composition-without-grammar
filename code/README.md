# Code

One directory per figure in the paper, named for what the figure shows. Code
used by more than one figure lives in `shared/`. Scripts that draw figures no
longer cited by the current draft are kept in `not_in_current_draft/`.

Entry points read the repository they live in; anything outside it comes from an
environment variable (see the table in the top-level README). `.sbatch` /
`.slurm` scripts take the Slurm account from `$SLURM_ACCOUNT`.

`notebooks/reproduce_figures.ipynb` runs the whole set in paper order.

```
shared/                                used by more than one figure
  dataset/                               benchmark construction
    build_dataset.py                       E. coli/B. subtilis singleton sets + upstreams
    build_prompts_wide.py                  the 681-gene upstream-only prompts
    build_retrieval_set.py                 the retrieval candidate pool
    build_eval500.py                       the 500 query genes and their decoys
    analyze_grammar.py                     RegulonDB element positions and spacings
    dataset_analysis.py                    composition of the annotation set
  generation/                            free-running decoding
    models.py                              Evo 1/2, HyenaDNA, PromoGen2 loading
    framing.py, generate.py                per-model prompt framing and sampling
    score.py, build_pwms.py                PWM scan and placement scoring
    figures.py                             per-model diagnostic panels

figure_1_task_and_association/         Fig. 1
  compose_fig1.py                        assembles the 2x2 setup figure
  make_fig1_panel_a.py                   panel a
  make_fig1_tasks.py                     panels b-d schematic
  fig1a_dataset_summary_remote.py        cluster-side summary for panel a
  grammar_singleton_pairs/               the orthologue-pair analysis behind panel a

figure_2_promoter_element_placement/   Fig. 2
  consensus.py, validate_seq.py          PWM construction and prompt validation
  overlay_regulondb.py                   RegulonDB elements onto each 200 bp window
  sd_denovo.py, sd_spacing_hist.py, sd_atg_tss_audit.py   SD and TSS diagnostics
  update_fig2_paneld.py                  refreshes panel d
  fig2d_long_context/                    placement vs prompt length (panel d)

figure_4_retrieval/                    Fig. 4
  {c3p,evo2,lm,prok}_retrieval.py        one scorer per model family
  retrieval_eval.py                      the shared scoring path
  make_fig.py, collect.py                figure and aggregation

figure_C1_cds_completion/              Fig. C.1
figure_C2_cds_vs_upstream_gap/         Fig. C.2
figure_E2_promoter_logos/              Fig. E.2
figure_E3_spacer_histograms/           Fig. E.3
figure_F1_training_corpus/             Fig. F.1  (vendored pyrodigal/archspec
                                       under vendor/, aarch64 build for Isambard)

not_in_current_draft/                  drawn, but not cited by main.tex
  make_supp_fig1a_dataset.py, make_supp_promoter_geometry.py,
  make_supp_element_composition.py, make_fig_multiscale_loss.py,
  loss_decomposition.py + run_loss_decomposition.sbatch,
  cds_upstream_retrieval/, and the two notebooks that go with them
```

Fig. 2, Fig. 3 and Fig. E.1 are drawn by scripts that sit next to their data
under `results/` (`code/figure_2_promoter_element_placement/make_figure_v2.py`,
`code/figure_3_cds_conditioning/render_fig_testset.py`,
`code/figure_E1_benchmark_composition/make_upstream_only_dataset_figure.py`). They read CSV/npz files from
their own directory, so they stay there rather than moving into `code/`.
