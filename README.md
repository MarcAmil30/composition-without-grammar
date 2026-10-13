# Composition Without Grammar: Next-Token Models Miss Regulatory Structure in Bacterial DNA

Manuscript, figures, evaluation code and source tables for the ICBINB-BIO @
NeurIPS 2026 workshop submission. The repository is anonymous: it carries no
author names, no institutional identifiers, and no cluster paths or usernames.

## What the paper claims

Regulatory-sequence design is conditional — an upstream region has to work in
its host *and* suit the gene it regulates. The paper asks whether autoregressive
genomic language models capture either half of that, and separates two
capabilities that are usually conflated:

1. **Local promoter grammar** — motif identity and position (does the model put
   the proximal box at the right spacing from the distal box?).
2. **Gene conditioning** — does the model use the *particular* coding sequence,
   or does it write a generic species-typical upstream?

Across five released checkpoints, promoter placement stays at the shuffled
control. A conditionally fine-tuned Evo writes species-typical sequence and a
Shine–Dalgarno signal, but shuffling, replacing or removing the CDS moves its
output by at most 0.014 bits per position. A contrastive C3P checkpoint retrieves
the matched upstream for 82.7% of held-out *E. coli* genes against 6.3% for the
best autoregressive model, so the pairwise signal is learnable under a different
objective. The supported failure mode is composition without local grammar or
gene-specific dependence — not a claim that next-token loss is the sole cause.

## Layout

| path | contents |
|---|---|
| `main.tex`, `supplementary.tex` | the paper; `supplementary.tex` is `\input` by `main.tex` so appendices A–H compile into one PDF |
| `neurips_2026.sty` | required workshop style, double-blind mode |
| `Literature_Sources/references.bib` | bibliography |
| `figures/` | the PDFs the manuscript includes, plus the two panel sources that `Fig1_setup.pdf` is composed from |
| `code/` | evaluation, benchmark-construction and figure code, grouped by experiment |
| `data/` | prompt manifests, gene lists and per-model rank tables |
| `results/` | the source tables and cached statistics behind every reported number |
| `docs/` | provenance map and long-form method notes |

`docs/PROVENANCE.md` maps every figure and table in the paper to the script that
draws it and the file it reads. Start there.

## Building the paper

```bash
tectonic main.tex
```

The submission artifact is `main.pdf`. Do not upload a separate appendix PDF —
the workshop wants the appendix inside the main file, which is what
`\input{supplementary}` produces.

## Running the code

```bash
pip install -r requirements.txt
```

Scripts read the repository they live in and take an output path on the command
line where they write one. Anything that needs material outside the repository —
model checkpoints, per-shard generation tensors, the Gammaproteobacteria
orthologue tables — reads it from an environment variable with a documented
default:

| variable | what it points at |
|---|---|
| `ICBINB_ROOT` | this repository (or the cluster copy of it) |
| `CL_ROOT` | the `Contrastive_Learning_NeurIPS2026` workspace holding the orthologue-pair parquet behind Fig. 1a |
| `FINETUNED_EVO` | the CDS-conditioned Evo fine-tune checkpoint |
| `C3P_REPO`, `C3P_CHECKPOINT` | the public C3P checkout and its `C3P_100M.pth` |
| `HF_CACHE` | Hugging Face cache for the released checkpoints |
| `SLURM_ACCOUNT` | account for the `.sbatch` / `.slurm` submission scripts |

### Two kinds of script

`code/` holds both halves of the pipeline, and they have very different
requirements. Read this table before running anything.

| | runs from this repository | needs the cluster |
|---|---|---|
| what | figure renderers, reading the aggregates in `data/` and `results/` | generation, likelihood scoring, retrieval, dataset construction |
| needs | Python + `requirements.txt` | GPUs, model checkpoints, the raw generations |
| where | `make_*.py`, `render_*.py`, `plot_*.py`, `compose_*.py` | `*.sbatch`, `*_retrieval.py`, `fig_stats.py`, `code/shared/` |
| reads | paths inside this repository | `data/benchmark/` — **not shipped**, see *What is not in the repository* |

Around thirty files reference `data/benchmark/`. That tree is the full benchmark
(raw generations, likelihood matrices, per-shard probability tensors) and runs to
tens of gigabytes, so it stays on the cluster. Those scripts are released so the
analysis can be audited and rerun, not because they will execute here.

### Regenerating the paper's figures

Every command below was run from a clean checkout of this repository and
completed with no arguments and no cluster access:

```bash
python code/figure_1_task_and_association/make_fig1_panel_a.py      # Fig. 1a
python code/figure_1_task_and_association/make_fig1_tasks.py        # Fig. 1b-d strip
python code/figure_1_task_and_association/compose_fig1.py           # Fig. 1 (2x2)
python code/figure_2_promoter_element_placement/make_figure_v2.py   # Fig. 2
python code/figure_2_promoter_element_placement/make_sd_matches_appendix.py
python code/figure_3_cds_conditioning/render_fig_testset.py         # Fig. 3
python code/figure_3_cds_conditioning/make_kl_perposition_figure.py # per-position KL
python code/figure_4_retrieval/make_fig.py                          # Fig. 4
python code/figure_C2_cds_vs_upstream_gap/make_cds_vs_upstream_gap.py
python code/figure_E1_benchmark_composition/make_upstream_only_dataset_figure.py
python code/figure_E2_promoter_logos/upstream_only_figure.py
python code/figure_E3_spacer_histograms/make_supp_spacer_histograms.py
```

`code/figure_F1_training_corpus/noncoding_analyse_and_plot.py` also runs here but
additionally needs `pyrodigal` (in `requirements.txt`) and the reference genomes
it downloads on first use.

Run `compose_fig1.py` **after** `make_fig1_panel_a.py` and `make_fig1_tasks.py`,
since it composes their output. It finishes by flattening the result with
`pdftocairo` (poppler); without poppler the figure still renders correctly but
carries three clipped copies of the source strip, which vector editors display as
stacked duplicate layers.

### The one notebook that does not run here

`code/figure_3_cds_conditioning/make_cds_prompted_main_figure.ipynb` is the
full-population renderer and needs `cds_fig_stats.npz`, which is built on the
cluster from the raw generations. The released cache,
`results/cds_prompted/cds_fig_stats_testset.npz`, carries conditions C0, C2 and
C3 only. Use `render_fig_testset.py` above instead — it reproduces the published
Fig. 3 exactly. The notebook says the same and stops with an explanation rather
than a bare `FileNotFoundError`.

The two are deliberate variants of one figure, not an accident: the notebook is
the cluster version, drawing all seven CDS conditions from the full-population
cache, and `render_fig_testset.py` is the released version, drawing the three
conditions the shipped test-set cache carries. They share most of their plotting
code, so a change to one should be mirrored in the other. The notebook is kept in
the repository because it reads as a document on GitHub — the panels, their
claims and the aggregation rule are written out cell by cell — which the script
does not.

## What is not in the repository

* Model checkpoints. Evo 1, Evo 2, HyenaDNA, PromoGen2 and C3P are public; the
  CDS-conditioned Evo fine-tune is ours and is not redistributed here.
* Raw generations. The generation, likelihood and per-shard probability tensors
  are tens of gigabytes; `results/` ships the aggregates the paper reports, and
  `code/` ships the scripts that produce them.
* The orthologue-pair table behind Fig. 1a. `results/fig1a_dataset/` ships the
  summary statistics; the 22,394-row parquet is built by
  `code/figure_1_task_and_association/grammar_singleton_pairs/` from precomputed MMseqs2 gene families.

## Known caveats in the released analysis

Recorded here so a reader does not have to rediscover them.

* **Retrieval normalisation.** Reported autoregressive accuracies use the
  candidate-centred (PMI-like) score. Without centring the same likelihoods give
  7.5% for Evo 2 and 7.1% for Evo 1; C3P is reported uncentred because cosine in
  its trained space is its native score. The centred variant is reported because
  it is the one normalisation applied identically to every autoregressive model,
  and the conclusion does not turn on the choice.
* **The C3 track of Fig. 3c.** The "different gene's CDS" condition substitutes a
  partner drawn from the same gene set, so the *population-averaged* profile is a
  relabelling of the native one and is invariant by construction. The paired
  per-gene KL in Table A.1 is the test that actually measures CDS sensitivity;
  the C2 (shuffled-CDS) profile is the informative unpaired comparison.
* **Fig. 3 panel provenance.** Panels A and B use a held-out subset (673
  *E. coli*, 441 *B. subtilis*) regenerated in a later reseeded run; the C2 and
  C3 profiles in panel C come from the original run. The cached array is
  `results/cds_prompted/cds_fig_stats_testset.npz`.
* **Not homology-disjoint.** The fine-tuning benchmarks were not filtered against
  the training genes or their homologues, which favours the generator. C3P's
  published split holds out *E. coli* by species but is not reported as
  homology-disjoint.

## Licence

Code and derived tables are released under the MIT licence (`LICENSE`). The
underlying sequences and annotations come from public reference genomes
(*E. coli* K-12 MG1655, *B. subtilis* 168, Gammaproteobacteria assemblies) and
RegulonDB, and remain under their own terms.
