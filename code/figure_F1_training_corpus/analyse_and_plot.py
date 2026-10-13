#!/usr/bin/env python3
"""CPU-only nucleotide-composition analysis for three model training corpora.

The two multi-terabyte corpora are sampled reproducibly from their official
Hugging Face repositories without downloading complete shards.  HyenaDNA's
hg38 corpus is streamed and counted in full.  Ambiguous symbols are excluded.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import fsspec
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi


SEED = 20260820
BASES = "ACGT"
BASE_COLORS = {
    "A": "#2f8f4e",
    "C": "#2f6fb0",
    "G": "#e0a33e",
    "T": "#c94f4f",
}
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# Phase-one weights from the official OpenGenome2 dataset card.  This is the
# appropriate mixture for the short-context evo2_1b_base checkpoint evaluated
# in the manuscript. Values are normalized below because published percentages
# are rounded and sum to 100.52%.
OG2_WEIGHTS = {
    "gtdb_v220_imgpr": 0.18,
    "metagenomes": 0.24,
    "imgvr_untagged": 0.03,
    "mrna_splice_promoter": 0.09,
    "mrna": 0.09,
    "eukaryotic_genic_windows": 0.35,
    "organelle": 0.005,
    "ncrna": 0.02,
    "promoters": 0.0002,
}


def count_text(text: str) -> Counter:
    upper = text.upper()
    return Counter({base: upper.count(base) for base in BASES})


def total_valid(counts: Counter) -> int:
    return sum(counts[base] for base in BASES)


def proportions(counts: Counter) -> dict[str, float]:
    n = total_valid(counts)
    if not n:
        raise ValueError("No valid A/C/G/T bases were counted")
    return {base: counts[base] / n for base in BASES}


def sample_opengenome1() -> tuple[Counter, dict]:
    """One deterministic row group from every stage-one training shard.

    Sampling all twelve GTDB shards plus IMG/PR prevents a single contiguous
    shard from determining the estimate while keeping remote range reads small.
    """
    rng = random.Random(SEED)
    repo = "LongSafari/open-genome"
    paths = [
        f"stage1/gtdb/gtdb_train_shard_{i}.parquet" for i in range(12)
    ] + ["stage1/imgpr/imgpr_train.parquet"]
    counts = Counter()
    details = []
    for path in paths:
        url = HF_RESOLVE.format(repo=repo, path=path)
        with fsspec.open(url, "rb", block_size=8 << 20, cache_type="readahead") as handle:
            parquet = pq.ParquetFile(handle)
            row_group = rng.randrange(parquet.num_row_groups)
            table = parquet.read_row_group(row_group, columns=["text"])
            local = Counter()
            for item in table["text"]:
                local.update(count_text(item.as_py()))
            counts.update(local)
            details.append(
                {
                    "path": path,
                    "row_group": row_group,
                    "row_groups": parquet.num_row_groups,
                    "valid_bases": total_valid(local),
                }
            )
            print(f"Evo 1: {path} row-group {row_group}: {total_valid(local):,} bp", flush=True)
    return counts, {
        "repository": repo,
        "definition": "stratified stage-1 sample: one random row group from each of 12 GTDB shards and IMG/PR",
        "seed": SEED,
        "details": details,
    }


def og2_training_paths() -> dict[str, list[str]]:
    api = HfApi()
    grouped = defaultdict(list)
    prefix = "json/pretraining_or_both_phases/"
    for item in api.list_repo_tree(
        "arcinstitute/opengenome2",
        repo_type="dataset",
        recursive=True,
        expand=True,
    ):
        path = getattr(item, "path", "")
        name = path.rsplit("/", 1)[-1]
        if path.startswith(prefix) and re.search(r"_train_chunk\d+\.jsonl\.gz$", name):
            category = path.split("/")[2]
            grouped[category].append(path)
    missing = set(OG2_WEIGHTS) - set(grouped)
    if missing:
        raise RuntimeError(f"Missing OpenGenome2 phase-one categories: {sorted(missing)}")
    return {key: sorted(value) for key, value in grouped.items()}


def count_remote_jsonl(path: str, target: int) -> Counter:
    url = HF_RESOLVE.format(repo="arcinstitute/opengenome2", path=path)
    counts = Counter()
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for line in stream:
                record = json.loads(line)
                counts.update(count_text(record["text"]))
                if total_valid(counts) >= target:
                    break
    return counts


def sample_opengenome2(target_bases: int = 200_000_000) -> tuple[dict[str, float], dict]:
    """Estimate the phase-one dataloader-weighted base distribution.

    Within each source category, up to three deterministic random training
    shards are sampled. Category compositions are then combined using the
    published phase-one dataloader weights rather than raw repository size.
    """
    rng = random.Random(SEED)
    grouped = og2_training_paths()
    weight_sum = sum(OG2_WEIGHTS.values())
    normalized_weights = {key: value / weight_sum for key, value in OG2_WEIGHTS.items()}
    category_results = {}
    sampled_total = 0
    for category, weight in normalized_weights.items():
        quota = max(50_000, round(target_bases * weight))
        candidates = grouped[category]
        chosen = rng.sample(candidates, min(3, len(candidates)))
        per_shard = max(20_000, quota // len(chosen))
        category_counts = Counter()
        shard_details = []
        for path in chosen:
            local = count_remote_jsonl(path, per_shard)
            category_counts.update(local)
            shard_details.append({"path": path, "valid_bases": total_valid(local)})
            print(
                f"Evo 2: {category}: {path.rsplit('/', 1)[-1]}: "
                f"{total_valid(local):,} bp",
                flush=True,
            )
        sampled_total += total_valid(category_counts)
        category_results[category] = {
            "weight": weight,
            "counts": dict(category_counts),
            "proportions": proportions(category_counts),
            "shards": shard_details,
        }

    weighted = {
        base: sum(
            category_results[category]["weight"]
            * category_results[category]["proportions"][base]
            for category in normalized_weights
        )
        for base in BASES
    }
    return weighted, {
        "repository": "arcinstitute/opengenome2",
        "definition": "phase-one source-stratified sample, combined using published dataloader weights",
        "seed": SEED,
        "sampled_valid_bases": sampled_total,
        "target_valid_bases": target_bases,
        "categories": category_results,
    }


def count_hg38() -> tuple[Counter, dict]:
    """Count the complete UCSC hg38 reference used as the HyenaDNA corpus."""
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    counts = Counter()
    records = 0
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for raw in stream:
                if raw.startswith(b">"):
                    records += 1
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                counts.update(count_text(line))
                if records and records % 100 == 0:
                    print(f"hg38: reached FASTA record {records}", flush=True)
    return counts, {
        "source": url,
        "definition": "complete UCSC hg38 reference",
        "fasta_records": records,
    }


def human_bases(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} billion"
    return f"{value / 1_000_000:.1f} million"


def plot_figure(results: dict, output_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.25))
    panels = [
        ("a", "Evo 1", "OpenGenome", results["evo1"]),
        ("b", "Evo 2", "OpenGenome2", results["evo2"]),
        ("c", "HyenaDNA", "hg38", results["hyenadna"]),
    ]
    for ax, (letter, model, corpus, entry) in zip(axes, panels):
        values = [entry["proportions"][base] for base in BASES]
        wedges, _ = ax.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=[BASE_COLORS[base] for base in BASES],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            radius=0.92,
        )
        ax.text(-0.08, 1.10, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="left", va="top")
        ax.set_title(model, fontsize=11, pad=9)
        ax.text(0.5, -0.03, corpus, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.7, color="#3d3d3d")
        n_label = human_bases(entry["n_bases"])
        ax.text(0.5, -0.115, f"{n_label} valid bases", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.6, color="#707070")

        # Direct labels make every panel independently readable and avoid a
        # detached legend requiring repeated eye movements.
        for wedge, base, value in zip(wedges, BASES, values):
            angle = (wedge.theta1 + wedge.theta2) / 2
            import math

            x = 0.60 * math.cos(math.radians(angle))
            y = 0.60 * math.sin(math.radians(angle))
            text_color = "white" if base in {"A", "C", "T"} else "#1a1a1a"
            ax.text(x, y, f"{base}\n{100 * value:.1f}%", ha="center", va="center",
                    fontsize=8.3, fontweight="bold", color=text_color,
                    linespacing=0.92)
        ax.set_aspect("equal")

    fig.suptitle("Nucleotide composition of model pretraining data",
                 fontsize=12.2, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.025,
        "Ambiguous symbols excluded. Evo 1: stage-1 stratified sample; "
        "Evo 2: phase-one sample weighted by published dataloader mixture; "
        "hg38: complete reference.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.17, wspace=0.22)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--og2-target", type=int, default=200_000_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    evo1_counts, evo1_meta = sample_opengenome1()
    evo2_props, evo2_meta = sample_opengenome2(args.og2_target)
    hg38_counts, hg38_meta = count_hg38()

    results = {
        "evo1": {
            "counts": dict(evo1_counts),
            "proportions": proportions(evo1_counts),
            "n_bases": total_valid(evo1_counts),
            "metadata": evo1_meta,
        },
        "evo2": {
            "counts": None,
            "proportions": evo2_props,
            "n_bases": evo2_meta["sampled_valid_bases"],
            "metadata": evo2_meta,
        },
        "hyenadna": {
            "counts": dict(hg38_counts),
            "proportions": proportions(hg38_counts),
            "n_bases": total_valid(hg38_counts),
            "metadata": hg38_meta,
        },
    }

    json_path = args.out_dir / "training_corpus_nucleotide_composition.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    csv_path = args.out_dir / "training_corpus_nucleotide_composition.csv"
    with csv_path.open("w") as handle:
        handle.write("model,corpus,n_valid_bases,p_A,p_C,p_G,p_T,p_AT,p_GC\n")
        for model, corpus in [("evo1", "OpenGenome"), ("evo2", "OpenGenome2"),
                              ("hyenadna", "hg38")]:
            row = results[model]
            p = row["proportions"]
            handle.write(
                f"{model},{corpus},{row['n_bases']},"
                f"{p['A']:.10f},{p['C']:.10f},{p['G']:.10f},{p['T']:.10f},"
                f"{p['A'] + p['T']:.10f},{p['C'] + p['G']:.10f}\n"
            )

    pdf_path = args.out_dir / "supp_training_corpus_nucleotide_composition.pdf"
    plot_figure(results, pdf_path)
    print(f"WROTE {json_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {pdf_path}")


if __name__ == "__main__":
    main()
