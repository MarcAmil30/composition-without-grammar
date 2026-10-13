#!/usr/bin/env python3
"""CPU-only composition of non-coding sequence represented in training data.

Definitions
-----------
Evo 1
    Bases outside Pyrodigal-predicted CDS intervals in the same reproducible
    OpenGenome stage-one sample used by ``analyse_and_plot.py``.
Evo 2
    Bases outside Pyrodigal-predicted CDS intervals in microbial, metagenomic,
    viral and organellar phase-one sources, plus the explicitly non-coding
    promoter and ncRNA sources. Source weights are renormalized over this
    annotatable 47.3% of the published phase-one dataloader mixture. Eukaryotic
    genic and mRNA-derived sources are excluded because the released JSONL
    records do not carry CDS coordinates.
HyenaDNA
    hg38 bases outside merged GENCODE v49 CDS intervals.

Ambiguous symbols are excluded throughout. Non-coding does not imply regulatory.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import fsspec
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyrodigal
import requests


SEED = 20260820
BASES = "ACGT"
BASE_COLORS = {"A": "#2f8f4e", "C": "#2f6fb0", "G": "#e0a33e", "T": "#c94f4f"}
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/"
    "gencode.v49.annotation.gtf.gz"
)

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
OG2_PREDICTED = {"gtdb_v220_imgpr", "metagenomes", "imgvr_untagged", "organelle"}
OG2_EXPLICIT_NONCODING = {"ncrna", "promoters"}
OG2_INCLUDED = OG2_PREDICTED | OG2_EXPLICIT_NONCODING


def clean_sequence(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch in BASES)


def count_text(text: str) -> Counter:
    text = text.upper()
    return Counter({base: text.count(base) for base in BASES})


def total(counts: Counter) -> int:
    return sum(counts[base] for base in BASES)


def props(counts: Counter) -> dict[str, float]:
    n = total(counts)
    if not n:
        raise ValueError("No non-coding A/C/G/T bases counted")
    return {base: counts[base] / n for base in BASES}


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def predicted_noncoding(text: str, finder: pyrodigal.GeneFinder) -> tuple[Counter, int]:
    seq = clean_sequence(text)
    if len(seq) < 60:
        return count_text(seq), 0
    genes = finder.find_genes(seq.encode("ascii"))
    # Pyrodigal uses one-based inclusive coordinates.
    coding = merge_intervals([(max(0, gene.begin - 1), min(len(seq), gene.end)) for gene in genes])
    counts = Counter()
    cursor = 0
    coding_bases = 0
    for start, end in coding:
        if start > cursor:
            counts.update(count_text(seq[cursor:start]))
        coding_bases += max(0, end - start)
        cursor = max(cursor, end)
    if cursor < len(seq):
        counts.update(count_text(seq[cursor:]))
    return counts, coding_bases


def evo1_noncoding(previous: dict, finder: pyrodigal.GeneFinder) -> tuple[Counter, dict]:
    counts = Counter()
    input_bases = coding_bases = records = 0
    repo = "LongSafari/open-genome"
    for detail in previous["evo1"]["metadata"]["details"]:
        path = detail["path"]
        url = HF_RESOLVE.format(repo=repo, path=path)
        with fsspec.open(url, "rb", block_size=8 << 20, cache_type="readahead") as handle:
            table = pq.ParquetFile(handle).read_row_group(detail["row_group"], columns=["text"])
            for item in table["text"]:
                seq = item.as_py()
                valid = clean_sequence(seq)
                local, coding = predicted_noncoding(valid, finder)
                counts.update(local)
                input_bases += len(valid)
                coding_bases += coding
                records += 1
        print(f"Evo 1 non-coding: {path}: {total(counts):,} bp cumulative", flush=True)
    return counts, {
        "input_valid_bases": input_bases,
        "predicted_cds_bases": coding_bases,
        "noncoding_bases": total(counts),
        "records": records,
        "method": "Pyrodigal metagenomic-mode CDS complement",
        "seed": SEED,
    }


def iter_remote_jsonl(path: str, target_input_bases: int):
    url = HF_RESOLVE.format(repo="arcinstitute/opengenome2", path=path)
    seen = 0
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for line in stream:
                text = json.loads(line)["text"]
                seen += len(clean_sequence(text))
                yield text
                if seen >= target_input_bases:
                    break


def evo2_noncoding(previous: dict, finder: pyrodigal.GeneFinder) -> tuple[dict[str, float], int, dict]:
    categories = previous["evo2"]["metadata"]["categories"]
    included_weight = sum(OG2_WEIGHTS[key] for key in OG2_INCLUDED)
    full_weight = sum(OG2_WEIGHTS.values())
    coverage = included_weight / full_weight
    category_output = {}
    sampled_noncoding = 0
    for category in sorted(OG2_INCLUDED):
        cat_counts = Counter()
        input_bases = coding_bases = records = 0
        for shard in categories[category]["shards"]:
            for text in iter_remote_jsonl(shard["path"], shard["valid_bases"]):
                valid = clean_sequence(text)
                input_bases += len(valid)
                records += 1
                if category in OG2_PREDICTED:
                    local, coding = predicted_noncoding(valid, finder)
                    coding_bases += coding
                    cat_counts.update(local)
                else:
                    cat_counts.update(count_text(valid))
        sampled_noncoding += total(cat_counts)
        category_output[category] = {
            "counts": dict(cat_counts),
            "proportions": props(cat_counts),
            "input_valid_bases": input_bases,
            "predicted_cds_bases": coding_bases,
            "noncoding_bases": total(cat_counts),
            "records": records,
            "original_weight": OG2_WEIGHTS[category],
        }
        print(f"Evo 2 non-coding: {category}: {total(cat_counts):,} bp", flush=True)

    weighted = {
        base: sum(
            (OG2_WEIGHTS[category] / included_weight)
            * category_output[category]["proportions"][base]
            for category in OG2_INCLUDED
        )
        for base in BASES
    }
    return weighted, sampled_noncoding, {
        "phase1_mixture_coverage": coverage,
        "included_categories": sorted(OG2_INCLUDED),
        "excluded_categories_no_cds_coordinates": sorted(set(OG2_WEIGHTS) - OG2_INCLUDED),
        "method": "Pyrodigal CDS complement plus explicit promoter/ncRNA sources; weights renormalized",
        "categories": category_output,
    }


def load_gencode_cds() -> dict[str, list[tuple[int, int]]]:
    intervals = defaultdict(list)
    with requests.get(GENCODE_URL, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for raw in stream:
                if raw.startswith(b"#"):
                    continue
                fields = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) >= 5 and fields[2] == "CDS":
                    intervals[fields[0]].append((int(fields[3]) - 1, int(fields[4])))
    merged = {chrom: merge_intervals(values) for chrom, values in intervals.items()}
    print(f"GENCODE: merged CDS intervals on {len(merged)} sequences", flush=True)
    return merged


def hg38_noncoding() -> tuple[Counter, dict]:
    intervals = load_gencode_cds()
    counts = Counter()
    input_bases = coding_bases = records = annotated_records = 0
    chrom = None
    offset = 0
    active = []
    index = 0
    with requests.get(HG38_URL, stream=True, timeout=(30, 900)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for raw in stream:
                if raw.startswith(b">"):
                    chrom = raw[1:].decode("ascii", errors="ignore").split()[0]
                    active = intervals.get(chrom)
                    index = 0
                    offset = 0
                    records += 1
                    if active is not None:
                        annotated_records += 1
                    continue
                # GENCODE v49 annotates the primary chromosomes. Exclude hg38
                # alternate/unplaced records rather than falsely treating an
                # unannotated record as entirely non-coding.
                if active is None:
                    continue
                seq = clean_sequence(raw.decode("ascii", errors="ignore"))
                start, end = offset, offset + len(seq)
                input_bases += len(seq)
                while index < len(active) and active[index][1] <= start:
                    index += 1
                cursor = start
                j = index
                while j < len(active) and active[j][0] < end:
                    cds_start, cds_end = active[j]
                    left = max(start, cds_start)
                    right = min(end, cds_end)
                    if left > cursor:
                        counts.update(count_text(seq[cursor - start:left - start]))
                    if right > left:
                        coding_bases += right - left
                    cursor = max(cursor, right)
                    if cds_end <= end:
                        j += 1
                    else:
                        break
                if cursor < end:
                    counts.update(count_text(seq[cursor - start:]))
                offset = end
    return counts, {
        "input_valid_bases": input_bases,
        "annotated_cds_bases": coding_bases,
        "noncoding_bases": total(counts),
        "fasta_records": records,
        "annotated_fasta_records": annotated_records,
        "method": "GENCODE-annotated hg38 chromosomes outside merged GENCODE v49 CDS intervals",
        "hg38_source": HG38_URL,
        "annotation_source": GENCODE_URL,
    }


def human_n(value: int) -> str:
    return f"{value / 1_000_000_000:.2f} billion" if value >= 1_000_000_000 else f"{value / 1_000_000:.1f} million"


def plot(results: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.45))
    panels = [
        ("A", "Evo 1", "OpenGenome", results["evo1"]),
        ("B", "Evo 2", "OpenGenome2", results["evo2"]),
        ("C", "HyenaDNA", "hg38", results["hyenadna"]),
    ]
    for ax, (letter, model, corpus, entry) in zip(axes, panels):
        values = [entry["proportions"][base] for base in BASES]
        wedges, _ = ax.pie(values, startangle=90, counterclock=False,
                           colors=[BASE_COLORS[b] for b in BASES], radius=0.91,
                           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.text(-0.08, 1.10, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="left", va="top")
        ax.set_title(model, fontsize=11, fontweight="bold", pad=9)
        for wedge, base, value in zip(wedges, BASES, values):
            angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
            color = "white" if base in {"A", "C", "T"} else "#1a1a1a"
            ax.text(0.59 * math.cos(angle), 0.59 * math.sin(angle),
                    f"{base}\n{100 * value:.1f}%", ha="center", va="center",
                    fontsize=8.3, fontweight="bold", color=color, linespacing=0.92)
        ax.text(0.5, -0.03, corpus, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.7, color="#3d3d3d")
        ax.text(0.5, -0.115, f"{human_n(entry['n_bases'])} non-coding bases",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
                color="#707070")
        ax.set_aspect("equal")

    coverage = 100 * results["evo2"]["metadata"]["phase1_mixture_coverage"]
    fig.suptitle("Nucleotide composition of non-coding training sequence",
                 fontsize=12.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.047,
             "Non-coding = bases outside CDS, not necessarily regulatory. "
             "Microbial CDS: Pyrodigal; hg38 CDS: GENCODE v49.",
             ha="center", va="bottom", fontsize=6.7, color="#555555")
    fig.text(0.5, 0.018,
             f"Evo 2 covers {coverage:.1f}% of the phase-one mixture with separable CDS labels; "
             "unannotated eukaryotic genic/mRNA sources are excluded.",
             ha="center", va="bottom", fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.85, bottom=0.19, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reuse-evo", action="store_true",
                        help="Reuse completed Evo results and recompute only corrected hg38")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "noncoding_nucleotide_composition.json"
    if args.reuse_evo:
        cached = json.loads(json_path.read_text())
        evo1_entry = cached["evo1"]
        evo2_entry = cached["evo2"]
    else:
        previous = json.loads(args.previous_json.read_text())
        finder = pyrodigal.GeneFinder(meta=True)
        evo1_counts, evo1_meta = evo1_noncoding(previous, finder)
        evo2_props, evo2_n, evo2_meta = evo2_noncoding(previous, finder)
        evo1_entry = {"counts": dict(evo1_counts), "proportions": props(evo1_counts),
                      "n_bases": total(evo1_counts), "metadata": evo1_meta}
        evo2_entry = {"counts": None, "proportions": evo2_props,
                      "n_bases": evo2_n, "metadata": evo2_meta}
    hg_counts, hg_meta = hg38_noncoding()
    results = {
        "evo1": evo1_entry,
        "evo2": evo2_entry,
        "hyenadna": {"counts": dict(hg_counts), "proportions": props(hg_counts),
                     "n_bases": total(hg_counts), "metadata": hg_meta},
    }
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    csv_path = args.out_dir / "noncoding_nucleotide_composition.csv"
    with csv_path.open("w") as handle:
        handle.write("model,corpus,n_noncoding_bases,p_A,p_C,p_G,p_T,p_AT,p_GC\n")
        for key, corpus in [("evo1", "OpenGenome"), ("evo2", "OpenGenome2"),
                            ("hyenadna", "hg38")]:
            entry = results[key]; p = entry["proportions"]
            handle.write(f"{key},{corpus},{entry['n_bases']},{p['A']:.10f},{p['C']:.10f},"
                         f"{p['G']:.10f},{p['T']:.10f},{p['A']+p['T']:.10f},"
                         f"{p['C']+p['G']:.10f}\n")
    pdf_path = args.out_dir / "supp_noncoding_nucleotide_composition.pdf"
    plot(results, pdf_path)
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {csv_path}", flush=True)
    print(f"WROTE {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
