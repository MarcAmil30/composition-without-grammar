#!/usr/bin/env python3
"""Build validated long-context MG1655 prompts for the Figure 2d extension.

This preserves the exact task used by the existing context ladder: ``width`` is
the genomic upstream window ending at the start codon, while generation begins
at the end of the annotated -35 box.  Thus the actual model prompt is shorter
than ``width`` by the gene-specific -35-to-start-codon distance.  Existing
Figure 2d uses the same convention.
"""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
PROK_REPO = _os.environ.get("PROK_REPO", "")
# -------------------------------------------------------------------

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(
    f"{ICBINB_ROOT}/data"
)
DEFAULT_GFF = Path(
    f"{PROK_REPO}/code/annotation/"
    "ecoli_annotation/GCF_000005845.2_ASM584v2_genomic.gff"
)
GEN_LEN = 45
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def load_fasta(path: Path) -> str:
    with path.open() as handle:
        seq = "".join(line.strip() for line in handle if not line.startswith(">"))
    seq = seq.upper()
    if not seq or set(seq) - set("ACGTN"):
        raise ValueError(f"invalid or empty FASTA: {path}")
    return seq


def gene_coords(path: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            attrs = dict(
                item.split("=", 1) for item in cols[8].split(";") if "=" in item
            )
            locus = attrs.get("locus_tag")
            if locus:
                start, end, strand = int(cols[3]), int(cols[4]), cols[6]
                out[locus] = (start if strand == "+" else end, strand)
    return out


def upstream(genome: str, atg: int, strand: str, width: int) -> str:
    """Return ``width`` bases 5' of the ATG, oriented 5' to 3' and circular."""
    n = len(genome)
    if strand == "+":
        indices = [(atg - 2 - k) % n for k in range(width)][::-1]
        return "".join(genome[i] for i in indices)
    indices = [(atg + k) % n for k in range(width)]
    return revcomp("".join(genome[i] for i in indices))


def build(base: pd.DataFrame, genome: str, coords: dict[str, tuple[int, str]],
          width: int, limit: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    failures: list[dict] = []
    source = base.iloc[:limit] if limit else base
    for row in source.to_dict("records"):
        locus = row["locus_tag"]
        if locus not in coords:
            failures.append({"locus_tag": locus, "reason": "missing from GFF"})
            continue
        atg, strand = coords[locus]
        seq = upstream(genome, atg, strand, width)
        overlap = min(len(seq), int(row["L"]))
        if seq[-overlap:] != str(row["upstream_seq"])[-overlap:]:
            failures.append({"locus_tag": locus, "reason": "reference overlap mismatch"})
            continue

        shift = len(seq) - int(row["L"])
        cut = int(row["cut_promoter"]) + shift
        out = dict(row)
        for key in ("m35_start", "m35_end", "m10_start", "m10_end",
                    "tss_idx", "sd_start", "sd_end"):
            out[key] = int(row[key]) + shift
        out.update(
            upstream_seq=seq,
            L=len(seq),
            cut_promoter=cut,
            prompt_promoter=seq[:cut],
            prompt_promoter_len=cut,
            target_promoter=seq[cut:cut + GEN_LEN],
            prompt_sd=seq[:len(seq) - 20],
            short_prompt=int(cut < 20),
            requested_context_bp=width,
            actual_prompt_bp=cut,
            prompt_sha256=hashlib.sha256(seq[:cut].encode()).hexdigest(),
            genome_atg_1based=atg,
            genome_strand=strand,
        )
        if int(out["m10_offset_in_target"]) != int(out["m10_start"]) - cut:
            raise AssertionError(f"-10 offset changed for {locus}")
        if int(out["tss_offset_in_target"]) != int(out["tss_idx"]) - cut:
            raise AssertionError(f"TSS offset changed for {locus}")
        # A small number of legacy annotations place the -35 box fewer than
        # 45 bp before the start codon.  Their natural target therefore ends
        # at the ATG and is shorter than the fixed 45-bp generation budget.
        # This is expected and was permitted by the original benchmark.
        rows.append(out)
    return pd.DataFrame(rows), failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gff", type=Path, default=DEFAULT_GFF)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    bench = args.root / "benchmark/upstream_only"
    base_path = bench / "prompts_681_leaked.csv"
    genome_path = args.root / "genome/mg1655.fna"
    output = args.out or (
        args.root / "benchmark/fig2d_long_context/prompts"
        / f"prompts_681_ctx{args.width}.csv.gz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_path)
    genome = load_fasta(genome_path)
    coords = gene_coords(args.gff)
    built, failures = build(base, genome, coords, args.width, args.limit)
    expected_source = min(len(base), args.limit) if args.limit else len(base)
    if len(built) + len(failures) != expected_source:
        raise RuntimeError("prompt row count changed")
    # The published ladder already excludes b4522 because the reference/GFF
    # overlap does not reproduce its legacy 200-bp sequence. Preserve that
    # documented exclusion, but reject any new failure.
    unexpected = [item for item in failures if item["locus_tag"] != "b4522"]
    if unexpected:
        raise RuntimeError(f"unexpected prompt failures: {unexpected[:5]}")

    temp = output.with_suffix(output.suffix + ".partial")
    # Level 1 is intentionally used: these files live on capacious scratch and
    # are read once per job, while level-9 compression of the 250k-1M ladders
    # needlessly occupies the login CPU for many minutes.
    built.to_csv(temp, index=False, compression={"method": "gzip", "compresslevel": 1})
    os.replace(temp, output)
    manifest = {
        "width_bp": args.width,
        "n_genes": len(built),
        "excluded": failures,
        "genome_bp": len(genome),
        "base_prompts": str(base_path),
        "genome": str(genome_path),
        "gff": str(args.gff),
        "actual_prompt_min": int(built.actual_prompt_bp.min()),
        "actual_prompt_median": float(built.actual_prompt_bp.median()),
        "actual_prompt_max": int(built.actual_prompt_bp.max()),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
