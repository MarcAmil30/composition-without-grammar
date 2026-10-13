#!/usr/bin/env python3
"""Run one resumable Figure 2d long-context generation block on one GPU."""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import argparse
import contextlib
import csv
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

DEFAULT_DATA = Path(f"{ICBINB_ROOT}/data")


def import_generation(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    import framing  # type: ignore
    import models  # type: ignore
    return framing, models


def register(models) -> dict[str, str]:
    """Add explicit long-checkpoint keys without altering the old benchmark."""
    models.REGISTRY["evo1_131k"] = models.ModelSpec(
        "evo1_131k", "Evo 1 (131k)",
        "togethercomputer/evo-1-131k-base", "hf_causal",
        notes="Separate Evo 1 checkpoint trained at 131,072 context.",
        kwargs={"trust_remote_code": True}, seqs_per_forward=1,
    )
    models.REGISTRY["evo2_7b"] = models.ModelSpec(
        "evo2_7b", "Evo 2 (7B, 1M)", "arcinstitute/evo2_7b", "evo2",
        notes="Separate Evo 2 checkpoint context-extended to 1M.",
        seqs_per_forward=1,
    )
    return {
        "evo1": "evo1",
        "evo2": "evo2",
        "hyenadna": "hyenadna",
        "evo1_131k": "evo1_131k",
        "evo2_7b": "evo2_7b",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-width", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-group", default="generations",
                        help="experiment subdirectory; use 'smokes' for pilots")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--generation-code", type=Path)
    args = parser.parse_args()

    code_dir = args.generation_code or args.data_root / "code/generation"
    framing, models = import_generation(code_dir)
    allowed = register(models)
    if args.model not in allowed:
        raise ValueError(f"unknown model {args.model}; choose {sorted(allowed)}")
    spec = models.REGISTRY[args.model]
    spec.seqs_per_forward = args.forward_width

    experiment = args.data_root / "benchmark/fig2d_long_context"
    prompt_path = experiment / "prompts" / f"prompts_681_ctx{args.context}.csv.gz"
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    prompt_manifest_path = prompt_path.with_suffix(".manifest.json")
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    import pandas as pd
    prompts = pd.read_csv(prompt_path)
    stop = min(len(prompts), args.start + args.limit)
    prompts = prompts.iloc[args.start:stop]
    if prompts.empty:
        raise ValueError("empty gene block")

    out_dir = experiment / args.output_group / args.model / f"ctx{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"genes_{args.start:04d}_{stop:04d}"
    out_path = out_dir / f"{stem}.csv.gz"
    manifest_path = out_dir / f"{stem}.manifest.json"
    if out_path.exists() and manifest_path.exists() and not args.force:
        print(f"[skip] validated block exists: {out_path}")
        return

    cfg = models.SamplingConfig(
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=0,
        top_p=1.0,
        greedy=True,
        seed=args.seed,
        batch_size=args.forward_width,
    )
    framed = [framing.frame(row, args.model, "generate_promoter")
              for row in prompts.to_dict("records")]
    if any(item.dropped for item in framed):
        raise RuntimeError("unexpected dropped prompt in long-context task")

    print(
        f"model={args.model} checkpoint={spec.checkpoint} context={args.context} "
        f"genes={args.start}:{stop} samples={args.n_samples} width={args.forward_width}",
        flush=True,
    )
    adapter = models.build(args.model).load()
    fields = ["locus_tag", "model", "checkpoint", "context_bp", "gene_index",
              "decode", "sample_idx", "seed", "prompt_len", "prompt_sha256",
              "generation", "gen_len"]
    temp = out_path.with_suffix(out_path.suffix + ".partial")
    with __import__("gzip").open(temp, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, (source, item) in enumerate(zip(prompts.to_dict("records"), framed)):
            # Vortex prints every complete prompt and continuation. At 1M bp
            # this would create multi-gigabyte Slurm logs and stress metadata
            # storage. Keep errors on stderr and emit our own per-gene progress.
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                result = adapter.continue_([item.prompt], framing.GEN_LEN["generate_promoter"], cfg)[0]
            base = {
                "locus_tag": item.locus_tag,
                "model": args.model,
                "checkpoint": spec.checkpoint,
                "context_bp": args.context,
                "gene_index": args.start + offset,
                "seed": args.seed,
                "prompt_len": len(item.prompt),
                "prompt_sha256": source["prompt_sha256"],
            }
            if result["greedy"] is not None:
                seq = result["greedy"]
                writer.writerow({**base, "decode": "greedy", "sample_idx": -1,
                                 "generation": seq, "gen_len": len(seq)})
            for sample_idx, seq in enumerate(result["samples"]):
                writer.writerow({**base, "decode": "sample", "sample_idx": sample_idx,
                                 "generation": seq, "gen_len": len(seq)})
            handle.flush()
            print(f"gene {args.start + offset + 1}/{stop} {item.locus_tag}", flush=True)
    os.replace(temp, out_path)

    manifest = {
        "status": "complete",
        "model": args.model,
        "checkpoint": spec.checkpoint,
        "context_bp": args.context,
        "start": args.start,
        "stop": stop,
        "n_genes": stop - args.start,
        "sampling": cfg.as_dict(),
        "forward_width": args.forward_width,
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": prompt_manifest["output_sha256"],
        "output": str(out_path),
        "output_group": args.output_group,
        "output_sha256": sha256(out_path),
        "effective_top_k": 4 if args.model in {"evo2", "evo2_7b"} else 0,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
