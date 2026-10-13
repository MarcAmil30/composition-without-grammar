#!/usr/bin/env python3
"""
Generation driver for the upstream-only benchmark.

One run = one (model, task, background) shard. Writes generations to disk and
nothing else -- no scoring happens here, so PWMs, Promoter Calculator and RBS
Calculator can all be built and revised afterwards without re-running any model.

    # validate framing only -- no GPU, no weights, safe to run anywhere
    python generate.py --dry-run

    # 2 genes through one model, to prove the adapter works
    python generate.py --model promogen2 --task generate_sd --background leaked \
                       --smoke

    # a real shard
    python generate.py --model evo1 --task generate_promoter --background clean

Output: <out>/<model>/<task>__<background>.csv plus a .manifest.json recording
the checkpoint, sampling config, framing, and every dropped gene.

Resume is automatic: a shard whose CSV already exists is skipped unless --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import framing                                     # noqa: E402
from framing import BACKGROUNDS, GEN_LEN, TASKS    # noqa: E402
import models                                      # noqa: E402
from models import DEFAULT_SAMPLING, REGISTRY, SamplingConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = os.environ.get("BENCH_DIR", REPO_ROOT / "data" / "upstream_only")
DATA = os.fspath(DATA)
OUT = os.environ.get("GEN_DIR", os.path.join(DATA, "generations"))

FIELDS = ["locus_tag", "model", "task", "background", "decode", "sample_idx",
          "prompt_len", "generation", "gen_len", "framing_note"]


def load_prompts(background: str):
    path = os.path.join(DATA, f"prompts_681_{background}.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def frame_all(rows, model: str, task: str):
    """Frame every row; return (kept, dropped)."""
    framed = [framing.frame(r, model, task) for r in rows]
    return [f for f in framed if not f.dropped], [f for f in framed if f.dropped]


def batched_by_length(items, n):
    """Yield batches of at most n items, all with the same prompt length.

    Equal-length batching removes padding entirely. Necessary because Evo's
    ByteTokenizer has no pad token, and sufficient to avoid the right-padding
    trap where a causal model generates after the pads.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for it in items:
        groups[len(it.prompt)].append(it)
    for L in sorted(groups):
        g = groups[L]
        for i in range(0, len(g), n):
            yield g[i:i + n]


# =====================================================================
def run_shard(model_key, task, background, cfg: SamplingConfig,
              out_dir=OUT, force=False, limit=None, device="cuda"):
    spec = REGISTRY[model_key]
    if task not in spec.tasks:
        print(f"[skip] {model_key} does not run {task}")
        return

    shard_dir = os.path.join(out_dir, model_key)
    os.makedirs(shard_dir, exist_ok=True)
    csv_path = os.path.join(shard_dir, f"{task}__{background}.csv")
    tmp_path = csv_path + ".partial"
    man_path = csv_path.replace(".csv", ".manifest.json")

    if os.path.exists(csv_path) and not force:
        print(f"[skip] {csv_path} exists (use --force to overwrite)")
        return

    rows = load_prompts(background)
    kept, dropped = frame_all(rows, model_key, task)
    if limit:
        kept = kept[:limit]

    n_bp = GEN_LEN[task]
    print(f"[{model_key}] {task} / {background}: {len(kept)} genes, "
          f"{len(dropped)} dropped, generate {n_bp} bp, "
          f"{cfg.n_samples} samples + {'greedy' if cfg.greedy else 'no greedy'}")

    adapter = models.build(model_key, device=device).load()

    n_done = 0
    with open(tmp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for chunk in batched_by_length(kept, 512):
            res = adapter.continue_([f.prompt for f in chunk], n_bp, cfg)
            for f, r in zip(chunk, res):
                base = dict(locus_tag=f.locus_tag, model=model_key, task=task,
                            background=background, prompt_len=len(f.prompt),
                            framing_note=f.note)
                if r["greedy"] is not None:
                    w.writerow({**base, "decode": "greedy", "sample_idx": -1,
                                "generation": r["greedy"], "gen_len": len(r["greedy"])})
                for i, s in enumerate(r["samples"]):
                    w.writerow({**base, "decode": "sample", "sample_idx": i,
                                "generation": s, "gen_len": len(s)})
            n_done += len(chunk)
            print(f"  {n_done}/{len(kept)}", flush=True)

    # only now is the shard real -- a crash leaves a .partial, not a stub that
    # would make the next run think this shard was already done
    os.replace(tmp_path, csv_path)

    json.dump({
        "model": model_key,
        "display": spec.display,
        "checkpoint": spec.checkpoint,
        "notes": spec.notes,
        "task": task,
        "background": background,
        "gen_len_bp": n_bp,
        "sampling": cfg.as_dict(),
        "n_genes_generated": len(kept),
        "n_genes_dropped": len(dropped),
        "dropped": [{"locus_tag": d.locus_tag, "reason": d.drop_reason} for d in dropped],
        "framing_notes": sorted({f.note for f in kept}),
    }, open(man_path, "w"), indent=2)

    print(f"[done] {csv_path}")


# =====================================================================
def dry_run():
    """Validate framing for every model x task x background. No weights loaded."""
    print(f"benchmark dir: {DATA}\noutput dir:    {OUT}\n")
    print("sampling (identical for every model):", DEFAULT_SAMPLING.as_dict(), "\n")

    total = 0
    for bg in BACKGROUNDS:
        rows = load_prompts(bg)
        for task in TASKS:
            print(f"--- {task} / {bg}  (generate {GEN_LEN[task]} bp) ---")
            for key, spec in REGISTRY.items():
                if task not in spec.tasks:
                    print(f"  {key:10s} n/a")
                    continue
                kept, dropped = frame_all(rows, key, task)
                lens = sorted(len(f.prompt) for f in kept)
                n_rows = len(kept) * (DEFAULT_SAMPLING.n_samples + int(DEFAULT_SAMPLING.greedy))
                total += n_rows
                print(f"  {key:10s} {len(kept):3d} genes  {len(dropped):2d} dropped  "
                      f"prompt {lens[0]:3d}-{lens[-1]:3d} bp  "
                      f"-> {n_rows:6d} generations")
            print()
    print(f"TOTAL generations across the whole benchmark: {total:,}")
    print("\nframing self-check:")
    os.system(f"{sys.executable} {os.path.dirname(os.path.abspath(__file__))}/framing.py {DATA} "
              "| tail -3")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(REGISTRY))
    p.add_argument("--task", choices=list(TASKS))
    p.add_argument("--background", choices=list(BACKGROUNDS))
    p.add_argument("--all", action="store_true",
                   help="run every model x task x background in sequence")
    p.add_argument("--dry-run", action="store_true",
                   help="validate framing and print the work plan; loads nothing")
    p.add_argument("--smoke", action="store_true",
                   help="2 genes, 2 samples -- proves the adapter works")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=OUT)
    p.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLING.n_samples)
    p.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING.temperature)
    p.add_argument("--batch-size", type=int, default=DEFAULT_SAMPLING.batch_size)
    p.add_argument("--seed", type=int, default=DEFAULT_SAMPLING.seed)
    a = p.parse_args()

    if a.dry_run:
        return dry_run()

    cfg = SamplingConfig(n_samples=2 if a.smoke else a.n_samples,
                         temperature=a.temperature,
                         batch_size=a.batch_size,
                         seed=a.seed)
    limit = 2 if a.smoke else None

    if a.all:
        jobs = [(m, t, b) for m in REGISTRY for t in REGISTRY[m].tasks for b in BACKGROUNDS]
    else:
        if not (a.model and a.task and a.background):
            p.error("give --model, --task and --background, or use --all / --dry-run")
        jobs = [(a.model, a.task, a.background)]

    for m, t, b in jobs:
        run_shard(m, t, b, cfg, out_dir=a.out, force=a.force or a.smoke,
                  limit=limit, device=a.device)


if __name__ == "__main__":
    main()
