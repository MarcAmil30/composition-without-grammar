#!/usr/bin/env python3
"""Recover raw A/C/G/T probabilities for one completed generation block."""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(f"{ICBINB_ROOT}/data")
BASES = "ACGT"
EPSILON = 1e-12


def load_code(data_root: Path):
    generation = data_root / "code/generation"
    experiment = data_root / "code/fig2d_long_context"
    sys.path.insert(0, str(generation))
    sys.path.insert(0, str(experiment))
    import framing  # type: ignore
    import models  # type: ignore
    from run_generation import register  # type: ignore
    register(models)
    return framing, models


def base_ids(adapter) -> list[int]:
    if adapter.tok is None:
        return [ord(base) for base in BASES]
    result = []
    for base in BASES:
        ids = adapter.tok(base, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"{base} is not a single token: {ids}")
        result.append(ids[0])
    return result


def encode(adapter, sequences: list[str]):
    import torch
    if adapter.tok is None:
        tokenizer = adapter.model.tokenizer
        return torch.tensor(
            [tokenizer.tokenize(sequence) for sequence in sequences],
            dtype=torch.long, device=adapter.device,
        )
    return adapter.tok(sequences, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(adapter.device)


def logits(adapter, ids):
    if adapter.tok is None:
        output = adapter.model.forward(ids)
        while isinstance(output, (tuple, list)):
            output = output[0]
        return output.float()
    return adapter.model(ids).logits.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    framing, models = load_code(args.data_root)
    exp = args.data_root / "benchmark/fig2d_long_context"
    stem = f"genes_{args.start:04d}_{args.stop:04d}"
    generation_path = exp / "generations" / args.model / f"ctx{args.context}" / f"{stem}.csv.gz"
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)
    output_dir = exp / "probabilities" / args.model / f"ctx{args.context}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.csv.gz"
    if output_path.exists() and metadata_path.exists():
        print(f"[skip] {output_path}")
        return

    generations = pd.read_csv(generation_path, keep_default_na=False)
    prompt_path = exp / "prompts" / f"prompts_681_ctx{args.context}.csv.gz"
    prompts = pd.read_csv(prompt_path).set_index("locus_tag")
    adapter = models.build(args.model).load()
    ids_acgt = base_ids(adapter)

    rows = []
    for row in generations.itertuples(index=False):
        source = prompts.loc[row.locus_tag].to_dict() | {"locus_tag": row.locus_tag}
        framed = framing.frame(source, args.model, "generate_promoter")
        rows.append((row, framed.prompt, framed.prompt + row.generation))

    import torch
    probabilities = np.full((len(rows), 45, 4), np.nan, dtype=np.float32)
    p_off = np.full((len(rows), 45), np.nan, dtype=np.float32)
    metadata = []
    for begin in range(0, len(rows), args.batch):
        chunk = rows[begin:begin + args.batch]
        lengths = {len(full) for _, _, full in chunk}
        if len(lengths) != 1:
            # Long-context prompts differ slightly by promoter geometry. Keep
            # exact no-padding semantics by falling back to individual passes.
            for single in chunk:
                rows.insert(begin + 1, single)
            raise RuntimeError("batch contained unequal lengths; use --batch 1")
        encoded = encode(adapter, [full for _, _, full in chunk])
        with torch.inference_mode():
            output_logits = logits(adapter, encoded)
        for local, (row, prompt, full) in enumerate(chunk):
            length = len(row.generation)
            selected = output_logits[local, len(prompt) - 1:len(prompt) - 1 + length]
            full_probs = torch.softmax(selected, dim=-1)
            p4 = full_probs[:, ids_acgt].cpu().numpy()
            probabilities[begin + local, :length] = p4
            p_off[begin + local, :length] = 1.0 - p4.sum(axis=1)
            metadata.append({
                "row_index": begin + local,
                "locus_tag": row.locus_tag,
                "decode": row.decode,
                "sample_idx": row.sample_idx,
                "prompt_len": len(prompt),
                "generation": row.generation,
            })
        print(f"{min(begin + args.batch, len(rows))}/{len(rows)}", flush=True)

    q = probabilities / np.clip(probabilities.sum(axis=-1, keepdims=True), EPSILON, None)
    entropy = -(q * np.log2(np.clip(q, EPSILON, None))).sum(axis=-1)
    temp = output_path.with_suffix(".partial.npz")
    np.savez_compressed(temp, p_acgt=probabilities, p_off=p_off, entropy_bits=entropy,
                        bases=np.array(list(BASES)))
    os.replace(temp, output_path)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False, compression="gzip")
    manifest = {
        "model": args.model, "context_bp": args.context,
        "start": args.start, "stop": args.stop, "n_generations": len(rows),
        "shape": list(probabilities.shape), "source": str(generation_path),
        "output": str(output_path), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output_dir.joinpath(f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
