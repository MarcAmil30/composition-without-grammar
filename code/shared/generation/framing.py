"""
Per-model input framing for the upstream-only benchmark.

The sampling procedure is identical for every model (20 samples at T=1.0 plus one
greedy decode). The ONLY thing that varies per model is how the prompt is framed
before it is handed to the tokenizer. That lives here, isolated, so nothing
model-specific can leak into the sampling path.

Two models need framing:

  PROK        prefix the prompt with `+$`  (E. coli species token + upstream
              delimiter). PROK was finetuned on `<species_token>$<200bp upstream>`
              rows; without the prefix the prompt is off-distribution.

  PromoGen2   reframe to its native window: the last 160 bp of the 200 bp
              upstream, because that is the length it was trained on (160 bp
              ending at the translation start site). Genes whose -35 box falls
              outside that window are DROPPED, not truncated -- the task premise
              ("you are given the -35") cannot hold for them.

Everything else is passed through unchanged.

Run this file directly to validate the framing against the prompt CSVs without
loading a single model:

    python framing.py /path/to/benchmark/upstream_only
"""
from __future__ import annotations

from dataclasses import dataclass

# PromoGen2 was trained on 160 bp windows ending at the start codon.
PROMOGEN2_WINDOW = 160
# E. coli species token + upstream-segment delimiter, as used in PROK finetuning.
PROK_PREFIX = "+$"

TASKS = ("generate_promoter", "generate_sd")
BACKGROUNDS = ("leaked", "clean")

# Generation budget per task, in bp. Fixed before generation -- changing either
# of these invalidates every output produced so far.
GEN_LEN = {"generate_promoter": 45, "generate_sd": 40}


@dataclass
class Framed:
    """One gene's prompt, ready for a specific model."""
    locus_tag: str
    prompt: str                 # exactly what the tokenizer receives
    gen_len: int                # how many bp to generate
    offset_shift: int           # subtract this from annotation offsets to align
    note: str                   # what was done, recorded with the generation

    dropped: bool = False
    drop_reason: str = ""


def _passthrough(row, task: str) -> Framed:
    prompt = row["prompt_promoter"] if task == "generate_promoter" else row["prompt_sd"]
    return Framed(row["locus_tag"], prompt, GEN_LEN[task], 0, "none")


def _prok(row, task: str) -> Framed:
    f = _passthrough(row, task)
    f.prompt = PROK_PREFIX + f.prompt
    f.note = f"prefix={PROK_PREFIX!r}"
    return f
    # NB: the prefix shifts token positions but NOT the generated continuation,
    # which is what every metric is measured against. offset_shift stays 0.


def _promogen2(row, task: str) -> Framed:
    """Reframe to the last 160 bp of the upstream; drop genes that lose their -35."""
    L = int(row["L"])
    start = max(0, L - PROMOGEN2_WINDOW)          # where the native window opens

    if task == "generate_promoter":
        cut = int(row["cut_promoter"])            # = m35_end
        if cut <= start:
            return Framed(row["locus_tag"], "", 0, 0, "",
                          dropped=True,
                          drop_reason=f"-35 ends at {cut}, before the {PROMOGEN2_WINDOW}bp "
                                      f"window opens at {start}")
        prompt = row["upstream_seq"][start:cut]
    else:
        cut = L - 20
        prompt = row["upstream_seq"][start:cut]

    return Framed(row["locus_tag"], prompt, GEN_LEN[task], 0,
                  f"reframed to last {PROMOGEN2_WINDOW}bp (dropped {start} bp from the 5' end)")


# model key -> framing function. Anything absent is passed through unchanged.
FRAMERS = {
    "prok": _prok,
    "promogen2": _promogen2,
}


def frame(row, model: str, task: str) -> Framed:
    """Frame one row of a prompts_681_*.csv for one model and task."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return FRAMERS.get(model, _passthrough)(row, task)


# =====================================================================
# Self-check: no models required.
# =====================================================================
if __name__ == "__main__":
    import csv
    import sys
    from collections import Counter

    base = sys.argv[1] if len(sys.argv) > 1 else "."
    MODELS = ["evo1", "evo2", "prok", "hyenadna", "promogen2"]

    for bg in BACKGROUNDS:
        rows = list(csv.DictReader(open(f"{base}/prompts_681_{bg}.csv")))
        print(f"\n=== {bg}: {len(rows)} genes ===")
        for task in TASKS:
            print(f"\n  {task}  (generate {GEN_LEN[task]} bp)")
            for m in MODELS:
                framed = [frame(r, m, task) for r in rows]
                kept = [f for f in framed if not f.dropped]
                lens = [len(f.prompt) for f in kept]
                notes = Counter(f.note.split(" (")[0] for f in kept)
                drop = len(framed) - len(kept)
                print(f"    {m:10s} kept {len(kept):3d}  dropped {drop:3d}  "
                      f"prompt {min(lens):3d}-{max(lens):3d} bp (med {sorted(lens)[len(lens)//2]:3d})"
                      f"   {'; '.join(notes)}")

                # every kept prompt must be a real prefix of the upstream
                for f, r in zip(framed, rows):
                    if f.dropped:
                        continue
                    core = f.prompt[len(PROK_PREFIX):] if m == "prok" else f.prompt
                    assert core and core in r["upstream_seq"], (m, task, f.locus_tag)

                if m == "promogen2":
                    over = [f for f in kept if len(f.prompt) > PROMOGEN2_WINDOW]
                    assert not over, f"{len(over)} prompts exceed the native window"
                    if drop:
                        print(f"               dropped: "
                              f"{', '.join(f.locus_tag for f in framed if f.dropped)[:110]}...")

    print("\nall framing checks passed")
