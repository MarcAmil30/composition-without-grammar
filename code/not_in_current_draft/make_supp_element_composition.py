#!/usr/bin/env python3
"""
Supplementary figure: how much of an upstream is regulatory element, and what
those elements are made of.

Two claims in Section 6 rest on this. First, the identifiable elements occupy a
small minority of the window, so a per-position loss spends most of its capacity
on unannotated spacer. Second, the elements are AT-rich, which is why a model can
reproduce the correct average composition of the region without placing anything:
matching the composition of a promoter does not require knowing where the promoter
is.

Panels:
  A  pie -- share of the 681 x 200 bp of benchmark sequence occupied by each
     annotated element class, against unannotated spacer. Overlapping
     annotations are assigned once, in the priority order used for the clean
     control (boxes, then SD, then TFBS).
  B  stacked A/C/G/T composition of each element class and of the spacer, with
     the AT fraction printed above each bar.

Source: data/prompts_681_leaked.csv (the natural benchmark sequences and their
RegulonDB annotations).

Output: figures/supp_element_composition.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "prompts_681_leaked.csv"
OUT_STEMS = [ROOT / "results" / "supp_element_composition",
             ROOT / "figures" / "supp_element_composition"]

BASE_C = {"A": "#1B7837", "C": "#2166AC", "G": "#E6A700", "T": "#B2182B"}
INK, GREY = "#1a1a1a", "#8C8C8C"
# assignment priority: a base annotated twice counts once, for the class listed first
CLASSES = [("distal box", "#2166AC"),
           ("proximal box", "#E6A700"),
           ("Shine--Dalgarno", "#1B7837"),
           ("TF binding site", "#B2182B"),
           ("spacer", "#D9D9D9")]
LABEL = {"Shine--Dalgarno": "Shine–Dalgarno"}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "axes.linewidth": 1.6, "axes.edgecolor": INK,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
    })


def spans(row):
    """Every annotated interval in one upstream, in assignment priority order."""
    out = [("distal box", int(row.m35_start), int(row.m35_end)),
           ("proximal box", int(row.m10_start), int(row.m10_end))]
    if row.sd_present == 1 and not pd.isna(row.sd_start):
        out.append(("Shine--Dalgarno", int(row.sd_start), int(row.sd_end)))
    if isinstance(row.tfbs_spans, str) and row.tfbs_spans.strip():
        for part in row.tfbs_spans.split(";"):
            if "-" not in part:
                continue
            a, b = part.split("-")
            out.append(("TF binding site", int(a), int(b)))
    return out


def main() -> None:
    set_style()
    df = pd.read_csv(SRC)

    names = [c for c, _ in CLASSES]
    n_bases = {c: 0 for c in names}
    comp = {c: {b: 0 for b in "ACGT"} for c in names}

    for row in df.itertuples(index=False):
        seq = row.upstream_seq
        owner = np.full(len(seq), -1, np.int8)      # -1 = spacer
        for cls, a, b in spans(row):
            a, b = max(a, 0), min(b, len(seq))
            k = names.index(cls)
            sl = owner[a:b]
            sl[sl < 0] = k                          # first class to claim a base keeps it
        owner[owner < 0] = names.index("spacer")
        arr = np.frombuffer(seq.encode(), np.uint8)
        for k, cls in enumerate(names):
            m = owner == k
            n_bases[cls] += int(m.sum())
            for b in "ACGT":
                comp[cls][b] += int((arr[m] == ord(b)).sum())

    total = sum(n_bases.values())
    print(f"{len(df)} upstreams, {total:,} bases")
    for c in names:
        at = (comp[c]["A"] + comp[c]["T"]) / max(sum(comp[c].values()), 1)
        print(f"  {c:16s} {n_bases[c]:>7,} bases  {100*n_bases[c]/total:5.1f}%   AT {100*at:.1f}%")

    fig, (axp, axb) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # ---- A: how much of the window is annotated at all
    vals = [n_bases[c] for c, _ in CLASSES]
    cols = [c for _, c in CLASSES]
    frac_elem = 100 * (total - n_bases["spacer"]) / total
    wedges, _, txt = axp.pie(
        vals, colors=cols, startangle=90, counterclock=False,
        autopct=lambda v: f"{v:.0f}%" if v >= 4 else "",
        pctdistance=0.74, textprops=dict(fontsize=8.5, fontweight="bold"),
        wedgeprops=dict(edgecolor="white", linewidth=1.0))
    for t in txt:
        t.set_color("white")
    axp.legend(wedges, [LABEL.get(c, c) for c, _ in CLASSES], loc="center left",
               bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=8.5)
    axp.set_title(f"{frac_elem:.0f}% of the window is an annotated element",
                  fontsize=9.5, fontweight="bold", pad=8)

    # ---- B: what those bases are made of
    xs = np.arange(len(names))
    bottom = np.zeros(len(names))
    for b in "ACGT":
        h = np.array([100 * comp[c][b] / max(sum(comp[c].values()), 1) for c in names])
        axb.bar(xs, h, 0.62, bottom=bottom, color=BASE_C[b], edgecolor="white",
                linewidth=0.6, label=b)
        for x, (y0, y) in enumerate(zip(bottom, h)):
            if y >= 9:
                axb.text(x, y0 + y / 2, b, ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bottom += h
    for x, c in enumerate(names):
        at = 100 * (comp[c]["A"] + comp[c]["T"]) / max(sum(comp[c].values()), 1)
        axb.text(x, 101.5, f"AT {at:.0f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
    axb.set_xticks(xs)
    axb.set_xticklabels([LABEL.get(c, c).replace(" ", "\n", 1) for c in names],
                        fontsize=8.5)
    axb.tick_params(axis="x", length=0)
    axb.set_ylim(0, 108)
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_ylabel("base composition (%)")
    axb.set_title("The elements are AT-rich, like the spacer around them",
                  fontsize=9.5, fontweight="bold", pad=14)

    fig.tight_layout(w_pad=2.0)
    for stem in OUT_STEMS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + ext))
        print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
