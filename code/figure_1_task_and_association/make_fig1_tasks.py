"""Figure 1: schematic of the three tasks (generation x2, recognition).

Redraw of ICBINB_BIO_NEURIPS2026_FIGS_1.pdf: same three-panel layout, cleaner
line work and a restricted palette. Vector PDF, drawn at final size (5.5 in =
\linewidth) so point sizes match the manuscript text.

    python code/make_fig1_tasks.py
"""

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

INK      = "#1A1A1A"   # text, prompt segments
DARK     = "#4A5158"   # model-output dashes where colour is not needed
GREY     = "#868D95"   # DNA backbone, secondary text
GREY_L   = "#D5D9DD"   # box strokes, dividers
GREY_F   = "#F2F4F6"   # neutral fills
ORANGE   = "#D2762F"   # task i  (proximal box + TSS)
BLUE     = "#2E6E9E"   # task ii (ribosome site + start codon)
BLUE_F   = "#DEEAF3"
GREEN    = "#3F8F63"   # the true CDS-upstream pair
GREEN_F  = "#DFEDE5"

FS_PANEL, FS_TITLE, FS_LAB, FS_NOTE = 8.5, 6.9, 6.0, 5.3
LW = 0.9
DASH = (0, (2.4, 1.8))

W, H = 5.5, 2.45
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


# --------------------------------------------------------------- helpers ---
def text(x, y, s, size=FS_LAB, color=INK, ha="left", va="center", weight=None):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, linespacing=1.25)


def box(x0, x1, y, h, fc="white", ec=GREY_L, lw=LW, label=None,
        label_color=INK, size=FS_LAB, r=0.02, z=3):
    ax.add_patch(FancyBboxPatch(
        (x0, y - h / 2), x1 - x0, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    if label:
        text((x0 + x1) / 2, y, label, size=size, color=label_color, ha="center")


def seg(x0, x1, y, color, dashed=False, lw=1.4, cap=True):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=4,
            solid_capstyle="butt", linestyle=DASH if dashed else "-")
    if cap:  # end ticks make the span boundaries unambiguous
        for x in (x0, x1):
            ax.plot([x, x], [y - 0.032, y + 0.032], color=color, lw=lw, zorder=4)


def arrow(x0, y0, x1, y1, color=GREY, lw=0.8, hw=1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=5,
        arrowstyle=f"-|>,head_width={hw},head_length={hw * 1.6}",
        mutation_scale=1.5, joinstyle="miter"))


def panel(letter, title, x, y=None):
    y = H - 0.13 if y is None else y
    text(x, y, letter, size=FS_PANEL, weight="bold")
    text(x + 0.15, y - 0.005, title, size=FS_TITLE, weight="bold")


# dividers between panels
for xd in (2.32, 3.88):
    ax.plot([xd, xd], [0.14, H - 0.05], color=GREY_L, lw=0.6, zorder=1)


# ================================================================ panel A ===
panel("a", "Upstream-only completion", 0.05)

xL, xR = 0.13, 2.20
y_map, hb = 1.82, 0.16
D0, D1 = 0.28, 0.60          # distal box
P0, P1 = 0.78, 1.10          # proximal box
xTSS = 1.32                  # transcription start site
S0, S1 = 1.60, 1.86          # ribosome binding (Shine-Dalgarno) site
A0, A1 = 1.94, 2.18          # start codon

ax.plot([xL, xR], [y_map, y_map], color=GREY, lw=1.0, zorder=2)
box(D0, D1, y_map, hb, label="distal", size=FS_NOTE)
box(P0, P1, y_map, hb, label="proximal", size=FS_NOTE)
box(S0, S1, y_map, hb, label="SD", size=FS_NOTE)
box(A0, A1, y_map, hb, fc=GREY_F, label="ATG", size=FS_NOTE)

# TSS: bent arrow off the backbone
ax.plot([xTSS, xTSS], [y_map, y_map + 0.13], color=INK, lw=0.9, zorder=4)
arrow(xTSS, y_map + 0.13, xTSS + 0.09, y_map + 0.13, color=INK, lw=0.9, hw=1.4)

text((D0 + P1) / 2, y_map + 0.20, "promoter boxes", size=FS_NOTE, color=GREY,
     ha="center")
text(xTSS - 0.02, y_map + 0.235, "TSS", size=FS_NOTE, color=GREY, ha="right")
text((S0 + S1) / 2, y_map + 0.20, "ribosome site", size=FS_NOTE, color=GREY,
     ha="center")
text((A0 + A1) / 2, y_map + 0.20, "start", size=FS_NOTE, color=GREY, ha="center")

# the two prompt cut points, carried down to the tracks
y_i, y_ii = 1.32, 0.90
for xc, yb in ((D1, y_i), (S0, y_ii)):
    ax.plot([xc, xc], [yb + 0.05, y_map - 0.13], color=GREY_L, lw=0.6,
            linestyle=(0, (1.2, 1.6)), zorder=1)

text(xL - 0.05, y_i, "i", size=FS_LAB, color=GREY, ha="right")
seg(xL, D1, y_i, INK)
seg(D1, xTSS, y_i, ORANGE, dashed=True)
text((D1 + xTSS) / 2, y_i + 0.14, "proximal box, TSS", size=FS_NOTE,
     color=ORANGE, ha="center")

text(xL - 0.05, y_ii, "ii", size=FS_LAB, color=GREY, ha="right")
seg(xL, S0, y_ii, INK)
seg(S0, A1, y_ii, BLUE, dashed=True)
text(A1, y_ii + 0.14, "ribosome site, start codon", size=FS_NOTE, color=BLUE,
     ha="right")

# legend
y_leg = 0.36
seg(xL, xL + 0.18, y_leg, INK, cap=False)
text(xL + 0.23, y_leg, "prompt", size=FS_NOTE, color=GREY)
seg(xL + 0.60, xL + 0.78, y_leg, DARK, dashed=True, cap=False)
text(xL + 0.83, y_leg, "model output", size=FS_NOTE, color=GREY)


# ================================================================ panel B ===
panel("b", "CDS-conditioned generation", 2.38)

bx0, bx1 = 2.46, 2.92        # CDS box
ux0, ux1 = 3.10, 3.76        # generated upstream
rows = [("native", BLUE_F), ("shuffled", GREY_F), ("other gene", "white")]
y_rows = [1.82, 1.42, 1.02]

text((bx0 + bx1) / 2, 2.09, "CDS prompt", size=FS_NOTE, color=GREY, ha="center")
text((ux0 + ux1) / 2, 2.09, "generated upstream", size=FS_NOTE, color=GREY,
     ha="center")

for (lab, fc), yr in zip(rows, y_rows):
    box(bx0, bx1, yr, 0.20, fc=fc, label=lab, size=FS_NOTE)
    arrow(bx1 + 0.04, yr, ux0 - 0.04, yr)
    seg(ux0, ux1, yr, DARK, dashed=True, cap=False)

text(2.46, 0.62, "one upstream window,\nthree CDS prompts", size=FS_NOTE,
     color=GREY)
text(2.46, 0.30, "compare P(A, C, G, T)\nat every upstream position", size=FS_NOTE,
     color=INK, va="center")


# ================================================================ panel C ===
panel("c", "Recognition", 3.94)

qx0, qx1, yq = 3.99, 4.33, 1.88
box(qx0, qx1, yq, 0.20, fc=BLUE_F, label="CDS", size=FS_NOTE)
text((qx0 + qx1) / 2, yq + 0.19, "query", size=FS_NOTE, color=GREY, ha="center")

# candidate upstreams: one true (green) + 99 decoys (grey)
cx0, cx1 = 4.52, 5.11
y_cand = [2.06, 1.94, 1.82, 1.70]
arrow(qx1 + 0.05, yq, cx0 - 0.08, yq)
for k, yc in enumerate(y_cand):
    is_true = (k == 1)
    ax.add_patch(Rectangle((cx0, yc - 0.036), cx1 - cx0, 0.072, zorder=3,
                           facecolor=GREEN_F if is_true else GREY_F,
                           edgecolor=GREEN if is_true else GREY_L, lw=LW))
text(cx1 + 0.05, y_cand[1], "true", size=FS_NOTE, color=GREEN)
text(cx0 - 0.02, y_cand[-1] - 0.14, "1 true upstream\n+ 99 from other genes",
     size=FS_NOTE, color=GREY, va="top")

# score -> rank
arrow(4.20, 1.30, 4.20, 1.10)
text(4.30, 1.20, "rank by likelihood\nor cosine similarity", size=FS_NOTE,
     color=GREY, va="center")

rx0, rw = 4.13, 0.92
yr0, dy = 0.90, 0.135
for k in range(4):
    yy = yr0 - k * dy
    is_true = (k == 2)
    ax.add_patch(Rectangle((rx0, yy - 0.04), rw * (1.0 - 0.14 * k), 0.08,
                           zorder=3,
                           facecolor=GREEN_F if is_true else GREY_F,
                           edgecolor=GREEN if is_true else GREY_L, lw=LW))
    text(rx0 - 0.04, yy, f"{k + 1}", size=FS_NOTE, color=GREY, ha="right")
for j in range(3):  # vertical ellipsis
    ax.plot([rx0 - 0.075], [yr0 - 4 * dy + 0.035 - j * 0.033], marker="o",
            ms=0.7, color=GREY, zorder=3)
text(rx0 - 0.09, 0.22, "chance = 1 in 100", size=FS_NOTE, color=INK)

out = "figures/ICBINB_BIO_NEURIPS2026_FIGS_1.pdf"
fig.savefig(out)
print("wrote", out)
