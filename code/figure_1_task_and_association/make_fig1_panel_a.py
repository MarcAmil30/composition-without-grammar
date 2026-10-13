"""
Panel (a) of Figure 1: upstream similarity vs gene relatedness.

Rebuilds the standalone figure `figures/fig_grammar_singleton_pairs.pdf` at
panel size. The point cloud is lifted straight out of that PDF as a high-res
raster (it was rasterised there too), the running-median curves are recovered
as exact vector paths, and every piece of text is redrawn at panel-appropriate
size. The source parquet lives on CREATE, so nothing here needs the cluster.

Output: figures/fig1_panel_a.pdf   (186 x 155 pt, drawn 1:1 for the composite)
"""
import numpy as np, pymupdf, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC   = "figures/fig_grammar_singleton_pairs.pdf"
STATS = "figures/fig_grammar_singleton_pairs_stats.csv"
OUT   = "figures/fig1_panel_a.pdf"

W_PT, H_PT = 202.0, 141.0          # panel size in the composite
INK, GREY  = "#1A1A1A", "#8C8C8C"
RANKS = [("same genus", "#6A3D9A"), ("same family", "#D1442F"),
         ("same order", "#E8A33D"), ("same class", "#3FA7C4")]

# --- source geometry (measured from the PDF, see docstring) -------------------
AX = (71.9, 55.8, 518.3, 388.44)                 # axes rect, PDF pts, y down
XL, YL = (25.0, 100.0), (-0.05, 1.0)
sx = lambda x: AX[0] + (x - XL[0]) * (AX[2] - AX[0]) / (XL[1] - XL[0])
sy = lambda y: 372.6 - y * 316.8                  # y=0 -> 372.6, y=1 -> 55.8
ix = lambda p: XL[0] + (p - AX[0]) * (XL[1] - XL[0]) / (AX[2] - AX[0])
iy = lambda p: (372.6 - p) / 316.8

doc  = pymupdf.open(SRC)
page = doc[0]

# --- 1. point cloud: render the axes interior, blank out legend + stats box ---
PAD  = 1.6                                        # step inside the spines
clip = pymupdf.Rect(AX[0] + PAD, AX[1] + PAD, AX[2] - PAD, AX[3] - PAD)
for r in [pymupdf.Rect(75, 55, 266, 149), pymupdf.Rect(77, 159, 360, 214)]:
    page.draw_rect(r, color=None, fill=(1, 1, 1), overlay=True)
pix = page.get_pixmap(clip=clip, dpi=600, alpha=False)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
extent = (ix(clip.x0), ix(clip.x1), iy(clip.y1), iy(clip.y0))

# --- 2. running medians: exact vector paths out of the same page --------------
def path_xy(width):
    g = [d for d in page.get_drawings()
         if abs((d.get("width") or 0) - width) < .01 and len(d["items"]) > 5][0]
    pts = [it[1] for it in g["items"]] + [g["items"][-1][2]]
    return np.array([ix(p.x) for p in pts]), np.array([iy(p.y) for p in pts])
mx, my   = path_xy(3.4)          # orthologue
cx, cy   = path_xy(2.8)          # control

st = pd.read_csv(STATS).set_index("series")

# --- 3. redraw ----------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "axes.linewidth": 1.1, "axes.edgecolor": INK,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "text.color": INK, "axes.labelsize": 6.4, "axes.labelweight": "normal",
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.3,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True,
})
fig = plt.figure(figsize=(W_PT / 72, H_PT / 72))
ax  = fig.add_axes([0.125, 0.245, 0.830, 0.655])

ax.imshow(img, extent=extent, aspect="auto", origin="upper",
          interpolation="antialiased", zorder=1)
ax.plot(mx, my, lw=1.7, color=INK, zorder=7, solid_capstyle="round")
ax.plot(cx, cy, lw=1.5, color=GREY, ls=(0, (5, 3)), zorder=7)

f = lambda s, c: (f"{st.loc[s, c]:+.2f}" if c == "spearman_rho"
                  else f"{st.loc[s, c]:.2f}")
ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xticks([30, 40, 50, 60, 70, 80, 90, 100])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Gene relatedness\n(identity at third codon positions, %)", labelpad=1.0)
ax.set_ylabel("Upstream similarity (D2*)", labelpad=2.0)
taxa = ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for l, c in RANKS],
                 loc="upper left", bbox_to_anchor=(-0.012, 1.025), ncol=2,
                 handlelength=1.0, handleheight=0.7, labelspacing=0.22,
                 columnspacing=0.9, handletextpad=0.4, borderpad=0.0)
ax.add_artist(taxa)
ax.legend(handles=[
    Line2D([0], [0], color=INK, lw=1.7,
           label=f"orthologue   $\\rho$ = {f('orthologue','spearman_rho')},  "
                 f"$r^2$ = {f('orthologue','r_squared')}"),
    Line2D([0], [0], color=GREY, lw=1.5, ls=(0, (5, 3)),
           label=f"control          $\\rho$ = {f('control','spearman_rho')},  "
                 f"$r^2$ = {f('control','r_squared')}")],
    loc="upper left", bbox_to_anchor=(-0.012, 0.845), handlelength=1.6,
    labelspacing=0.24, handletextpad=0.5, borderpad=0.0)
ax.set_title("Related genes have similar upstreams", fontsize=6.9,
             pad=3.5)
fig.savefig(OUT)
print("wrote", OUT, "raster", img.shape)
