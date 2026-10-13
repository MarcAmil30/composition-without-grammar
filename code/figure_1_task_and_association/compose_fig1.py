"""
Figure 1 of the ICBINB-BIO paper: motivation + the three benchmark tasks, 2 x 2.

(a) upstream similarity vs gene relatedness   <- figures/fig1_panel_a.pdf
(b) upstream-only completion                  |
(c) CDS-conditioned generation                |  cropped out of the Affinity
(d) recognition                               |  strip figures/ICBINB_..._FIGS_1.pdf

Everything stays vector. The strip's baked-in "a b c" panel letters are painted
over and redrawn as "b c d" so the lettering runs across the whole figure.

Each of (b)-(d) is a CLIPPED view of the same source strip, so the raw composite
embeds the whole strip three times behind clipping masks. Conformant viewers
honour the clips, but a vector editor shows three stacked copies of the full
strip -- a reviewer of the submitted version read this as duplicated panel
layers. The final step therefore flattens the page with pdftocairo, which keeps
the vectors and the text layer but discards everything outside the clips.

Output: figures/Fig1_setup.pdf  (page width = the paper's 5.5in \linewidth, so
LaTeX places it 1:1 with \includegraphics[width=\linewidth]).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import pymupdf

PANEL_A = "figures/fig1_panel_a.pdf"
STRIP   = "figures/ICBINB_BIO_NEURIPS2026_FIGS_1.pdf"
OUT     = "figures/Fig1_setup.pdf"

W       = 396.0                      # 5.5 in
MARGIN  = 5.0
COLGAP  = 9.0
ROWGAP  = 11.0
LETTER  = 9.0                        # pt, Helvetica-Bold
INK     = (0.102, 0.102, 0.102)
BOLD_FONT = str(Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")

# ink boxes of the three strip panels, and their baked-in letter boxes
PAD = 1.0                            # breathing room around the ink boxes
# ink boxes already include half the stroke width, so nothing gets shaved off
STRIP_PANELS = [                     # (ink rect, letter rect)
    ((2.62, 4.02, 158.61, 126.79), (2.6, 3.4, 11.0, 13.2)),     # upstream-only
    # bottom cropped at 84 pt: the third prompt row ("other gene") is dropped,
    # since the paper's Fig. 3c now contrasts only the native and shuffled CDS
    ((170.87, 4.02, 271.13, 84.00), (170.4, 3.4, 178.8, 13.2)),  # CDS-conditioned
    ((283.19, 4.02, 389.90, 138.55), (282.7, 3.4, 290.6, 13.2)),# recognition
]
STRIP_PANELS = [((r[0] - PAD, r[1] - PAD, r[2] + PAD, r[3] + PAD), l)
                for r, l in STRIP_PANELS]
S_BOT   = 1.28                       # bottom-row crops

a_doc, s_doc = pymupdf.open(PANEL_A), pymupdf.open(STRIP)
a_w, a_h = a_doc[0].rect.width, a_doc[0].rect.height

# row 1: panel (a) at its native size, the strip crop takes the rest of the width
colw_b = W - 2 * MARGIN - COLGAP - a_w
S_TOP  = colw_b / (STRIP_PANELS[0][0][2] - STRIP_PANELS[0][0][0])
colw   = (W - 2 * MARGIN - COLGAP) / 2
row1_h = max(a_h, (STRIP_PANELS[0][0][3] - STRIP_PANELS[0][0][1]) * S_TOP)
row2_h = max((r[3] - r[1]) * S_BOT for r, _ in STRIP_PANELS[1:])
H = MARGIN + row1_h + ROWGAP + row2_h + MARGIN

doc = pymupdf.open()
page = doc.new_page(width=W, height=H)
page.insert_font(fontname="panelbold", fontfile=BOLD_FONT)

x_left, x_right = MARGIN, MARGIN + colw + COLGAP
x_b = MARGIN + a_w + COLGAP
y_top, y_bot = MARGIN, MARGIN + row1_h + ROWGAP
cells = [(x_left, y_top), (x_b, y_top), (x_left, y_bot), (x_right, y_bot)]

# (a) the scatter panel, placed 1:1
page.show_pdf_page(pymupdf.Rect(x_left, y_top, x_left + a_w, y_top + a_h), a_doc, 0)

# (b)-(d) crops of the strip
for (ink, letter), (cx, cy), s in zip(STRIP_PANELS, cells[1:], (S_TOP, S_BOT, S_BOT)):
    clip = pymupdf.Rect(*ink)
    rect = pymupdf.Rect(cx, cy, cx + clip.width * s, cy + clip.height * s)
    page.show_pdf_page(rect, s_doc, 0, clip=clip)
    lb = pymupdf.Rect(cx + (letter[0] - clip.x0) * s, cy + (letter[1] - clip.y0) * s,
                      cx + (letter[2] - clip.x0) * s, cy + (letter[3] - clip.y0) * s)
    page.draw_rect(lb, color=None, fill=(1, 1, 1))

# panel letters, all four drawn here so they match in size and position
for lab, (cx, cy) in zip("ABCD", cells):
    page.insert_text((cx + 0.5, cy + LETTER), lab, fontname="panelbold",
                     fontsize=LETTER, color=INK)

doc.save(OUT, garbage=4, deflate=True)
print(f"wrote {OUT}  {W:.0f} x {H:.0f} pt")


# ---- flatten: drop the clipped-away copies of the source strip ----------------
# Without this the file carries the whole strip three times under clipping masks,
# which vector editors display as stacked duplicate layers. pdftocairo re-emits
# only what is actually visible; output stays vector and the text stays
# selectable. Rendering is unchanged (checked pixel-for-pixel at 150 dpi).
if shutil.which("pdftocairo") is None:
    print("WARNING: pdftocairo not found -- the figure still carries three "
          "clipped copies of the source strip. Install poppler and re-run.",
          file=sys.stderr)
else:
    tmp = Path(OUT).with_suffix(".flat.pdf")
    subprocess.run(["pdftocairo", "-pdf", OUT, str(tmp)], check=True)
    before, after = Path(OUT).stat().st_size, tmp.stat().st_size
    tmp.replace(OUT)
    n = len(pymupdf.open(OUT)[0].get_xobjects())
    print(f"flattened {OUT}  {before/1e3:.0f} kB -> {after/1e3:.0f} kB, "
          f"{n} embedded xobjects remaining")
