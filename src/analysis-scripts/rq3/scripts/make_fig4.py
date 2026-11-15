#!/usr/bin/env python3
"""Fig. 11: distribution of unique compilation errors over the nine categories,
as the share of each translator's errors.

The figure is drawn at its printed size, so no scaling happens in LaTeX:
  --width column   3.5 in, for \\includegraphics[width=\\columnwidth]   (default)
  --width page     7.16 in, for a figure* with \\includegraphics[width=\\textwidth]
Fonts are embedded as TrueType (Type 42), not Type 3, and use a Times-like
serif to match the IEEE text font. The PDF is vector; the PNG is a 600-dpi
preview.

Input:  data/compile_error_taxonomy.tsv
Output: results/rq3_error_distribution.pdf and results/rq3_error_distribution.png

Usage (from the rq3/ directory):
  python3 scripts/make_fig4.py [--width column|page] [--dpi 600] [--out results/rq3_error_distribution]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
TR = {"CL": "FT-CodeLlama-13B", "SC": "FT-StarCoderBase-7B", "HC": "FT-HPC-Coder-6.7B"}
COLORS = {"CL": "#2a78d6", "SC": "#eb6834", "HC": "#1baf7a"}
# Categories that arise from the gap between CUDA's and OpenMP's execution models
# (marked with a dagger, as in Fig. 10).
MODEL_SPECIFIC = {"Incorrect loop construct", "Unsupported construct usage", "Invalid construct usage"}

ap = argparse.ArgumentParser()
ap.add_argument("--width", choices=["column", "page"], default="column")
ap.add_argument("--dpi", type=int, default=600, help="resolution of the PNG preview")
ap.add_argument("--out", default=str(ROOT / "results" / "rq3_error_distribution"), help="output path without extension")
args = ap.parse_args()

# ------------------------------------------------------------------ data
rows = list(csv.reader(open(ROOT / "data" / "compile_error_taxonomy.tsv"), delimiter="\t"))
header = rows[0]
cats = []
for r in rows[1:]:
    if not r or r[0].strip().lower() == "total":
        continue
    cats.append(dict(name=r[0].strip(), counts={k: int(r[header.index(k)]) for k in TR},
                     total=int(r[header.index("Total")])))
cats.sort(key=lambda c: -c["total"])                  # same order as Fig. 10
T = {k: sum(c["counts"][k] for c in cats) for k in TR}

# ------------------------------------------------------------------ style
# Times-compatible TrueType font (Times New Roman or its metric clone Liberation
# Serif), embedded as Type 42. Install one with, e.g., apt install fonts-liberation.
avail = {f.name for f in font_manager.fontManager.ttflist if f.fname.lower().endswith(".ttf")}
serif = [n for n in ("Times New Roman", "Liberation Serif") if n in avail] or ["DejaVu Serif"]

if args.width == "column":
    W, H, fs, fs_small, title_fs, xmax = 3.5, 2.45, 6.5, 5.5, 5.8, 72
else:
    W, H, fs, fs_small, title_fs, xmax = 7.16, 2.6, 8, 7, 8, 75

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
    "font.size": fs, "axes.edgecolor": "#52514e", "axes.linewidth": 0.5,
    "xtick.color": "#52514e", "ytick.color": "#0b0b0b", "xtick.labelsize": fs_small, "ytick.labelsize": fs,
    "xtick.major.pad": 1.5, "ytick.major.pad": 2,
})

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
labels = [c["name"] + (" †" if c["name"] in MODEL_SPECIFIC else "") for c in cats]
y = list(range(len(cats)))[::-1]
for ax, k in zip(axes, TR):
    vals = [100 * c["counts"][k] / T[k] for c in cats]
    ax.barh(y, vals, height=0.66, color=COLORS[k], linewidth=0)
    for yi, v, c in zip(y, vals, cats):
        txt = "<1%" if 0 < v < 0.5 else f"{v:.0f}%"
        if args.width == "page":
            txt += f" ({c['counts'][k]:,})"
        ax.text(v + 1.2, yi, txt, va="center", ha="left", fontsize=fs_small, color="#52514e")
    ax.set_title(f"{TR[k]}\n{T[k]:,} errors", fontsize=title_fs, pad=3, linespacing=1.15)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 25, 50] if xmax < 75 else [0, 25, 50, 75])
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels)
axes[0].set_ylim(-0.6, len(cats) - 0.4)
fig.supxlabel("Share of the translator's errors (%)", fontsize=fs, y=0.0)
fig.tight_layout(pad=0.2, w_pad=0.9)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {out.with_suffix('.pdf')} ({W} in wide, font {', '.join(serif[:1])}) and {out.with_suffix('.png')} ({args.dpi} dpi)")
