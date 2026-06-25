"""Visualise each detection layer separately for a given page.

Usage:  python training/debug_layers.py page_0003
Saves:
  debug/{page}_L1_projection.jpg   — vertical projection profile + major groups
  debug/{page}_L1_groups.jpg       — major groups overlaid on page
  debug/{page}_L2_hsep.jpg         — horizontal separators overlaid on page
  debug/{page}_L2_sections.jpg     — sections coloured on page
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

from src.vertical_ocr.layout import (
    _binarise, _vertical_projection, _smooth,
    _find_valleys, _valleys_to_spans,
    detect_major_groups, detect_horizontal_separators, _separators_to_sections,
)

stem = sys.argv[1] if len(sys.argv) > 1 else "page_0003"
img = Image.open(f"training/images/all/{stem}.jpg").convert("RGB")
W, H = img.size
gray = np.array(img.convert("L"))
binary = _binarise(gray)
bh, bw = binary.shape
MARGIN = int(bh * 0.01)
binary_crop = binary[MARGIN:bh - MARGIN, :]

COLOURS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]

# ── Layer 1: vertical projection ──────────────────────────────────────────────
profile = _smooth(_vertical_projection(binary_crop), sigma=10)
PROF_H = 120
prof_img = Image.new("RGB", (bw, PROF_H), (240, 240, 240))
d = ImageDraw.Draw(prof_img)
mx = profile.max() or 1
for x, v in enumerate(profile):
    bar = int(v / mx * (PROF_H - 4))
    d.line([(x, PROF_H), (x, PROF_H - bar)], fill="#457b9d")

groups = detect_major_groups(binary_crop, wide_valley_frac=0.005)
for x1, x2 in groups:
    d.rectangle([x1, 0, x2, PROF_H - 1], outline="#e63946", width=2)
print(f"L1 groups: {groups}")
prof_img.save(f"training/debug/{stem}_L1_projection.jpg", quality=90)

# Overlay groups on page
g_img = img.copy()
gd = ImageDraw.Draw(g_img)
for i, (x1, x2) in enumerate(groups):
    c = COLOURS[i % len(COLOURS)]
    gd.rectangle([x1, 0, x2, H - 1], outline=c, width=4)
    gd.text((x1 + 4, 4), f"G{i}", fill=c)
g_img.save(f"training/debug/{stem}_L1_groups.jpg", quality=88)
print(f"Saved L1 group images.")

# ── Layer 2: horizontal separators ────────────────────────────────────────────
h_seps = detect_horizontal_separators(binary_crop, min_width_frac=0.30)
sections = _separators_to_sections(h_seps, bh, edge_margin_frac=0.05)
print(f"L2 separators: {h_seps}")
print(f"L2 sections: {sections}")

sep_img = img.copy()
sd = ImageDraw.Draw(sep_img)
for y in h_seps:
    real_y = y + MARGIN
    sd.line([(0, real_y), (W - 1, real_y)], fill="#e63946", width=3)
sep_img.save(f"training/debug/{stem}_L2_hsep.jpg", quality=88)

sec_img = img.copy().convert("RGBA")
overlay = Image.new("RGBA", sec_img.size, (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
sec_colours = [(70, 130, 180, 40), (42, 157, 143, 40), (233, 196, 106, 40)]
for i, (y1, y2) in enumerate(sections):
    real_y1, real_y2 = y1 + MARGIN, y2 + MARGIN
    c = sec_colours[i % len(sec_colours)]
    od.rectangle([0, real_y1, W - 1, real_y2], fill=c)
sec_img = Image.alpha_composite(sec_img, overlay).convert("RGB")
ImageDraw.Draw(sec_img)
sec_img.save(f"training/debug/{stem}_L2_sections.jpg", quality=88)
print(f"Saved L2 separator/section images.")
