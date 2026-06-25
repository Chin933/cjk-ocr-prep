"""Visualise the raw vertical line detection on a specific page.

Shows:
  - Left:  original page
  - Middle: binary vertical-line mask (what the detector sees)
  - Right:  detected boundary positions overlaid on the original

Usage:  python training/debug_lines.py page_0006
"""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

from src.vertical_ocr.layout import _to_binary, _vertical_line_mask, \
    _line_density_profile, _find_boundary_peaks

stem = sys.argv[1] if len(sys.argv) > 1 else "page_0006"
img_path = Path(f"training/images/all/{stem}.jpg")
img = Image.open(img_path).convert("RGB")
W, H = img.size

gray = np.array(img.convert("L"))
binary = _to_binary(gray)
line_mask = _vertical_line_mask(binary, min_segment_height_frac=0.08)
profile = _line_density_profile(line_mask)
peaks = _find_boundary_peaks(profile, min_prominence_frac=0.10)

print(f"{stem}: image={W}x{H},  detected {len(peaks)} boundary peaks")
print(f"  Peak x-positions: {peaks}")

# Visualise peaks on the original image
overlay = img.copy()
draw = ImageDraw.Draw(overlay)
for x in peaks:
    draw.line([(x, 0), (x, H)], fill="red", width=3)
overlay.save(f"training/debug/{stem}_lines.jpg", quality=88)
print(f"  Saved: training/debug/{stem}_lines.jpg")

# Save the line mask itself
mask_img = Image.fromarray(line_mask)
mask_img.save(f"training/debug/{stem}_mask.jpg")
print(f"  Saved: training/debug/{stem}_mask.jpg  (white = detected line segments)")

# Plot the projection profile as a simple bar chart image
scale = 200
prof_h = int(profile.max()) if profile.max() > 0 else 1
prof_img = Image.new("L", (W, scale), 255)
d = ImageDraw.Draw(prof_img)
for x, v in enumerate(profile):
    bar_h = int(v / prof_h * (scale - 2))
    d.line([(x, scale), (x, scale - bar_h)], fill=0)
for x in peaks:
    d.line([(x, 0), (x, scale)], fill=128)
prof_img.save(f"training/debug/{stem}_profile.jpg")
print(f"  Saved: training/debug/{stem}_profile.jpg  (vertical projection profile)")
