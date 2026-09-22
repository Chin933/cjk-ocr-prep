"""Auto-annotate training images using our ruling-line column detector.

Runs the projection-profile + grid-fitting detector on every image in
training/images/all/ and writes YOLO-format label files to
training/labels/all/.

Also saves colour-coded debug images to training/debug/ so you can
visually spot-check quality before training.

YOLO label format (one line per box):
    <class_id> <cx> <cy> <w> <h>   (all values normalised 0-1)

Classes
-------
    0  text_column   — individual vertical text column (main output)
    1  page_frame    — outer printed border of the page
"""

import sys
from pathlib import Path

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vertical_ocr.layout import detect_columns, draw_columns
from PIL import Image

IMG_DIR   = Path("training/images/all")
LBL_DIR   = Path("training/labels/all")
DEBUG_DIR = Path("training/debug")

LBL_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLUMN = 0

images = sorted(IMG_DIR.glob("*.jpg"))
if not images:
    print(f"No images found in {IMG_DIR}/")
    sys.exit(1)

print(f"Auto-annotating {len(images)} images...\n")

total_boxes = 0
for img_path in images:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    columns = detect_columns(img, dpi=200)

    # Write YOLO label file
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    lines = []
    for col in columns:
        # Skip columns that are suspiciously thin or tall (likely noise)
        if col.width < W * 0.01 or col.height < H * 0.05:
            continue
        cx = (col.x1 + col.x2) / 2 / W
        cy = (col.y1 + col.y2) / 2 / H
        cw = col.width  / W
        ch = col.height / H
        lines.append(f"{CLASS_COLUMN} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")

    lbl_path.write_text("\n".join(lines))
    n = len(lines)
    total_boxes += n

    # Save debug image with boxes drawn
    debug_img = draw_columns(img, columns)
    debug_path = DEBUG_DIR / img_path.name
    debug_img.save(str(debug_path), quality=85)

    conf_boxes  = sum(1 for c in columns if c.confidence == 1.0)
    infer_boxes = sum(1 for c in columns if c.confidence < 1.0)
    print(f"  {img_path.name}: {n} boxes  "
          f"({conf_boxes} line-detected, {infer_boxes} inferred)")

print(f"\nDone. {total_boxes} boxes across {len(images)} images.")
print(f"Labels  → {LBL_DIR}/")
print(f"Debug   → {DEBUG_DIR}/   ← open these to spot-check quality")
print()
print("Next steps:")
print("  1. Open 5 debug images and check box alignment")
print("  2. Fix obvious errors: labelimg training/images/all/ "
      "training/labels/all/")
print("  3. Run: python training/split_dataset.py")
print("  4. Run: modal run training/train_modal.py")
