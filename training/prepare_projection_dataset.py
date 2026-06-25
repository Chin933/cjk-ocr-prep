"""Build a non-destructive YOLO dataset from the projection layout detector.

This reads training/images/all/*.jpg, runs src.vertical_ocr.layout.detect_columns,
and writes a fresh train/val dataset under training/generated_projection/.
Existing hand or auto labels in training/labels/all are left untouched.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vertical_ocr.layout import _binarise, detect_columns, draw_columns


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
IMG_ALL = ROOT / "images" / "all"
OUT = ROOT / "generated_projection"
DEBUG = OUT / "debug"


def yolo_line(box, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    bw = box.width / width
    bh = box.height / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def is_text_like(box, binary: np.ndarray, page_height: int) -> bool:
    if box.width < 18 or box.height < page_height * 0.04:
        return False
    if box.height < box.width * 1.5:
        return False

    strip = binary[box.y1:box.y2, box.x1:box.x2]
    density = strip.sum() / (strip.size * 255 + 1)
    return density >= 0.13


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    images = sorted(IMG_ALL.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {IMG_ALL}")

    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_names = {p.name for p in shuffled[:n_val]}

    total_boxes = 0
    split_counts = {"train": 0, "val": 0}
    for img_path in images:
        split = "val" if img_path.name in val_names else "train"
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        binary = _binarise(np.array(image.convert("L")))
        columns = detect_columns(image, dpi=args.dpi)
        columns = [col for col in columns if is_text_like(col, binary, height)]

        label_lines = [yolo_line(col, width, height) for col in columns]
        total_boxes += len(label_lines)
        split_counts[split] += 1

        shutil.copy2(img_path, OUT / "images" / split / img_path.name)
        (OUT / "labels" / split / f"{img_path.stem}.txt").write_text(
            "\n".join(label_lines),
            encoding="utf-8",
        )
        draw_columns(image, columns).save(DEBUG / img_path.name, quality=85)
        print(f"{img_path.name}: {len(label_lines)} boxes -> {split}")

    dataset_yaml = OUT / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {OUT.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names: [text_column]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        f"\nWrote {dataset_yaml}\n"
        f"Images: {split_counts['train']} train / {split_counts['val']} val\n"
        f"Boxes: {total_boxes}\n"
        f"Debug previews: {DEBUG}"
    )


if __name__ == "__main__":
    main()
