"""Draw manually reviewed glyph-record truth without detector predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "training/glyph_graph_regressions.json",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    page = next(item for item in spec["records"] if item["image"] == args.image.name)
    with Image.open(args.image) as source:
        output = source.convert("RGB")
    draw = ImageDraw.Draw(output)

    for index, box in enumerate(page.get("gold_records", [])):
        draw.rectangle(box, outline="#6a00f4", width=4)
        draw.rectangle((box[0], box[1], box[0] + 28, box[1] + 14), fill="white")
        draw.text((box[0] + 1, box[1] + 1), f"U{index}", fill="#6a00f4")

    primaries = page.get("lower_gold_primaries", [])
    for index, box in enumerate(primaries):
        draw.rectangle(box, outline="#6a00f4", width=4)
        label_y = max(0, box[1] - 14)
        draw.rectangle((box[0], label_y, box[0] + 30, label_y + 14), fill="white")
        draw.text((box[0] + 1, label_y + 1), f"L{index}", fill="#6a00f4")

    for check in page.get("lower_annotation_checks", []):
        x, y = check["point"]
        box = primaries[check["primary"]]
        target_x = min(max(x, box[0]), box[2])
        target_y = min(max(y, box[1]), box[3])
        draw.line((x, y, target_x, target_y), fill="#f77f00", width=3)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#f77f00")

    for source_index, target_index in page.get("lower_next_records", []):
        source = primaries[source_index]
        target = primaries[target_index]
        draw.line(
            (
                (source[0] + source[2]) / 2,
                source[3],
                (target[0] + target[2]) / 2,
                target[1],
            ),
            fill="#008f5a",
            width=5,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output, quality=92)


if __name__ == "__main__":
    main()
