"""Precompute typed example results for the Modal random-example endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from digitalization.modal_pipeline import analyze_image, load_segmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training/layout_review/page_types_20260918.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".modal-examples")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_segmenter(args.checkpoint)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    index = {"examples": []}
    for entry in manifest["pages"]:
        stem = entry["image"]
        candidates = [path for path in args.images.glob(f"{stem}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not candidates:
            raise FileNotFoundError(f"No image found for {stem}")
        with Image.open(candidates[0]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        result = analyze_image(image, entry["kind"], model, checkpoint)
        result_name = f"{stem}.json"
        image_name = f"{stem}.webp"
        (args.output / result_name).write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        image.save(args.output / image_name, "WEBP", quality=88, method=6)
        index["examples"].append(
            {
                "id": stem,
                "page_kind": entry["kind"],
                "case": entry.get("case"),
                "image": image_name,
                "result": result_name,
            }
        )
        print(f"{stem}: {len(result['elements'])} elements")
    (args.output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
