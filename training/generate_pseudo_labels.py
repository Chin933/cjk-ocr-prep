"""Generate layout-tree pseudo-labels for a directory of page images."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from digitalization import LayoutConfig, __version__, detect_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    config = LayoutConfig()
    records = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            document = detect_layout(source.convert("RGB"), config)
        records.append({"image": path.name, "layout": document.to_dict()})
        print(f"[{index:03d}/{len(paths):03d}] {path.name}", flush=True)

    payload = {
        "format": "digitalization-layout-tree-v1",
        "generator_version": __version__,
        "config": asdict(config),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Pseudo-labels: {args.output}")


if __name__ == "__main__":
    main()
