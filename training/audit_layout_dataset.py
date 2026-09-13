"""Audit the layout-tree detector over a directory of page images."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from digitalization import LayoutNode, detect_layout


def _nodes(root: LayoutNode) -> list[LayoutNode]:
    result: list[LayoutNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(node.children)
    return result


def audit_page(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    document = detect_layout(image)
    nodes = _nodes(document.root)
    leaves = document.reading_order
    empty_leaves = document.root.leaves(include_empty=True)
    empty_count = sum(node.kind == "empty" for node in empty_leaves)
    widths = [node.bbox.width / width for node in leaves]
    inferred_count = sum("inferred_pitch" in node.evidence for node in nodes)
    alignment_count = sum(
        "text_alignment" in node.evidence
        or "page_text_alignment" in node.evidence
        for node in nodes
    )
    partial_subcolumn_count = sum(
        "partial_subcolumn_alignment" in node.evidence for node in nodes
    )

    flags: list[str] = []
    if len(document.root.children) != 2:
        flags.append("spread_not_split_into_two_pages")
    if len(leaves) < 8:
        flags.append("very_few_text_regions")
    if len(leaves) > 120:
        flags.append("very_many_text_regions")
    if any(relative_width > 0.20 for relative_width in widths):
        flags.append("very_wide_text_region")
    if inferred_count > len(nodes) * 0.45:
        flags.append("heavy_pitch_inference")

    evidence_ratio = (
        inferred_count + alignment_count + partial_subcolumn_count
    ) / max(1, len(nodes))
    review_priority = (
        len(flags) * 2.0
        + max(widths, default=0.0) * 2.0
        + evidence_ratio
        + abs(len(leaves) - 35) / 100
    )

    return {
        "image": path.name,
        "width": width,
        "height": height,
        "page_count": len(document.root.children),
        "node_count": len(nodes),
        "text_region_count": len(leaves),
        "empty_region_count": empty_count,
        "max_text_width_fraction": round(max(widths, default=0.0), 4),
        "inferred_pitch_node_count": inferred_count,
        "text_alignment_node_count": alignment_count,
        "partial_subcolumn_node_count": partial_subcolumn_count,
        "review_priority": round(review_priority, 4),
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not images:
        raise SystemExit(f"No page images found in {args.image_dir}")

    pages = []
    for index, path in enumerate(images, start=1):
        pages.append(audit_page(path))
        print(f"[{index:03d}/{len(images):03d}] {path.name}", flush=True)

    flag_counts = Counter(flag for page in pages for flag in page["flags"])
    summary = {
        "page_count": len(pages),
        "flagged_page_count": sum(bool(page["flags"]) for page in pages),
        "flag_counts": dict(sorted(flag_counts.items())),
        "text_region_count": {
            "min": min(page["text_region_count"] for page in pages),
            "max": max(page["text_region_count"] for page in pages),
            "mean": round(
                sum(page["text_region_count"] for page in pages) / len(pages), 2
            ),
        },
        "review_queue": [
            {
                "image": page["image"],
                "review_priority": page["review_priority"],
                "flags": page["flags"],
            }
            for page in sorted(
                pages, key=lambda item: item["review_priority"], reverse=True
            )[:12]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "pages": pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
