"""Command line interface for layout-only image analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .layout import detect_layout, detect_rule_graph, draw_layout, draw_rule_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="digitalization-layout",
        description="Detect hierarchical layout and reading order in a page image.",
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--graph-output", type=Path)
    parser.add_argument("--graph-overlay", type=Path)
    args = parser.parse_args()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    result = detect_layout(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        draw_layout(image, result).save(args.overlay)
    if args.graph_output or args.graph_overlay:
        graph = detect_rule_graph(image)
        if args.graph_output:
            args.graph_output.parent.mkdir(parents=True, exist_ok=True)
            args.graph_output.write_text(
                json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.graph_overlay:
            args.graph_overlay.parent.mkdir(parents=True, exist_ok=True)
            draw_rule_graph(image, graph).save(args.graph_overlay)

    print(
        f"pages={len(result.root.children)} "
        f"reading_order_regions={len(result.reading_order)} output={args.output}"
    )


if __name__ == "__main__":
    main()
