"""Run structural regressions for the local multi-scale glyph graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from digitalization import detect_glyph_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "training/glyph_graph_regressions.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "training/images/archive_diverse",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pages = []
    failures = []
    for record in spec["records"]:
        with Image.open(args.images / record["image"]) as source:
            graph = detect_glyph_graph(source.convert("RGB"))
        by_node = {node.id: node for node in graph.nodes}
        checks = {}
        minimum, maximum = record["node_count"]
        checks["node_count"] = minimum <= len(graph.nodes) <= maximum
        minimum, maximum = record["chain_count"]
        checks["chain_count"] = minimum <= len(graph.chains) <= maximum
        wide_large = sum(
            chain.scale_class == "large"
            and chain.bbox.width > graph.glyph_scale * 1.9
            for chain in graph.chains
        )
        checks["wide_large_chains"] = wide_large <= record["max_wide_large_chains"]

        if "barrier" in record:
            barrier = record["barrier"]
            crossings = sum(
                by_node[relation.source].bbox.y1 < barrier["top_y"]
                and by_node[relation.target].bbox.y2 > barrier["bottom_y"]
                for relation in graph.relations
                if relation.relation == "continues_to"
            )
            checks["barrier_crossings"] = crossings <= barrier["max_crossings"]
        for name in ("upper", "lower"):
            if name not in record:
                continue
            section = record[name]
            nodes = [
                node
                for node in graph.nodes
                if section["start_y"] <= node.center_y < section["end_y"]
            ]
            checks[f"{name}_small"] = (
                sum(node.scale_class == "small" for node in nodes)
                >= section["min_small"]
            )
            checks[f"{name}_large"] = (
                sum(node.scale_class == "large" for node in nodes)
                >= section["min_large"]
            )
        if "relations" in record:
            relation_counts = {
                relation: sum(
                    edge.relation == relation for edge in graph.relations
                )
                for relation in ("annotates", "next_record")
            }
            checks["annotation_relations"] = (
                relation_counts["annotates"] >= record["relations"]["min_annotates"]
            )
            checks["record_relations"] = (
                relation_counts["next_record"]
                >= record["relations"]["min_next_record"]
            )
        else:
            relation_counts = None

        failures.extend(
            f"{record['image']}:{name}" for name, passed in checks.items() if not passed
        )
        pages.append(
            {
                "image": record["image"],
                "glyph_scale": round(graph.glyph_scale, 2),
                "node_count": len(graph.nodes),
                "chain_count": len(graph.chains),
                "scale_counts": {
                    scale_class: sum(
                        node.scale_class == scale_class for node in graph.nodes
                    )
                    for scale_class in ("fragment", "small", "medium", "large")
                },
                "relation_counts": relation_counts,
                "checks": checks,
            }
        )

    result = {"passed": not failures, "failures": failures, "pages": pages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Glyph-graph regression failed")


if __name__ == "__main__":
    main()
