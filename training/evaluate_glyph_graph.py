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


def _iou(left: list[int], right) -> float:
    width = max(0, min(left[2], right.x2) - max(left[0], right.x1))
    height = max(0, min(left[3], right.y2) - max(left[1], right.y1))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    return intersection / max(1, left_area + right.width * right.height - intersection)


def _record_scores(gold: list[list[int]], predicted: list) -> dict:
    candidates = []
    for gold_index, box in enumerate(gold):
        for predicted_index, record in enumerate(predicted):
            score = _iou(box, record.bbox)
            if score >= 0.45:
                candidates.append((score, gold_index, predicted_index))
    matched_gold = set()
    matched_predicted = set()
    for _, gold_index, predicted_index in sorted(candidates, reverse=True):
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
    return {
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "matched": len(matched_gold),
        "recall": round(len(matched_gold) / max(1, len(gold)), 4),
        "precision": round(len(matched_predicted) / max(1, len(predicted)), 4),
    }


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
        by_chain = {chain.id: chain for chain in graph.chains}
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
        record_scores = None
        if "gold_records" in record:
            rx1, ry1, rx2, ry2 = record["evaluation_region"]
            predicted_records = [
                item
                for item in graph.records
                if rx1 <= by_chain[item.primary_chain_id].center_x <= rx2
                and max(0, min(ry2, item.bbox.y2) - max(ry1, item.bbox.y1))
                / max(1, item.bbox.height)
                >= 0.65
            ]
            record_scores = _record_scores(record["gold_records"], predicted_records)
            checks["record_recall"] = (
                record_scores["recall"] >= record.get("min_record_recall", 0.0)
            )
            checks["record_precision"] = (
                record_scores["precision"] >= record.get("min_record_precision", 0.0)
            )

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
            record_by_primary = {
                item.primary_chain_id: item for item in graph.records
            }
            checks["record_order_consistency"] = all(
                record_by_primary[edge.source].order
                < record_by_primary[edge.target].order
                for edge in graph.relations
                if edge.relation == "next_record"
                and edge.source in record_by_primary
                and edge.target in record_by_primary
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
                "record_count": len(graph.records),
                "record_scores": record_scores,
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
