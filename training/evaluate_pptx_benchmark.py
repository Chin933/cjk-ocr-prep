"""Evaluate layout structure against the hand-drawn PowerPoint annotations.

The PowerPoint boxes label nested regions rather than OCR text. Evaluation is
class-agnostic on the prediction side: a gold block, major column, or subcolumn
is recovered when any node in the predicted tree matches it. Reading order is
evaluated pairwise among peer boxes in the same annotated block.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from digitalization import Box, LayoutNode, detect_layout


def _iou(left: Box, right: Box) -> float:
    width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _containment(inner: Box, outer: Box) -> float:
    width = max(0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    height = max(0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    area = inner.width * inner.height
    return width * height / area if area else 0.0


def _nodes(root: LayoutNode) -> list[LayoutNode]:
    result: list[LayoutNode] = []
    stack = list(root.children)
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(node.children)
    return result


def _first_order(node: LayoutNode) -> int:
    orders = [leaf.order for leaf in node.leaves() if leaf.order is not None]
    return min(orders, default=10**9)


def _box(values: list[int]) -> Box:
    return Box(*map(int, values))


def evaluate_record(record: dict, image_dir: Path) -> dict:
    image_path = image_dir / record["image"]
    with Image.open(image_path) as source:
        document = detect_layout(source.convert("RGB"))
    predicted = _nodes(document.root)
    gold = [dict(item, box=_box(item["bbox"])) for item in record["annotations"]]

    matches: list[dict] = []
    matched_nodes: list[LayoutNode] = []
    for item in gold:
        node = max(predicted, key=lambda candidate: _iou(item["box"], candidate.bbox))
        score = _iou(item["box"], node.bbox)
        matched_nodes.append(node)
        matches.append(
            {
                "class_name": item["class_name"],
                "gold_bbox": item["bbox"],
                "predicted_node": node.id,
                "predicted_bbox": node.bbox.as_list(),
                "iou": round(score, 4),
            }
        )

    order_correct = 0
    order_total = 0
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(gold):
        by_class[item["class_name"]].append(index)
    blocks = [item["box"] for item in gold if item["class_name"] == "block"]
    for class_name in ("major_column", "subcolumn"):
        indices = by_class[class_name]
        for offset, left_index in enumerate(indices):
            left = gold[left_index]["box"]
            for right_index in indices[offset + 1 :]:
                right = gold[right_index]["box"]
                same_block = any(
                    _containment(left, block) >= 0.8
                    and _containment(right, block) >= 0.8
                    for block in blocks
                )
                y_overlap = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
                if not same_block or y_overlap < 0.5 * min(left.height, right.height):
                    continue
                expected = (left.x1 + left.x2) > (right.x1 + right.x2)
                left_order = _first_order(matched_nodes[left_index])
                right_order = _first_order(matched_nodes[right_index])
                actual = left_order < right_order
                order_correct += int(left_order != right_order and actual == expected)
                order_total += 1

    class_summary = {}
    for class_name, indices in by_class.items():
        scores = [matches[index]["iou"] for index in indices]
        class_summary[class_name] = {
            "count": len(scores),
            "mean_best_iou": round(sum(scores) / len(scores), 4),
            "recall_iou_50": round(sum(score >= 0.5 for score in scores) / len(scores), 4),
            "recall_iou_75": round(sum(score >= 0.75 for score in scores) / len(scores), 4),
        }
    return {
        "image": record["image"],
        "predicted_region_count": len(document.reading_order),
        "classes": class_summary,
        "pairwise_reading_order": {
            "correct": order_correct,
            "total": order_total,
            "accuracy": round(order_correct / order_total, 4) if order_total else None,
        },
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "training" / "pptx_layout_dataset" / "annotations.json",
    )
    parser.add_argument(
        "--images", type=Path, default=PROJECT_ROOT / "training" / "images" / "all"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-order-accuracy", type=float)
    args = parser.parse_args()

    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    pages = [evaluate_record(record, args.images) for record in source["records"]]
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "iou_sum": 0.0, "recall_50": 0, "recall_75": 0}
    )
    for page in pages:
        for match in page["matches"]:
            bucket = totals[match["class_name"]]
            bucket["count"] += 1
            bucket["iou_sum"] += match["iou"]
            bucket["recall_50"] += int(match["iou"] >= 0.5)
            bucket["recall_75"] += int(match["iou"] >= 0.75)
    classes = {
        name: {
            "count": int(values["count"]),
            "mean_best_iou": round(values["iou_sum"] / values["count"], 4),
            "recall_iou_50": round(values["recall_50"] / values["count"], 4),
            "recall_iou_75": round(values["recall_75"] / values["count"], 4),
        }
        for name, values in totals.items()
    }
    order_correct = sum(page["pairwise_reading_order"]["correct"] for page in pages)
    order_total = sum(page["pairwise_reading_order"]["total"] for page in pages)
    summary = {
        "page_count": len(pages),
        "classes": classes,
        "pairwise_reading_order": {
            "correct": order_correct,
            "total": order_total,
            "accuracy": round(order_correct / order_total, 4) if order_total else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "pages": pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    accuracy = summary["pairwise_reading_order"]["accuracy"]
    if (
        args.min_order_accuracy is not None
        and (accuracy is None or accuracy < args.min_order_accuracy)
    ):
        raise SystemExit(
            f"Reading-order regression: {accuracy} < {args.min_order_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
