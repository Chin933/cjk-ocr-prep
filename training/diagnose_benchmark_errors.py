"""Classify structural benchmark failures by their geometric failure mode."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _size(box: list[int]) -> tuple[int, int]:
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.benchmark.read_text(encoding="utf-8"))
    matches = [
        {**match, "image": page["image"]}
        for page in source["pages"]
        for match in page["matches"]
    ]
    report = {}
    for class_name in sorted({match["class_name"] for match in matches}):
        class_matches = [
            match for match in matches if match["class_name"] == class_name
        ]
        failures = [match for match in class_matches if match["iou"] < 0.5]
        categories: Counter[str] = Counter()
        area_ratios = []
        width_ratios = []
        height_ratios = []
        node_uses: Counter[tuple[str, str]] = Counter()
        examples: dict[str, list[dict]] = defaultdict(list)
        for match in failures:
            area_ratio = _area(match["predicted_bbox"]) / max(
                1, _area(match["gold_bbox"])
            )
            predicted_width, predicted_height = _size(match["predicted_bbox"])
            gold_width, gold_height = _size(match["gold_bbox"])
            width_ratio = predicted_width / gold_width
            height_ratio = predicted_height / gold_height
            area_ratios.append(area_ratio)
            width_ratios.append(width_ratio)
            height_ratios.append(height_ratio)
            if width_ratio > 1.5:
                category = "merged_across_columns"
            elif width_ratio < 0.67:
                category = "over_split_width"
            elif height_ratio > 1.5:
                category = "overextended_vertical_range"
            elif height_ratio < 0.67:
                category = "short_vertical_range"
            else:
                category = "boundary_or_hierarchy"
            categories[category] += 1
            node_uses[(match["image"], match["predicted_node"])] += 1
            if len(examples[category]) < 5:
                examples[category].append(
                    {
                        "gold_bbox": match["gold_bbox"],
                        "predicted_bbox": match["predicted_bbox"],
                        "image": match["image"],
                        "iou": match["iou"],
                        "area_ratio": round(area_ratio, 3),
                        "width_ratio": round(width_ratio, 3),
                        "height_ratio": round(height_ratio, 3),
                    }
                )
        report[class_name] = {
            "count": len(class_matches),
            "failed_iou_50": len(failures),
            "failure_rate": round(len(failures) / max(1, len(class_matches)), 4),
            "failure_modes": dict(categories),
            "median_failed_area_ratio": round(statistics.median(area_ratios), 3)
            if area_ratios
            else None,
            "median_failed_width_ratio": round(statistics.median(width_ratios), 3)
            if width_ratios
            else None,
            "median_failed_height_ratio": round(statistics.median(height_ratios), 3)
            if height_ratios
            else None,
            "duplicate_gold_matches": sum(
                count - 1 for count in node_uses.values() if count > 1
            ),
            "examples": dict(examples),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
