"""Run structural invariants on visually reviewed archive pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from digitalization import detect_content_graph


def _stream_scores(gold: list[list[int]], predicted: list) -> dict:
    candidates = []
    for gold_index, box in enumerate(gold):
        gx1, gy1, gx2, gy2 = box
        gold_center = (gx1 + gx2) / 2
        gold_height = gy2 - gy1
        for predicted_index, group in enumerate(predicted):
            predicted_center = (group.bbox.x1 + group.bbox.x2) / 2
            overlap = max(0, min(gy2, group.bbox.y2) - max(gy1, group.bbox.y1))
            vertical_recall = overlap / max(1, gold_height)
            center_error = abs(gold_center - predicted_center)
            tolerance = max(10, (gx2 - gx1) * 0.32)
            if vertical_recall >= 0.62 and center_error <= tolerance:
                candidates.append(
                    (
                        vertical_recall - center_error / max(tolerance, 1),
                        gold_index,
                        predicted_index,
                    )
                )
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


def _iou(left: list[int], right) -> float:
    intersection_width = max(0, min(left[2], right.x2) - max(left[0], right.x1))
    intersection_height = max(0, min(left[3], right.y2) - max(left[1], right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = right.width * right.height
    return intersection / max(1, left_area + right_area - intersection)


def _group_scores(gold: list[list[int]], predicted: list) -> dict:
    candidates = sorted(
        (
            (_iou(box, group.bbox), gold_index, predicted_index)
            for gold_index, box in enumerate(gold)
            for predicted_index, group in enumerate(predicted)
        ),
        reverse=True,
    )
    matched_gold = set()
    matched_predicted = set()
    for score, gold_index, predicted_index in candidates:
        if score < 0.45:
            break
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
        default=ROOT / "training/content_graph_regressions.json",
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
            graph = detect_content_graph(source.convert("RGB"))
        streams = [
            stream
            for stream in graph.streams
            if stream.role in {"text", "satellite"}
        ]
        checks = {}
        minimum, maximum = record["text_stream_count"]
        checks["text_stream_count"] = minimum <= len(streams) <= maximum
        stream_scores = None
        if "gold_streams" in record:
            stream_scores = _stream_scores(record["gold_streams"], graph.groups)
            checks["stream_recall"] = (
                stream_scores["recall"] >= record.get("min_stream_recall", 0.0)
            )
            checks["stream_precision"] = (
                stream_scores["precision"] >= record.get("min_stream_precision", 0.0)
            )
        group_scores = None
        if "gold_groups" in record:
            rx1, ry1, rx2, ry2 = record["evaluation_region"]
            predicted_groups = [
                group
                for group in graph.groups
                if max(0, min(ry2, group.bbox.y2) - max(ry1, group.bbox.y1))
                / max(1, group.bbox.height)
                >= 0.65
                and rx1 <= (group.bbox.x1 + group.bbox.x2) / 2 <= rx2
            ]
            group_scores = _group_scores(record["gold_groups"], predicted_groups)
            checks["group_recall"] = (
                group_scores["recall"] >= record.get("min_group_recall", 0.0)
            )
            checks["group_precision"] = (
                group_scores["precision"] >= record.get("min_group_precision", 0.0)
            )
        if "header" in record:
            header = record["header"]
            count = sum(stream.bbox.y2 <= header["end_y"] for stream in streams)
            checks["header_streams"] = count >= header["min_streams"]
        if "barrier" in record:
            barrier = record["barrier"]
            count = sum(
                stream.bbox.y1 < barrier["top_y"]
                and stream.bbox.y2 > barrier["bottom_y"]
                for stream in streams
            )
            checks["barrier_crossings"] = count <= barrier["max_crossing_streams"]
        if "upper" in record:
            upper = record["upper"]
            count = sum(stream.bbox.y1 < upper["end_y"] for stream in streams)
            checks["upper_streams"] = count >= upper["min_streams"]
        if "lower" in record:
            lower = record["lower"]
            count = sum(stream.bbox.y2 > lower["start_y"] for stream in streams)
            checks["lower_streams"] = count >= lower["min_streams"]
            wide_narrow = sum(
                stream.bbox.y2 > lower["start_y"]
                and stream.median_width < graph.glyph_scale * 0.75
                and stream.bbox.width > graph.glyph_scale * 2.0
                for stream in streams
            )
            checks["lower_wide_narrow_streams"] = (
                wide_narrow <= lower.get("max_wide_narrow_streams", wide_narrow)
            )
        pages.append(
            {
                "image": record["image"],
                "glyph_scale": round(graph.glyph_scale, 2),
                "text_stream_count": len(streams),
                "group_count": len(graph.groups),
                "stream_scores": stream_scores,
                "group_scores": group_scores,
                "checks": checks,
            }
        )
        failures.extend(
            f"{record['image']}:{name}" for name, passed in checks.items() if not passed
        )

    result = {"passed": not failures, "failures": failures, "pages": pages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Content-graph regression failed")


if __name__ == "__main__":
    main()
