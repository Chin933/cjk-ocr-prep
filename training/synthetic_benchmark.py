"""Generate exact synthetic layout trees and stress-test the detector.

The benchmark expands shape coverage without paid or manual annotation. It
does not synthesize readable characters; glyph-like marks exist only to create
realistic geometric evidence for layout and reading order.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from digitalization import Box, LayoutNode, detect_layout, draw_layout

TEMPLATE_COUNT = 10


@dataclass
class GoldNode:
    box: Box
    axis: str = "none"
    children: list["GoldNode"] = field(default_factory=list)

    def leaves(self) -> list["GoldNode"]:
        if not self.children:
            return [self]
        return [leaf for child in self.children for leaf in child.leaves()]


def _split(node: GoldNode, axis: str, weights: list[float]) -> list[GoldNode]:
    total = sum(weights)
    length = node.box.width if axis == "x" else node.box.height
    offsets = [0]
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        offsets.append(round(length * running / total))
    offsets.append(length)
    natural = []
    for start, end in zip(offsets, offsets[1:]):
        if axis == "x":
            box = Box(node.box.x1 + start, node.box.y1, node.box.x1 + end, node.box.y2)
        else:
            box = Box(node.box.x1, node.box.y1 + start, node.box.x2, node.box.y1 + end)
        natural.append(GoldNode(box))
    node.axis = axis
    node.children = list(reversed(natural)) if axis == "x" else natural
    return natural


def _make_page(box: Box, template: int, rng: random.Random) -> GoldNode:
    root = GoldNode(box)
    if template == 0:
        _split(root, "x", [1 + rng.random() * 0.2 for _ in range(7)])
    elif template == 1:
        top, bottom = _split(root, "y", [0.38, 0.62])
        _split(top, "x", [1, 1.8, 1])
        _split(bottom, "x", [1] * 7)
    elif template == 2:
        content, side_title = _split(root, "x", [0.78, 0.22])
        _split(side_title, "y", [0.42, 0.58])
        upper, lower = _split(content, "y", [0.46, 0.54])
        _split(upper, "x", [1] * 5)
        _split(lower, "x", [1] * 8)
    elif template == 3:
        bands = _split(root, "y", [0.25, 0.36, 0.39])
        for band, count in zip(bands, (3, 7, 4)):
            _split(band, "x", [1] * count)
    elif template == 4:
        columns = _split(root, "x", [1] * 5)
        for index in (1, 3):
            sections = _split(columns[index], "y", [0.22, 0.48, 0.30])
            _split(sections[1], "x", [1, 1])
    elif template == 5:
        upper, lower = _split(root, "y", [0.55, 0.45])
        upper_columns = _split(upper, "x", [1, 1, 2.1, 1])
        _split(upper_columns[2], "y", [0.3, 0.7])
        _split(lower, "x", [1, 1, 1.8, 1, 1])
    elif template == 6:
        heading, body = _split(root, "y", [0.16, 0.84])
        _split(heading, "x", [0.22, 0.56, 0.22])
        columns = _split(body, "x", [1, 1, 1.15, 1, 1, 1])
        _split(columns[2], "y", [0.37, 0.63])
    elif template == 7:
        columns = _split(root, "x", [0.65, 1.3, 0.8, 1.55])
        _split(columns[0], "y", [0.28, 0.72])
        middle = _split(columns[1], "y", [0.22, 0.46, 0.32])
        _split(middle[1], "x", [1, 1])
        _split(columns[3], "y", [0.61, 0.39])
    elif template == 8:
        heading, body, colophon = _split(root, "y", [0.13, 0.72, 0.15])
        _split(heading, "x", [1, 2.4, 1])
        columns = _split(body, "x", [1, 1, 1.7, 1, 1, 1])
        _split(columns[2], "y", [0.31, 0.39, 0.30])
        _split(colophon, "x", [1, 1, 1, 1])
    else:
        columns = _split(root, "x", [1.4, 0.75, 1.8])
        left_sections = _split(columns[0], "y", [0.35, 0.65])
        _split(left_sections[1], "x", [1, 1])
        center_sections = _split(columns[1], "y", [0.24, 0.52, 0.24])
        _split(center_sections[1], "x", [1, 1])
        right_sections = _split(columns[2], "y", [0.58, 0.42])
        _split(right_sections[0], "x", [1, 1, 1])
    return root


def _draw_rule(draw: ImageDraw.ImageDraw, node: GoldNode, rng: random.Random) -> None:
    if not node.children:
        return
    color = rng.choice(("#111111", "#333333", "#777777"))
    width = rng.choice((2, 3, 4))
    if node.axis == "x":
        cuts = sorted({child.box.x1 for child in node.children} | {child.box.x2 for child in node.children})
        for x in cuts[1:-1]:
            if rng.random() < 0.35:
                for y in range(node.box.y1, node.box.y2, 34):
                    draw.line((x, y, x, min(y + 22, node.box.y2)), fill=color, width=width)
            else:
                draw.line((x, node.box.y1, x, node.box.y2), fill=color, width=width)
    else:
        cuts = sorted({child.box.y1 for child in node.children} | {child.box.y2 for child in node.children})
        for y in cuts[1:-1]:
            draw.line((node.box.x1, y, node.box.x2, y), fill=color, width=width)
    for child in node.children:
        _draw_rule(draw, child, rng)


def _draw_text(draw: ImageDraw.ImageDraw, leaf: GoldNode, rng: random.Random) -> None:
    box = leaf.box
    margin_x = max(6, int(box.width * 0.18))
    glyph_width = max(5, min(20, int(box.width * 0.28)))
    glyph_height = max(8, min(28, int(box.width * 0.38)))
    center = (box.x1 + box.x2) // 2 + rng.randint(-max(1, box.width // 12), max(1, box.width // 12))
    y = box.y1 + rng.randint(18, 45)
    while y + glyph_height < box.y2 - 12:
        half = min(glyph_width // 2, center - box.x1 - margin_x, box.x2 - margin_x - center)
        if half > 2:
            draw.rectangle((center - half, y, center + half, y + glyph_height), fill="#111111")
            if rng.random() < 0.35:
                draw.rectangle((center - half - 3, y + 5, center + half + 3, y + 8), fill="#111111")
        y += glyph_height + rng.randint(8, 18)


def generate_case(seed: int, template: int) -> tuple[Image.Image, GoldNode]:
    rng = random.Random(seed)
    image = Image.new("RGB", (1200, 820), "white")
    draw = ImageDraw.Draw(image)
    frame = Box(20, 20, 1180, 800)
    draw.rectangle(frame.as_list(), outline="#050505", width=7)
    center = 600 + rng.randint(-12, 12)
    draw.line((center, 20, center, 800), fill="#050505", width=7)
    root = GoldNode(frame, "x")
    right = _make_page(Box(center + 4, 27, 1173, 793), template, rng)
    left = _make_page(
        Box(27, 27, center - 4, 793), (template + 1) % TEMPLATE_COUNT, rng
    )
    root.children = [right, left]
    for page in root.children:
        _draw_rule(draw, page, rng)
        for leaf in page.leaves():
            _draw_text(draw, leaf, rng)
    for _ in range(120):
        x, y = rng.randrange(25, 1175), rng.randrange(25, 795)
        shade = rng.choice((150, 180, 210))
        draw.point((x, y), fill=(shade, shade, shade))
    return image, root


def _iou(left: Box, right: Box) -> float:
    width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _first_order(node: LayoutNode) -> int:
    return min((leaf.order for leaf in node.leaves() if leaf.order is not None), default=10**9)


def evaluate_case(image: Image.Image, gold: GoldNode) -> tuple[dict, object]:
    prediction = detect_layout(image)
    predicted_nodes = []
    stack = list(prediction.root.children)
    while stack:
        node = stack.pop()
        predicted_nodes.append(node)
        stack.extend(node.children)
    gold_leaves = gold.leaves()
    matches = [
        max(predicted_nodes, key=lambda node: _iou(leaf.box, node.bbox))
        for leaf in gold_leaves
    ]
    scores = [_iou(leaf.box, node.bbox) for leaf, node in zip(gold_leaves, matches)]
    correct = 0
    total = 0
    for left in range(len(matches)):
        for right in range(left + 1, len(matches)):
            left_order = _first_order(matches[left])
            right_order = _first_order(matches[right])
            correct += int(left_order < right_order)
            total += 1
    return (
        {
            "gold_regions": len(gold_leaves),
            "predicted_regions": len(prediction.reading_order),
            "mean_best_iou": round(sum(scores) / len(scores), 4),
            "recall_iou_50": round(sum(score >= 0.5 for score in scores) / len(scores), 4),
            "order_correct": correct,
            "order_total": total,
            "pairwise_order_accuracy": round(correct / total, 4) if total else None,
        },
        prediction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--min-region-recall", type=float)
    parser.add_argument("--min-order-accuracy", type=float)
    parser.add_argument("--min-template-recall", type=float)
    parser.add_argument("--min-template-order", type=float)
    args = parser.parse_args()

    cases = []
    rendered = []
    for index in range(args.cases):
        template = index % TEMPLATE_COUNT
        image, gold = generate_case(7100 + index, template)
        result, prediction = evaluate_case(image, gold)
        result.update({"case": index, "template": template})
        cases.append(result)
        rendered.append((image, prediction))

    total_pairs = sum(item["order_total"] for item in cases)
    summary = {
        "case_count": len(cases),
        "mean_recall_iou_50": round(sum(item["recall_iou_50"] for item in cases) / len(cases), 4),
        "pairwise_order_accuracy": round(
            sum(item["order_correct"] for item in cases) / total_pairs, 4
        ),
        "by_template": {
            str(template): {
                "mean_recall_iou_50": round(
                    sum(item["recall_iou_50"] for item in cases if item["template"] == template)
                    / sum(item["template"] == template for item in cases),
                    4,
                ),
                "mean_order_accuracy": round(
                    sum(item["pairwise_order_accuracy"] for item in cases if item["template"] == template)
                    / sum(item["template"] == template for item in cases),
                    4,
                ),
            }
            for template in range(TEMPLATE_COUNT)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8"
    )
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        for obsolete in args.preview_dir.glob("case_*.png"):
            obsolete.unlink()
        worst = sorted(range(len(cases)), key=lambda index: cases[index]["recall_iou_50"])[:6]
        for index in worst:
            image, prediction = rendered[index]
            draw_layout(image, prediction).save(args.preview_dir / f"case_{index:02d}.png")
    print(json.dumps(summary, indent=2))
    failures = []
    if (
        args.min_region_recall is not None
        and summary["mean_recall_iou_50"] < args.min_region_recall
    ):
        failures.append(
            f"region recall {summary['mean_recall_iou_50']:.4f} < {args.min_region_recall:.4f}"
        )
    if (
        args.min_order_accuracy is not None
        and summary["pairwise_order_accuracy"] < args.min_order_accuracy
    ):
        failures.append(
            f"order accuracy {summary['pairwise_order_accuracy']:.4f} < "
            f"{args.min_order_accuracy:.4f}"
        )
    if args.min_template_recall is not None:
        below = [
            name
            for name, metrics in summary["by_template"].items()
            if metrics["mean_recall_iou_50"] < args.min_template_recall
        ]
        if below:
            failures.append(
                "template recall below "
                f"{args.min_template_recall:.4f}: {', '.join(below)}"
            )
    if args.min_template_order is not None:
        below = [
            name
            for name, metrics in summary["by_template"].items()
            if metrics["mean_order_accuracy"] < args.min_template_order
        ]
        if below:
            failures.append(
                "template order below "
                f"{args.min_template_order:.4f}: {', '.join(below)}"
            )
    if failures:
        raise SystemExit("Synthetic benchmark regression: " + "; ".join(failures))


if __name__ == "__main__":
    main()
