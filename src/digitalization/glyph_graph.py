"""Local glyph-instance graph for mixed-size vertical layouts.

This layer deliberately stays below OCR.  It reconstructs conservative glyph
candidates from ink components, estimates relative scale inside each ruled
band, and keeps continuation, annotation, and record-transition relations
separate so that a wrong early column merge does not destroy the 2-D evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import cv2
import numpy as np
from PIL import Image

from .content_graph import (
    _estimate_glyph_scale,
    _horizontal_bands,
)
from .layout import Box, _binarize


@dataclass
class GlyphNode:
    id: str
    bbox: Box
    center_x: float
    center_y: float
    ink: int
    local_scale: float
    scale_class: str = "medium"
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "ink": self.ink,
            "local_scale": round(self.local_scale, 2),
            "scale_class": self.scale_class,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class GlyphRelation:
    source: str
    target: str
    relation: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class GlyphChain:
    id: str
    bbox: Box
    node_ids: list[str]
    scale_class: str
    center_x: float
    order: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "node_ids": self.node_ids,
            "scale_class": self.scale_class,
            "center_x": round(self.center_x, 2),
            "order": self.order,
        }


@dataclass
class GlyphGraph:
    width: int
    height: int
    region: Box
    glyph_scale: float
    nodes: list[GlyphNode]
    chains: list[GlyphChain]
    relations: list[GlyphRelation]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "region": self.region.as_list(),
            "glyph_scale": round(self.glyph_scale, 2),
            "reading_order": [
                chain.id for chain in sorted(self.chains, key=lambda item: item.order)
            ],
            "nodes": [node.to_dict() for node in self.nodes],
            "chains": [chain.to_dict() for chain in self.chains],
            "relations": [relation.to_dict() for relation in self.relations],
        }


def _overlap(left1: int, right1: int, left2: int, right2: int) -> int:
    return max(0, min(right1, right2) - max(left1, left2))


def _remove_rules_for_glyphs(binary: np.ndarray, scale: float) -> np.ndarray:
    """Remove only lines much longer than a plausible glyph."""
    height, width = binary.shape
    horizontal_length = max(int(scale * 4.0), int(width * 0.28), 30)
    vertical_length = max(int(scale * 4.0), int(height * 0.28), 30)
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    return cv2.subtract(binary, cv2.max(horizontal, vertical))


def _split_wide_blob(
    ink: np.ndarray, box: Box, region: Box, scale: float
) -> list[Box]:
    """Split a horizontally fused multi-glyph blob at a supported ink valley."""
    if box.width <= scale * 1.60 or box.width / max(1, box.height) <= 1.55:
        return [box]
    x1, x2 = box.x1 - region.x1, box.x2 - region.x1
    y1, y2 = box.y1 - region.y1, box.y2 - region.y1
    local = ink[y1:y2, x1:x2] > 0
    profile = local.sum(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (3, 1), 0).ravel()
    lower = max(2, int(box.width * 0.25))
    upper = min(box.width - 2, int(box.width * 0.75))
    if upper <= lower:
        return [box]
    cut = lower + int(np.argmin(profile[lower:upper]))
    positive = profile[profile > 0]
    if positive.size == 0 or profile[cut] > np.percentile(positive, 35) * 0.62:
        return [box]

    parts = []
    for start, end in ((0, cut), (cut, box.width)):
        ys, xs = np.nonzero(local[:, start:end])
        if xs.size == 0:
            continue
        part = Box(
            box.x1 + start + int(xs.min()),
            box.y1 + int(ys.min()),
            box.x1 + start + int(xs.max()) + 1,
            box.y1 + int(ys.max()) + 1,
        )
        parts.extend(_split_wide_blob(ink, part, region, scale))
    return parts if len(parts) >= 2 else [box]


def _component_nodes(ink: np.ndarray, region: Box, scale: float) -> list[GlyphNode]:
    mask = cv2.morphologyEx(
        (ink > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[Box, int]] = []
    minimum_ink = max(4, int(scale * scale * 0.004))
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_ink or width < 2 or height < 2:
            continue
        if width > scale * 2.1 or height > scale * 2.1:
            continue
        aspect = max(width / max(1, height), height / max(1, width))
        if aspect > 5.0:
            continue
        components.append(
            (Box(region.x1 + x, region.y1 + y, region.x1 + x + width, region.y1 + y + height), area)
        )

    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    cell = max(8, int(scale))
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (box, _) in enumerate(components):
        key = (int((box.x1 - region.x1) // cell), int((box.y1 - region.y1) // cell))
        buckets.setdefault(key, []).append(index)
    for index, (left, _) in enumerate(components):
        key_x = int((left.x1 - region.x1) // cell)
        key_y = int((left.y1 - region.y1) // cell)
        nearby = []
        for bx in range(key_x - 1, key_x + 2):
            for by in range(key_y - 1, key_y + 2):
                nearby.extend(buckets.get((bx, by), []))
        for other_index in nearby:
            if other_index <= index:
                continue
            right, _ = components[other_index]
            vertical_overlap = _overlap(left.y1, left.y2, right.y1, right.y2)
            overlap_ratio = vertical_overlap / max(1, min(left.height, right.height))
            horizontal_gap = max(0, max(left.x1, right.x1) - min(left.x2, right.x2))
            combined_width = max(left.x2, right.x2) - min(left.x1, right.x1)
            combined_height = max(left.y2, right.y2) - min(left.y1, right.y1)
            if (
                overlap_ratio >= 0.58
                and horizontal_gap <= scale * 0.22
                and combined_width <= scale * 1.28
                and combined_height <= scale * 1.35
                and min(left.width, right.width) <= scale * 0.55
            ):
                union(index, other_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(components)):
        grouped.setdefault(find(index), []).append(index)
    nodes = []
    for members in grouped.values():
        boxes = [components[index][0] for index in members]
        bbox = Box(
            min(box.x1 for box in boxes),
            min(box.y1 for box in boxes),
            max(box.x2 for box in boxes),
            max(box.y2 for box in boxes),
        )
        for part in _split_wide_blob(ink, bbox, region, scale):
            local = ink[
                part.y1 - region.y1 : part.y2 - region.y1,
                part.x1 - region.x1 : part.x2 - region.x1,
            ]
            area = int((local > 0).sum())
            local_scale = sqrt(max(1, part.width * part.height))
            fill = area / max(1, part.width * part.height)
            nodes.append(
                GlyphNode(
                    f"glyph_{len(nodes):05d}",
                    part,
                    (part.x1 + part.x2) / 2,
                    (part.y1 + part.y2) / 2,
                    area,
                    local_scale,
                    confidence=max(0.1, min(1.0, fill * 2.2)),
                )
            )
    return nodes


def _assign_scale_classes(
    nodes: list[GlyphNode],
    bands: list[tuple[int, int]],
    region: Box,
    scale: float,
) -> None:
    fragment_limit = max(4.0, scale * 0.28)
    for node in nodes:
        if node.local_scale < fragment_limit:
            node.scale_class = "fragment"
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        items = [
            node
            for node in nodes
            if top <= node.center_y < bottom and node.scale_class != "fragment"
        ]
        if len(items) < 10:
            continue
        values = np.asarray([log(max(2.0, node.local_scale)) for node in items])
        low, high = (float(value) for value in np.percentile(values, [10, 90]))
        for _ in range(12):
            low_items = values[np.abs(values - low) <= np.abs(values - high)]
            high_items = values[np.abs(values - low) > np.abs(values - high)]
            if low_items.size == 0 or high_items.size == 0:
                break
            low, high = float(np.median(low_items)), float(np.median(high_items))
        ratio = exp(high - low)
        lower_count = int(np.sum(np.abs(values - low) <= np.abs(values - high)))
        if ratio < 1.32 or min(lower_count, len(items) - lower_count) < len(items) * 0.12:
            continue
        boundary = (low + high) / 2
        for node in items:
            node.scale_class = "small" if log(max(2.0, node.local_scale)) <= boundary else "large"


def _smooth_scale_classes(
    nodes: list[GlyphNode],
    bands: list[tuple[int, int]],
    region: Box,
    scale: float,
) -> None:
    """Use vertically aligned neighbours to distinguish font scale from glyph shape."""
    updates = {}
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        items = [
            node
            for node in nodes
            if top <= node.center_y < bottom
            and node.scale_class in {"small", "large"}
        ]
        for node in items:
            neighbours = sorted(
                (
                    other
                    for other in items
                    if other is not node
                    and abs(other.center_x - node.center_x) <= scale * 0.34
                    and abs(other.center_y - node.center_y) <= scale * 4.0
                ),
                key=lambda other: abs(other.center_y - node.center_y),
            )[:6]
            if len(neighbours) < 2:
                continue
            labels = [other.scale_class for other in neighbours]
            majority = max(set(labels), key=labels.count)
            if labels.count(majority) / len(labels) >= 0.67:
                updates[node.id] = majority
    for node in nodes:
        if node.id in updates:
            node.scale_class = updates[node.id]


def _continuation_relations(
    nodes: list[GlyphNode], scale: float, bands: list[tuple[int, int]], region: Box
) -> list[GlyphRelation]:
    candidates = []
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        items = [node for node in nodes if top <= node.center_y < bottom]
        for source in items:
            for target in items:
                if target.center_y <= source.center_y:
                    continue
                vertical_gap = target.bbox.y1 - source.bbox.y2
                if vertical_gap < -min(source.bbox.height, target.bbox.height) * 0.25:
                    continue
                if vertical_gap > max(scale * 1.55, source.local_scale * 1.8):
                    continue
                horizontal_error = abs(source.center_x - target.center_x)
                tolerance = max(scale * 0.22, min(source.local_scale, target.local_scale) * 0.48)
                if horizontal_error > tolerance:
                    continue
                if (
                    source.scale_class != "medium"
                    and target.scale_class != "medium"
                    and source.scale_class != target.scale_class
                ):
                    continue
                size_ratio = max(source.local_scale, target.local_scale) / max(
                    1.0, min(source.local_scale, target.local_scale)
                )
                if size_ratio > 1.75:
                    continue
                confidence = exp(-horizontal_error / max(1.0, tolerance)) * exp(
                    -max(0, vertical_gap) / max(scale, 1.0)
                )
                candidates.append((confidence, source.id, target.id))
    used_source: set[str] = set()
    used_target: set[str] = set()
    selected = []
    for confidence, source, target in sorted(candidates, reverse=True):
        if source in used_source or target in used_target:
            continue
        used_source.add(source)
        used_target.add(target)
        selected.append(GlyphRelation(source, target, "continues_to", confidence))
    return selected


def _make_chains(nodes: list[GlyphNode], relations: list[GlyphRelation]) -> list[GlyphChain]:
    by_id = {node.id: node for node in nodes}
    following = {
        relation.source: relation.target
        for relation in relations
        if relation.relation == "continues_to"
    }
    incoming = set(following.values())
    chains = []
    for node in nodes:
        if node.id in incoming:
            continue
        members = [node.id]
        cursor = node.id
        while cursor in following:
            cursor = following[cursor]
            members.append(cursor)
        items = [by_id[item] for item in members]
        bbox = Box(
            min(item.bbox.x1 for item in items),
            min(item.bbox.y1 for item in items),
            max(item.bbox.x2 for item in items),
            max(item.bbox.y2 for item in items),
        )
        labels = [
            item.scale_class
            for item in items
            if item.scale_class not in {"medium", "fragment"}
        ]
        if labels:
            scale_class = max(set(labels), key=labels.count)
        elif all(item.scale_class == "fragment" for item in items):
            scale_class = "fragment"
        else:
            scale_class = "medium"
        chains.append(
            GlyphChain(
                f"chain_{len(chains):05d}",
                bbox,
                members,
                scale_class,
                float(np.median([item.center_x for item in items])),
                0,
            )
        )
    for order, chain in enumerate(
        sorted(chains, key=lambda item: (item.bbox.y1 // 200, -item.center_x, item.bbox.y1))
    ):
        chain.order = order
    return chains


def _split_drifting_chains(
    chains: list[GlyphChain], nodes: list[GlyphNode], scale: float
) -> list[GlyphChain]:
    """Prevent small pairwise shifts from accumulating into a neighbouring column."""
    by_node = {node.id: node for node in nodes}
    output = []
    for chain in chains:
        items = sorted((by_node[item] for item in chain.node_ids), key=lambda item: item.center_y)
        factor = {"small": 0.42, "large": 0.72}.get(chain.scale_class, 0.55)
        tolerance = scale * factor
        runs: list[list[GlyphNode]] = []
        for node in items:
            if not runs:
                runs.append([node])
                continue
            centers = [item.center_x for item in runs[-1]]
            if max([*centers, node.center_x]) - min([*centers, node.center_x]) <= tolerance:
                runs[-1].append(node)
            else:
                runs.append([node])
        for run in runs:
            bbox = Box(
                min(item.bbox.x1 for item in run),
                min(item.bbox.y1 for item in run),
                max(item.bbox.x2 for item in run),
                max(item.bbox.y2 for item in run),
            )
            labels = [
                item.scale_class
                for item in run
                if item.scale_class not in {"medium", "fragment"}
            ]
            if labels:
                scale_class = max(set(labels), key=labels.count)
            elif all(item.scale_class == "fragment" for item in run):
                scale_class = "fragment"
            else:
                scale_class = "medium"
            output.append(
                GlyphChain(
                    "",
                    bbox,
                    [item.id for item in run],
                    scale_class,
                    float(np.median([item.center_x for item in run])),
                    0,
                )
            )
    for index, chain in enumerate(output):
        chain.id = f"chain_{index:05d}"
    return output


def _split_annotation_chains(
    chains: list[GlyphChain], nodes: list[GlyphNode], scale: float
) -> list[GlyphChain]:
    """Split a small-print chain when its nearest primary record changes."""
    by_node = {node.id: node for node in nodes}
    primaries = [
        chain
        for chain in chains
        if chain.scale_class == "large" and len(chain.node_ids) >= 2
    ]
    output: list[GlyphChain] = []
    for chain in chains:
        if chain.scale_class != "small" or len(chain.node_ids) < 3:
            output.append(chain)
            continue
        items = sorted((by_node[item] for item in chain.node_ids), key=lambda item: item.center_y)
        labelled = []
        for node in items:
            candidates = []
            for primary in primaries:
                horizontal_gap = max(
                    0,
                    max(node.bbox.x1, primary.bbox.x1)
                    - min(node.bbox.x2, primary.bbox.x2),
                )
                if horizontal_gap > scale * 1.9:
                    continue
                if node.center_y < primary.bbox.y1:
                    vertical_distance = primary.bbox.y1 - node.center_y
                elif node.center_y > primary.bbox.y2:
                    vertical_distance = node.center_y - primary.bbox.y2
                else:
                    vertical_distance = 0.0
                if vertical_distance > scale * 1.4:
                    continue
                cost = (
                    horizontal_gap / max(scale, 1.0)
                    + vertical_distance / max(scale, 1.0)
                    + abs(node.center_x - primary.center_x) / max(scale * 3.0, 1.0)
                )
                candidates.append((cost, primary.id))
            label = min(candidates)[1] if candidates else None
            labelled.append((node, label))

        runs: list[list[GlyphNode]] = []
        run_labels: list[str | None] = []
        for node, label in labelled:
            if run_labels and label == run_labels[-1]:
                runs[-1].append(node)
            else:
                run_labels.append(label)
                runs.append([node])
        meaningful_labels = {label for label in run_labels if label is not None}
        if len(meaningful_labels) < 2:
            output.append(chain)
            continue
        for run in runs:
            bbox = Box(
                min(item.bbox.x1 for item in run),
                min(item.bbox.y1 for item in run),
                max(item.bbox.x2 for item in run),
                max(item.bbox.y2 for item in run),
            )
            output.append(
                GlyphChain(
                    "",
                    bbox,
                    [item.id for item in run],
                    "small",
                    float(np.median([item.center_x for item in run])),
                    0,
                )
            )
    for index, chain in enumerate(output):
        chain.id = f"chain_{index:05d}"
    for order, chain in enumerate(
        sorted(output, key=lambda item: (item.bbox.y1 // 200, -item.center_x, item.bbox.y1))
    ):
        chain.order = order
    return output


def _structural_relations(
    chains: list[GlyphChain], scale: float
) -> list[GlyphRelation]:
    relations = []
    large = [chain for chain in chains if chain.scale_class == "large"]
    for satellite in (chain for chain in chains if chain.scale_class == "small"):
        candidates = []
        for primary in large:
            vertical_overlap = _overlap(
                satellite.bbox.y1,
                satellite.bbox.y2,
                primary.bbox.y1,
                primary.bbox.y2,
            )
            overlap_ratio = vertical_overlap / max(
                1, min(satellite.bbox.height, primary.bbox.height)
            )
            horizontal_gap = max(
                0,
                max(satellite.bbox.x1, primary.bbox.x1)
                - min(satellite.bbox.x2, primary.bbox.x2),
            )
            if overlap_ratio < 0.28 or horizontal_gap > scale * 2.4:
                continue
            distance = abs(satellite.center_x - primary.center_x)
            confidence = overlap_ratio * exp(-distance / max(scale * 1.6, 1.0))
            candidates.append((confidence, primary))
        if candidates:
            confidence, primary = max(candidates, key=lambda item: item[0])
            if confidence >= 0.22:
                relations.append(
                    GlyphRelation(satellite.id, primary.id, "annotates", confidence)
                )

    for source in large:
        if len(source.node_ids) < 2:
            continue
        candidates = []
        for target in large:
            if len(target.node_ids) < 2:
                continue
            gap = target.bbox.y1 - source.bbox.y2
            if gap < scale * 0.70 or gap > scale * 3.5:
                continue
            horizontal_error = abs(source.center_x - target.center_x)
            if horizontal_error > scale * 1.25:
                continue
            confidence = exp(-gap / max(scale * 1.5, 1.0)) * exp(
                -horizontal_error / max(scale, 1.0)
            )
            candidates.append((confidence, target))
        if candidates:
            confidence, target = max(candidates, key=lambda item: item[0])
            if confidence >= 0.08:
                relations.append(
                    GlyphRelation(source.id, target.id, "next_record", confidence)
                )
    return relations


def detect_glyph_graph(image: Image.Image, region: Box | None = None) -> GlyphGraph:
    """Build a multi-scale 2-D glyph and relation graph without OCR."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    region = region or Box(0, 0, width, height)
    binary = _binarize(rgb)[region.y1 : region.y2, region.x1 : region.x2]
    initial_scale = _estimate_glyph_scale(binary)
    ink = _remove_rules_for_glyphs(binary, initial_scale)
    scale = _estimate_glyph_scale(ink)
    bands = _horizontal_bands(binary, scale)
    nodes = _component_nodes(ink, region, scale)
    _assign_scale_classes(nodes, bands, region, scale)
    _smooth_scale_classes(nodes, bands, region, scale)
    continuation = _continuation_relations(nodes, scale, bands, region)
    chains = _make_chains(nodes, continuation)
    chains = _split_drifting_chains(chains, nodes, scale)
    node_to_chain = {
        node_id: chain.id for chain in chains for node_id in chain.node_ids
    }
    continuation = [
        relation
        for relation in continuation
        if node_to_chain[relation.source] == node_to_chain[relation.target]
    ]
    chains = _split_annotation_chains(chains, nodes, scale)
    relations = [*continuation, *_structural_relations(chains, scale)]
    return GlyphGraph(width, height, region, scale, nodes, chains, relations)


def draw_glyph_graph(image: Image.Image, graph: GlyphGraph) -> Image.Image:
    """Draw local scale classes, chains, and typed structural relations."""
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    node_colors = {
        "fragment": "#ced4da",
        "small": "#f77f00",
        "medium": "#6c757d",
        "large": "#0077b6",
    }
    for node in graph.nodes:
        draw.rectangle(node.bbox.as_list(), outline=node_colors[node.scale_class], width=1)
    chain_colors = {
        "fragment": "#ced4da",
        "small": "#ff9f1c",
        "medium": "#8338ec",
        "large": "#00b4d8",
    }
    by_chain = {chain.id: chain for chain in graph.chains}
    for chain in graph.chains:
        if len(chain.node_ids) >= 2:
            draw.rectangle(chain.bbox.as_list(), outline=chain_colors[chain.scale_class], width=2)
    for relation in graph.relations:
        if relation.relation == "continues_to":
            continue
        source = by_chain[relation.source]
        target = by_chain[relation.target]
        if relation.relation == "annotates":
            color = "#e76f51"
            source_y = (source.bbox.y1 + source.bbox.y2) / 2
            target_y = min(max(source_y, target.bbox.y1), target.bbox.y2)
            endpoints = (source.center_x, source_y, target.center_x, target_y)
        else:
            color = "#2a9d8f"
            endpoints = (
                source.center_x,
                source.bbox.y2,
                target.center_x,
                target.bbox.y1,
            )
        draw.line(endpoints, fill=color, width=2)
    return output
