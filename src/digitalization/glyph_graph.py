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
    _vertical_frame_bounds,
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
class GlyphRecord:
    id: str
    bbox: Box
    primary_chain_id: str
    annotation_chain_ids: list[str]
    geometry: str
    order: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "primary_chain_id": self.primary_chain_id,
            "annotation_chain_ids": self.annotation_chain_ids,
            "geometry": self.geometry,
            "order": self.order,
            "confidence": round(self.confidence, 4),
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
    records: list[GlyphRecord]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "region": self.region.as_list(),
            "glyph_scale": round(self.glyph_scale, 2),
            "reading_order": [
                record.id for record in sorted(self.records, key=lambda item: item.order)
            ],
            "nodes": [node.to_dict() for node in self.nodes],
            "chains": [chain.to_dict() for chain in self.chains],
            "relations": [relation.to_dict() for relation in self.relations],
            "records": [record.to_dict() for record in self.records],
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
                # A missing large-glyph slot usually marks a new roster entry,
                # even when the neighbouring glyphs themselves are oversized.
                # Keep the page-scale cap so local glyph size cannot bridge it.
                if vertical_gap > min(
                    max(scale * 1.55, source.local_scale * 1.8),
                    scale * 1.70,
                ):
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


def _demote_off_lattice_primaries(
    chains: list[GlyphChain],
    nodes: list[GlyphNode],
    bands: list[tuple[int, int]],
    region: Box,
    scale: float,
) -> None:
    """Use repeated primary pitch to reject large-looking fused annotation lanes."""
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        band_nodes = [node for node in nodes if top <= node.center_y < bottom]
        small_sizes = [
            node.local_scale for node in band_nodes if node.scale_class == "small"
        ]
        large_sizes = [
            node.local_scale for node in band_nodes if node.scale_class == "large"
        ]
        if len(small_sizes) < 8 or len(large_sizes) < 4:
            continue
        median_small = float(np.median(small_sizes))
        scale_ratio = median_small / float(np.median(large_sizes))
        if not 0.45 <= scale_ratio <= 0.82 or median_small < scale * 0.50:
            continue
        primaries = sorted(
            (
                chain
                for chain in chains
                if chain.scale_class == "large"
                and len(chain.node_ids) >= 2
                and max(0, min(bottom, chain.bbox.y2) - max(top, chain.bbox.y1))
                / max(1, chain.bbox.height)
                >= 0.65
            ),
            key=lambda item: item.center_x,
        )
        gaps = [
            right.center_x - left.center_x
            for left, right in zip(primaries, primaries[1:])
        ]
        pitch_gaps = [gap for gap in gaps if scale * 2.25 <= gap <= scale * 4.2]
        if len(pitch_gaps) < 3:
            continue
        pitch = float(np.median(pitch_gaps))
        best: tuple[int, float, set[str]] | None = None
        for anchor in primaries:
            selected = set()
            residual_sum = 0.0
            for candidate in primaries:
                steps = round((candidate.center_x - anchor.center_x) / pitch)
                residual = abs(candidate.center_x - anchor.center_x - steps * pitch)
                if residual <= pitch * 0.24:
                    selected.add(candidate.id)
                    residual_sum += residual
            score = (len(selected), -residual_sum, selected)
            if best is None or score[:2] > best[:2]:
                best = score
        if best is None:
            continue
        selected = best[2]
        if len(selected) < 4 or len(selected) / max(1, len(primaries)) < 0.65:
            continue
        for primary in primaries:
            if primary.id not in selected:
                primary.scale_class = "small"


def _structural_relations(
    chains: list[GlyphChain], scale: float
) -> list[GlyphRelation]:
    relations = []
    large = [
        chain
        for chain in chains
        if chain.scale_class == "large" and len(chain.node_ids) >= 2
    ]
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
            if satellite.bbox.y2 < primary.bbox.y1:
                vertical_gap = primary.bbox.y1 - satellite.bbox.y2
            elif satellite.bbox.y1 > primary.bbox.y2:
                vertical_gap = satellite.bbox.y1 - primary.bbox.y2
            else:
                vertical_gap = 0
            horizontal_gap = max(
                0,
                max(satellite.bbox.x1, primary.bbox.x1)
                - min(satellite.bbox.x2, primary.bbox.x2),
            )
            if vertical_gap > scale * 1.4 or horizontal_gap > scale * 2.4:
                continue
            distance = abs(satellite.center_x - primary.center_x)
            confidence = (
                (0.70 + 0.30 * overlap_ratio)
                * exp(-distance / max(scale * 1.6, 1.0))
                * exp(-vertical_gap / max(scale * 1.2, 1.0))
            )
            candidates.append((confidence, primary))
        if candidates:
            confidence, primary = max(candidates, key=lambda item: item[0])
            if confidence >= 0.18:
                relations.append(
                    GlyphRelation(satellite.id, primary.id, "annotates", confidence)
                )

    return relations


def _make_records(
    chains: list[GlyphChain],
    nodes: list[GlyphNode],
    relations: list[GlyphRelation],
    bands: list[tuple[int, int]],
    region: Box,
    scale: float,
    frame_bounds: tuple[int, int],
) -> list[GlyphRecord]:
    """Build record hypotheses only in bands with a genuine two-font mixture."""
    by_chain = {chain.id: chain for chain in chains}
    attached: dict[str, list[str]] = {}
    for relation in relations:
        if relation.relation == "annotates":
            attached.setdefault(relation.target, []).append(relation.source)
    records = []
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        band_nodes = [node for node in nodes if top <= node.center_y < bottom]
        small_sizes = [
            node.local_scale for node in band_nodes if node.scale_class == "small"
        ]
        large_sizes = [
            node.local_scale for node in band_nodes if node.scale_class == "large"
        ]
        if len(small_sizes) < 8 or len(large_sizes) < 4:
            continue
        median_small = float(np.median(small_sizes))
        scale_ratio = median_small / float(np.median(large_sizes))
        # Punctuation may form a coherent narrow lane but is much smaller than
        # a plausible character.  A mixed-size record band needs both a clear
        # relative separation and a small mode that remains character-sized.
        if not 0.45 <= scale_ratio <= 0.82 or median_small < scale * 0.50:
            continue
        primaries = [
            chain
            for chain in chains
            if chain.scale_class == "large"
            and len(chain.node_ids) >= 2
            and max(0, min(bottom, chain.bbox.y2) - max(top, chain.bbox.y1))
            / max(1, chain.bbox.height)
            >= 0.65
        ]
        band_records = []
        for primary in primaries:
            annotation_ids = sorted(set(attached.get(primary.id, [])))
            items = [primary, *(by_chain[item] for item in annotation_ids)]
            bbox = Box(
                min(item.bbox.x1 for item in items),
                min(item.bbox.y1 for item in items),
                max(item.bbox.x2 for item in items),
                max(item.bbox.y2 for item in items),
            )
            confidence = min(
                1.0,
                0.35
                + 0.12 * min(4, len(primary.node_ids))
                + 0.05 * min(3, len(annotation_ids)),
            )
            band_records.append(
                GlyphRecord(
                    "",
                    bbox,
                    primary.id,
                    annotation_ids,
                    "compound",
                    0,
                    confidence,
                )
            )

        ordered_primaries = sorted(primaries, key=lambda item: item.center_x)
        primary_lanes: list[list[GlyphChain]] = []
        for primary in ordered_primaries:
            if (
                primary_lanes
                and abs(
                    primary.center_x
                    - float(np.median([item.center_x for item in primary_lanes[-1]]))
                )
                <= scale * 0.82
            ):
                primary_lanes[-1].append(primary)
            else:
                primary_lanes.append([primary])
        lane_centers = [
            float(np.median([item.center_x for item in lane]))
            for lane in primary_lanes
        ]
        gaps = [
            right - left for left, right in zip(lane_centers, lane_centers[1:])
        ]
        if len(gaps) >= 3:
            pitch = float(np.median(gaps))
            regular = sum(abs(gap - pitch) <= pitch * 0.25 for gap in gaps)
            repeated_lanes = sum(len(lane) > 1 for lane in primary_lanes)
            if (
                regular / len(gaps) >= 0.80
                and repeated_lanes / len(primary_lanes) <= 0.20
            ):
                left_frame, right_frame = frame_bounds
                boundaries = [float(region.x1 + left_frame)]
                boundaries.extend(
                    (left + right) / 2
                    for left, right in zip(lane_centers, lane_centers[1:])
                )
                boundaries.append(float(region.x1 + right_frame))
                by_primary = {
                    item.primary_chain_id: item for item in band_records
                }
                cell_records = []
                for index, lane in enumerate(primary_lanes):
                    primary = max(lane, key=lambda item: len(item.node_ids))
                    record = by_primary[primary.id]
                    record.annotation_chain_ids = sorted(
                        {
                            annotation_id
                            for item in lane
                            for annotation_id in by_primary[
                                item.id
                            ].annotation_chain_ids
                        }
                    )
                    record.bbox = Box(
                        int(round(boundaries[index])),
                        top,
                        int(round(boundaries[index + 1])),
                        bottom,
                    )
                    record.geometry = "cell"
                    cell_records.append(record)
                band_records = cell_records

        lanes: list[list[GlyphRecord]] = []
        for record in sorted(
            band_records,
            key=lambda item: -by_chain[item.primary_chain_id].center_x,
        ):
            center = by_chain[record.primary_chain_id].center_x
            candidates = [
                (abs(center - float(np.median([
                    by_chain[item.primary_chain_id].center_x for item in lane
                ]))), index)
                for index, lane in enumerate(lanes)
            ]
            if candidates and min(candidates)[0] <= scale * 0.82:
                lanes[min(candidates)[1]].append(record)
            else:
                lanes.append([record])
        lanes.sort(
            key=lambda lane: -float(
                np.median([by_chain[item.primary_chain_id].center_x for item in lane])
            )
        )
        for lane in lanes:
            records.extend(sorted(lane, key=lambda item: item.bbox.y1))

    for order, record in enumerate(records):
        record.id = f"record_{order:05d}"
        record.order = order
    return records


def _record_relations(
    records: list[GlyphRecord],
    chains: list[GlyphChain],
    bands: list[tuple[int, int]],
    region: Box,
    scale: float,
) -> list[GlyphRelation]:
    """Connect consecutive compound records inside one local vertical lane."""
    by_chain = {chain.id: chain for chain in chains}
    relations = []
    for start, end in bands:
        top, bottom = region.y1 + start, region.y1 + end
        band_records = [
            record
            for record in records
            if record.geometry == "compound"
            and top
            <= (
                by_chain[record.primary_chain_id].bbox.y1
                + by_chain[record.primary_chain_id].bbox.y2
            )
            / 2
            < bottom
        ]
        lanes: list[list[GlyphRecord]] = []
        for record in sorted(
            band_records,
            key=lambda item: -by_chain[item.primary_chain_id].center_x,
        ):
            center = by_chain[record.primary_chain_id].center_x
            candidates = [
                (
                    abs(
                        center
                        - float(
                            np.median(
                                [
                                    by_chain[item.primary_chain_id].center_x
                                    for item in lane
                                ]
                            )
                        )
                    ),
                    index,
                )
                for index, lane in enumerate(lanes)
            ]
            if candidates and min(candidates)[0] <= scale * 0.82:
                lanes[min(candidates)[1]].append(record)
            else:
                lanes.append([record])
        for lane in lanes:
            lane.sort(key=lambda item: by_chain[item.primary_chain_id].bbox.y1)
            for source_record, target_record in zip(lane, lane[1:]):
                source = by_chain[source_record.primary_chain_id]
                target = by_chain[target_record.primary_chain_id]
                gap = target.bbox.y1 - source.bbox.y2
                if gap < scale * 0.70 or gap > scale * 4.50:
                    continue
                horizontal_error = abs(source.center_x - target.center_x)
                confidence = exp(-gap / max(scale * 2.5, 1.0)) * exp(
                    -horizontal_error / max(scale, 1.0)
                )
                relations.append(
                    GlyphRelation(
                        source.id,
                        target.id,
                        "next_record",
                        confidence,
                    )
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
    _demote_off_lattice_primaries(chains, nodes, bands, region, scale)
    chains = _split_annotation_chains(chains, nodes, scale)
    relations = [*continuation, *_structural_relations(chains, scale)]
    records = _make_records(
        chains,
        nodes,
        relations,
        bands,
        region,
        scale,
        _vertical_frame_bounds(binary),
    )
    relations.extend(_record_relations(records, chains, bands, region, scale))
    record_by_primary = {record.primary_chain_id: record for record in records}
    annotation_pairs = {
        (annotation_id, record.primary_chain_id)
        for record in records
        for annotation_id in record.annotation_chain_ids
    }
    filtered_relations = []
    for relation in relations:
        if relation.relation == "annotates":
            if (relation.source, relation.target) in annotation_pairs:
                filtered_relations.append(relation)
            continue
        if relation.relation == "next_record":
            source = record_by_primary.get(relation.source)
            target = record_by_primary.get(relation.target)
            if source is None or target is None or source.order >= target.order:
                continue
            source_chain = next(chain for chain in chains if chain.id == relation.source)
            target_chain = next(chain for chain in chains if chain.id == relation.target)
            if abs(source_chain.center_x - target_chain.center_x) <= scale * 0.82:
                filtered_relations.append(relation)
            continue
        filtered_relations.append(relation)
    relations = filtered_relations
    return GlyphGraph(width, height, region, scale, nodes, chains, relations, records)


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
    for record in graph.records:
        primary = by_chain[record.primary_chain_id]
        if record.geometry == "cell":
            draw.rectangle(record.bbox.as_list(), outline="#8338ec", width=2)
        else:
            draw.rectangle(primary.bbox.as_list(), outline="#8338ec", width=3)
        draw.text(
            (primary.bbox.x1 + 2, primary.bbox.y1 + 2),
            f"R{record.order}",
            fill="#6a00f4",
        )
    return output


def draw_record_graph(image: Image.Image, graph: GlyphGraph) -> Image.Image:
    """Draw only inferred records, annotation membership, and record order."""
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    by_chain = {chain.id: chain for chain in graph.chains}
    ordered = sorted(graph.records, key=lambda item: item.order)

    for record in ordered:
        primary = by_chain[record.primary_chain_id]
        if record.geometry == "cell":
            draw.rectangle(record.bbox.as_list(), outline="#6a00f4", width=4)
        else:
            draw.rectangle(primary.bbox.as_list(), outline="#6a00f4", width=4)
            for annotation_id in record.annotation_chain_ids:
                annotation = by_chain[annotation_id]
                draw.rectangle(annotation.bbox.as_list(), outline="#f77f00", width=2)
                annotation_y = (annotation.bbox.y1 + annotation.bbox.y2) / 2
                primary_y = min(max(annotation_y, primary.bbox.y1), primary.bbox.y2)
                annotation_x = (
                    annotation.bbox.x1
                    if annotation.center_x > primary.center_x
                    else annotation.bbox.x2
                )
                primary_x = (
                    primary.bbox.x2
                    if annotation.center_x > primary.center_x
                    else primary.bbox.x1
                )
                draw.line(
                    (annotation_x, annotation_y, primary_x, primary_y),
                    fill="#f77f00",
                    width=2,
                )
        label_y = max(0, primary.bbox.y1 - 13)
        draw.rectangle(
            (primary.bbox.x1, label_y, primary.bbox.x1 + 30, label_y + 13),
            fill="white",
        )
        draw.text((primary.bbox.x1 + 1, label_y), f"R{record.order}", fill="#6a00f4")

    record_by_primary = {record.primary_chain_id: record for record in ordered}
    for relation in graph.relations:
        if relation.relation != "next_record":
            continue
        source_record = record_by_primary.get(relation.source)
        target_record = record_by_primary.get(relation.target)
        if source_record is None or target_record is None:
            continue
        source = by_chain[relation.source]
        target = by_chain[relation.target]
        draw.line(
            (source.center_x, source.bbox.y2, target.center_x, target.bbox.y1),
            fill="#008f5a",
            width=4,
        )
    return output
