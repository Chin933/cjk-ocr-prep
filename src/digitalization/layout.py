"""Hierarchical page-layout and reading-order detection.

This module deliberately has no OCR dependency.  It interprets frames, ruling
lines and whitespace as a recursive partition of the page.  The resulting tree
retains structure that a flat list of detector boxes cannot represent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from PIL import Image

Axis = Literal["x", "y", "none"]
NodeKind = Literal["spread", "page", "region", "text", "empty"]


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class LayoutNode:
    id: str
    kind: NodeKind
    bbox: Box
    split_axis: Axis = "none"
    children: list["LayoutNode"] = field(default_factory=list)
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    order: int | None = None

    def leaves(self, include_empty: bool = False) -> list["LayoutNode"]:
        if not self.children:
            if include_empty or self.kind != "empty":
                return [self]
            return []
        result: list[LayoutNode] = []
        for child in self.children:
            result.extend(child.leaves(include_empty=include_empty))
        return result

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "bbox": self.bbox.as_list(),
            "split_axis": self.split_axis,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "order": self.order,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class LayoutDocument:
    width: int
    height: int
    root: LayoutNode

    @property
    def reading_order(self) -> list[LayoutNode]:
        return self.root.leaves(include_empty=False)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "reading_order": [node.id for node in self.reading_order],
            "root": self.root.to_dict(),
        }


@dataclass(frozen=True)
class LayoutConfig:
    max_depth: int = 6
    frame_inset: int = 4
    min_region_width: int = 24
    min_region_height: int = 40
    min_text_density: float = 0.012
    horizontal_min_span: float = 0.58
    vertical_min_span: float = 0.10
    horizontal_min_region_fraction: float = 0.07
    whitespace_max_ink: float = 0.012
    whitespace_min_width: float = 0.018
    spread_min_aspect: float = 1.15
    text_alignment_min_strength: float = 0.34
    text_alignment_min_lanes: int = 3


def _binarize(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]


def _runs(mask: np.ndarray, merge_gap: int = 3) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    result: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if result and start - result[-1][1] <= merge_gap:
            result[-1] = (result[-1][0], int(end))
        else:
            result.append((int(start), int(end)))
    return result


def _line_profile(binary: np.ndarray, axis: Axis, kernel_fraction: float) -> np.ndarray:
    height, width = binary.shape
    if axis == "y":
        length = max(24, int(width * kernel_fraction))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return (opened > 0).mean(axis=1)
    length = max(24, int(height * kernel_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return (opened > 0).mean(axis=0)


def _frame_box(binary: np.ndarray, inset: int) -> Box:
    height, width = binary.shape
    horizontal = _line_profile(binary, "y", 0.28)
    vertical = _line_profile(binary, "x", 0.28)
    h_runs = _runs(horizontal >= 0.42, merge_gap=5)
    v_runs = _runs(vertical >= 0.42, merge_gap=5)
    if len(h_runs) < 2 or len(v_runs) < 2:
        return Box(0, 0, width, height)
    box = Box(
        v_runs[0][1] + inset,
        h_runs[0][1] + inset,
        v_runs[-1][0] - inset,
        h_runs[-1][0] - inset,
    )
    if box.width < width * 0.5 or box.height < height * 0.5:
        return Box(0, 0, width, height)
    return box


def _spread_cut(
    binary: np.ndarray, frame: Box, config: LayoutConfig
) -> int | None:
    crop = binary[frame.y1 : frame.y2, frame.x1 : frame.x2]
    height, width = crop.shape
    profile = _line_profile(crop, "x", 0.55)
    candidates = []
    for start, end in _runs(profile >= 0.35, merge_gap=5):
        center = (start + end) // 2
        if width * 0.38 <= center <= width * 0.62:
            candidates.append((center, float(profile[start:end].max())))
    if candidates:
        strong = [item for item in candidates if item[1] >= 0.88]
        chosen = min(strong or candidates, key=lambda item: abs(item[0] - width / 2))
        return frame.x1 + chosen[0]

    ink = (crop > 0).mean(axis=0).astype(np.float32)
    smoothed = cv2.GaussianBlur(ink.reshape(1, -1), (31, 1), 0).ravel()
    lo, hi = int(width * 0.43), int(width * 0.57)
    center = lo + int(np.argmin(smoothed[lo:hi]))
    if smoothed[center] <= 0.018:
        return frame.x1 + center
    if frame.width / frame.height >= config.spread_min_aspect:
        # Wide archive scans in the target corpus are two-page spreads even
        # when damage or marginalia obscures the center seam.
        return frame.x1 + frame.width // 2
    return None


def _separator_positions(
    crop: np.ndarray, axis: Axis, config: LayoutConfig, strict: bool = False
) -> list[tuple[int, str, float]]:
    height, width = crop.shape
    if axis == "y":
        profile = _line_profile(crop, axis, 0.72 if strict else 0.35)
        threshold = 0.82 if strict else config.horizontal_min_span
        length = height
    else:
        profile = _line_profile(crop, axis, 0.25)
        threshold = config.vertical_min_span
        length = width

    margin = max(8, int(length * 0.025))
    found: list[tuple[int, str, float]] = []
    for start, end in _runs(profile >= threshold, merge_gap=4):
        center = (start + end) // 2
        if strict and axis == "y":
            if end - start > max(7, int(height * 0.012)):
                continue
            band = crop[start:end] > 0
            occupied_columns = np.flatnonzero(band.any(axis=0))
            edge_tolerance = max(3, int(width * 0.03))
            if (
                occupied_columns.size == 0
                or occupied_columns[0] > edge_tolerance
                or occupied_columns[-1] < width - edge_tolerance - 1
            ):
                continue
            context = max(6, int(width * 0.08))
            above = crop[max(0, start - context) : start] > 0
            below = crop[end : min(height, end + context)] > 0
            if above.size and below.size:
                crossing = (above.any(axis=0) & below.any(axis=0)).mean()
                if crossing > 0.12:
                    continue
        if margin < center < length - margin:
            found.append((center, "ruling_line", float(profile[start:end].max())))

    if axis == "x":
        ink = (crop > 0).mean(axis=0)
        min_gap = max(8, int(width * config.whitespace_min_width))
        for start, end in _runs(ink <= config.whitespace_max_ink, merge_gap=2):
            if end - start < min_gap:
                continue
            center = (start + end) // 2
            if margin < center < width - margin:
                found.append((center, "whitespace", 0.75))

    found.sort(key=lambda item: item[0])
    merged: list[tuple[int, str, float]] = []
    for item in found:
        if merged and item[0] - merged[-1][0] <= 8:
            previous = merged[-1]
            merged[-1] = item if item[2] > previous[2] else previous
        else:
            merged.append(item)
    return merged


def _infer_pitch_boundaries(
    crop: np.ndarray,
    positions: list[tuple[int, str, float]],
    minimum: int,
) -> list[tuple[int, str, float]]:
    """Fill large gaps when several ruling lines establish a stable pitch.

    Some scans retain only alternating or partial rules.  We extrapolate only
    across gaps larger than two ordinary cells and only on ink-dense regions,
    which avoids subdividing common double-width title cells.
    """
    ruling = [item[0] for item in positions if item[1] == "ruling_line"]
    if len(ruling) < 3 or float((crop > 0).mean()) < 0.075:
        return positions
    gaps = np.diff(ruling)
    ordinary = gaps[(gaps >= minimum) & (gaps <= np.percentile(gaps, 60))]
    if ordinary.size < 2:
        return positions
    pitch = float(np.median(ordinary))
    if pitch < minimum:
        return positions

    length = crop.shape[1]
    anchors = [0] + ruling + [length]
    inferred: list[tuple[int, str, float]] = []
    for start, end in zip(anchors, anchors[1:]):
        gap = end - start
        if gap <= 2.3 * pitch:
            continue
        count = int(round(gap / pitch)) - 1
        for index in range(1, count + 1):
            position = int(round(start + index * gap / (count + 1)))
            if minimum <= position <= length - minimum:
                inferred.append((position, "inferred_pitch", 0.55))
    return sorted(positions + inferred, key=lambda item: item[0])


def _text_alignment_boundaries(
    crop: np.ndarray,
    page_width: int,
    config: LayoutConfig,
) -> list[tuple[int, str, float]]:
    """Infer column boundaries from repeated vertical text alignment.

    This is the fallback for pages whose printed ruling lines have disappeared.
    It deliberately uses only geometry: long rules are removed, the horizontal
    ink profile is autocorrelated, and cuts are placed between recurring text
    lanes.  OCR output never enters the decision.
    """
    height, width = crop.shape
    minimum_pitch = max(28, int(page_width * 0.025))
    maximum_pitch = min(320, int(page_width * 0.42), width // 2)
    if (
        width < minimum_pitch * config.text_alignment_min_lanes
        or height < config.min_region_height
        or maximum_pitch <= minimum_pitch
        or float((crop > 0).mean()) < config.min_text_density * 1.5
    ):
        return []

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(24, int(width * 0.22)), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(24, int(height * 0.22)))
    )
    rules = cv2.bitwise_or(
        cv2.morphologyEx(crop, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(crop, cv2.MORPH_OPEN, vertical_kernel),
    )
    text_only = cv2.bitwise_and(crop, cv2.bitwise_not(rules))
    profile = (text_only > 0).mean(axis=0).astype(np.float32)
    blur_width = max(5, min(19, int(width * 0.012)))
    if blur_width % 2 == 0:
        blur_width += 1
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (blur_width, 1), 0).ravel()
    centered = smooth - float(smooth.mean())
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8:
        return []

    correlations = np.array(
        [
            float(np.dot(centered[:-lag], centered[lag:]))
            / max(1e-8, float(np.linalg.norm(centered[:-lag]) * np.linalg.norm(centered[lag:])))
            for lag in range(minimum_pitch, maximum_pitch + 1)
        ],
        dtype=np.float32,
    )
    base_maximum = min(180, int(page_width * 0.18), maximum_pitch)
    base_count = max(1, base_maximum - minimum_pitch + 1)
    base_is_reliable = (
        float(correlations[:base_count].max()) >= config.text_alignment_min_strength
    )
    if not base_is_reliable and float((text_only > 0).mean()) >= 0.12:
        # Dense mixed-size passages create convincing long-lag harmonics. A
        # wide-pitch fallback is reserved for genuinely sparse 2--3-column
        # layouts; dense pages continue through ruling/whitespace evidence.
        return []
    search_count = base_count if base_is_reliable else len(correlations)
    searched = correlations[:search_count]
    local_peaks = [
        index
        for index in range(1, len(searched) - 1)
        if searched[index] >= searched[index - 1]
        and searched[index] >= searched[index + 1]
    ]
    strongest = float(searched.max())
    near_strongest = [
        index for index in local_peaks if searched[index] >= strongest - 0.06
    ]
    peak_offset = (
        int(np.argmax(searched))
        if base_is_reliable
        else min(near_strongest, default=int(np.argmax(searched)))
    )
    pitch = minimum_pitch + peak_offset
    strength = float(correlations[peak_offset])
    if strength < config.text_alignment_min_strength:
        return []

    # Locate the phase of the recurring ink lanes.  A small window makes this
    # robust to glyph shape while still distinguishing lanes from white gaps.
    radius = max(2, int(pitch * 0.12))
    phase_scores: list[float] = []
    phase_samples: list[list[float]] = []
    for phase in range(pitch):
        samples = [
            float(smooth[max(0, x - radius) : min(width, x + radius + 1)].max())
            for x in range(phase, width, pitch)
        ]
        phase_samples.append(samples)
        phase_scores.append(float(np.mean(samples)) if samples else 0.0)
    phase = int(np.argmax(phase_scores))
    centers = list(range(phase, width, pitch))
    lane_values = phase_samples[phase]
    if len(centers) < config.text_alignment_min_lanes:
        return []

    active_threshold = max(float(np.percentile(smooth, 65)), float(smooth.max()) * 0.22)
    occupied = [value >= active_threshold for value in lane_values]
    if sum(occupied) < config.text_alignment_min_lanes or np.mean(occupied) < 0.55:
        return []

    cuts = [
        (int(round((left + right) / 2)), "text_alignment", strength)
        for left, right in zip(centers, centers[1:])
        if occupied[centers.index(left)] or occupied[centers.index(right)]
    ]
    return _valid_boundaries(cuts, width, config.min_region_width)


def _valid_boundaries(
    positions: list[tuple[int, str, float]], length: int, minimum: int
) -> list[tuple[int, str, float]]:
    result: list[tuple[int, str, float]] = []
    previous = 0
    for item in positions:
        if item[0] - previous < minimum or length - item[0] < minimum:
            continue
        result.append(item)
        previous = item[0]
    return result


def _merge_boundaries(
    positions: list[tuple[int, str, float]], tolerance: int = 8
) -> list[tuple[int, str, float]]:
    """Deduplicate nearby cuts while retaining the strongest evidence."""
    merged: list[tuple[int, str, float]] = []
    for item in sorted(positions, key=lambda value: value[0]):
        if merged and item[0] - merged[-1][0] <= tolerance:
            if item[2] > merged[-1][2]:
                merged[-1] = item
        else:
            merged.append(item)
    return merged


def _leaf_kind(crop: np.ndarray, config: LayoutConfig) -> NodeKind:
    if crop.size == 0:
        return "empty"
    # Ignore a thin rim so frame/ruling ink does not turn a blank cell into text.
    rim_y = max(1, int(crop.shape[0] * 0.015))
    rim_x = max(1, int(crop.shape[1] * 0.03))
    inner = crop[rim_y:-rim_y or None, rim_x:-rim_x or None]
    density = float((inner > 0).mean()) if inner.size else 0.0
    return "text" if density >= config.min_text_density else "empty"


def _split_partial_subcolumns(
    node: LayoutNode,
    binary: np.ndarray,
    crop: np.ndarray,
    box: Box,
    config: LayoutConfig,
    page_width: int,
    next_id,
) -> bool:
    """Split a major column where a local passage changes into two small lanes."""
    height, width = crop.shape
    if not (
        page_width * 0.07 <= width <= page_width * 0.16
        and height >= max(160, width * 2)
    ):
        return False

    window = max(60, int(width * 0.90))
    center_radius = max(3, int(width * 0.05))
    ink = (crop > 0).astype(np.float32)
    cumulative = np.vstack(
        [np.zeros((1, width), dtype=np.float32), np.cumsum(ink, axis=0)]
    )
    rows = np.arange(height)
    starts = np.maximum(0, rows - window // 2)
    ends = np.minimum(height, rows + window // 2)
    profiles = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    middle = width // 2
    center_ink = profiles[
        :, middle - center_radius : middle + center_radius + 1
    ].mean(axis=1)
    left_ink = profiles[:, int(width * 0.08) : int(width * 0.42)].mean(axis=1)
    right_ink = profiles[:, int(width * 0.58) : int(width * 0.92)].mean(axis=1)
    side_ink = (left_ink + right_ink) / 2
    active = (
        (np.minimum(left_ink, right_ink) > 0.06)
        & (center_ink / np.maximum(side_ink, 1e-6) < 0.68)
    ).astype(np.uint8)

    run_length = max(70, int(width * 0.78))
    kernel = np.ones((run_length, 1), np.uint8)
    active = cv2.morphologyEx(active.reshape(-1, 1), cv2.MORPH_CLOSE, kernel)
    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, kernel).ravel()
    runs = [
        (start, end)
        for start, end in _runs(active > 0, merge_gap=max(8, width // 8))
        if end - start >= run_length
    ]
    if not runs:
        return False

    y_boundaries = sorted({0, height, *(point for run in runs for point in run)})
    children: list[LayoutNode] = []
    for start, end in zip(y_boundaries, y_boundaries[1:]):
        segment_box = Box(box.x1, box.y1 + start, box.x2, box.y1 + end)
        segment_crop = binary[
            segment_box.y1 : segment_box.y2, segment_box.x1 : segment_box.x2
        ]
        segment = LayoutNode(next_id("region"), "region", segment_box)
        is_subcolumn_band = any(start >= lo and end <= hi for lo, hi in runs)
        if is_subcolumn_band:
            profile = (segment_crop > 0).mean(axis=0).astype(np.float32)
            smooth = cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).ravel()
            lo, hi = int(width * 0.40), int(width * 0.60)
            middle = lo + int(np.argmin(smooth[lo:hi]))
            right_box = Box(box.x1 + middle, segment_box.y1, box.x2, segment_box.y2)
            left_box = Box(box.x1, segment_box.y1, box.x1 + middle, segment_box.y2)
            segment.split_axis = "x"
            segment.evidence = ["partial_subcolumn_alignment"]
            for child_box in (right_box, left_box):
                child_crop = binary[
                    child_box.y1 : child_box.y2, child_box.x1 : child_box.x2
                ]
                segment.children.append(
                    LayoutNode(
                        next_id("region"),
                        _leaf_kind(child_crop, config),
                        child_box,
                        confidence=0.62,
                        evidence=["partial_subcolumn_alignment"],
                    )
                )
        else:
            segment.kind = _leaf_kind(segment_crop, config)
            segment.evidence = ["content_density"]
        children.append(segment)

    node.split_axis = "y"
    node.children = children
    node.evidence = ["partial_subcolumn_alignment"]
    node.confidence = 0.62
    return True


def _partition(
    binary: np.ndarray,
    box: Box,
    config: LayoutConfig,
    depth: int,
    next_id,
    page_width: int,
    page_guides: list[tuple[int, str, float]],
    last_axis: Axis = "none",
) -> LayoutNode:
    crop = binary[box.y1 : box.y2, box.x1 : box.x2]
    node = LayoutNode(next_id("region"), "region", box)
    if depth >= config.max_depth:
        node.kind = _leaf_kind(crop, config)
        node.evidence = ["max_depth"]
        return node

    horizontal: list[tuple[int, str, float]] = []
    if (
        last_axis != "y"
        and box.width >= page_width * config.horizontal_min_region_fraction
    ):
        horizontal = _valid_boundaries(
            _separator_positions(
                crop,
                "y",
                config,
                strict=box.width < page_width * 0.32,
            ),
            box.height,
            config.min_region_height,
        )
    vertical: list[tuple[int, str, float]] = []
    if last_axis != "x":
        vertical_positions = _separator_positions(crop, "x", config)
        vertical_positions = _infer_pitch_boundaries(
            crop, vertical_positions, config.min_region_width
        )
        local_alignment = _text_alignment_boundaries(crop, page_width, config)
        short_band = box.height < page_width * 0.35
        if box.width >= page_width * 0.25 and not local_alignment and not short_band:
            vertical_positions = _merge_boundaries(
                vertical_positions
                + [
                    (x - box.x1, evidence, confidence)
                    for x, evidence, confidence in page_guides
                    if box.x1 + config.min_region_width
                    <= x
                    <= box.x2 - config.min_region_width
                ]
            )
        if local_alignment:
            # Local evidence may reveal a finer subcolumn rhythm than the
            # page-wide major-column guide.  Keeping both scales is essential
            # for nested historical layouts.
            vertical_positions = _merge_boundaries(
                vertical_positions + local_alignment
            )
        vertical = _valid_boundaries(
            vertical_positions,
            box.width,
            config.min_region_width,
        )

    # Long horizontal rules define major sections.  Inside those sections,
    # vertical rulings and whitespace recover columns from right to left.
    axis: Axis
    cuts: list[tuple[int, str, float]]
    if horizontal:
        axis, cuts = "y", horizontal
    elif vertical:
        axis, cuts = "x", vertical
    else:
        aligned = []
        if last_axis != "x":
            aligned = _text_alignment_boundaries(crop, page_width, config)
        if aligned:
            axis, cuts = "x", aligned
        else:
            if _split_partial_subcolumns(
                node, binary, crop, box, config, page_width, next_id
            ):
                return node
            node.kind = _leaf_kind(crop, config)
            node.evidence = ["content_density"]
            return node

    node.split_axis = axis
    boundaries = [0] + [cut[0] for cut in cuts] + [box.height if axis == "y" else box.width]
    evidence = sorted({cut[1] for cut in cuts})
    children: list[LayoutNode] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if axis == "y":
            child_box = Box(box.x1, box.y1 + start, box.x2, box.y1 + end)
        else:
            child_box = Box(box.x1 + start, box.y1, box.x1 + end, box.y2)
        child = _partition(
            binary,
            child_box,
            config,
            depth + 1,
            next_id,
            page_width,
            page_guides,
            axis,
        )
        child.evidence = sorted(set(child.evidence + evidence))
        children.append(child)
    if axis == "x":
        children.reverse()
    node.children = children
    node.evidence = evidence
    node.confidence = min(cut[2] for cut in cuts)
    return node


def detect_layout(
    image: Image.Image, config: LayoutConfig | None = None
) -> LayoutDocument:
    """Detect a hierarchical layout tree and its implied reading order."""
    config = config or LayoutConfig()
    rgb = image.convert("RGB")
    width, height = rgb.size
    binary = _binarize(rgb)
    frame = _frame_box(binary, config.frame_inset)
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter:04d}"

    root = LayoutNode(next_id("spread"), "spread", frame)
    center = _spread_cut(binary, frame, config)
    if center is not None:
        right_box = Box(center, frame.y1, frame.x2, frame.y2)
        left_box = Box(frame.x1, frame.y1, center, frame.y2)
        pages = [right_box, left_box]
        root.split_axis = "x"
        root.evidence = ["center_gutter"]
    else:
        pages = [frame]

    for page_box in pages:
        page_crop = binary[page_box.y1 : page_box.y2, page_box.x1 : page_box.x2]
        local_guides = _text_alignment_boundaries(page_crop, page_box.width, config)
        if not local_guides:
            page_height = page_crop.shape[0]
            bands = [
                page_crop[: int(page_height * 0.55)],
                page_crop[int(page_height * 0.40) :],
            ]
            candidates = [
                _text_alignment_boundaries(band, page_box.width, config)
                for band in bands
            ]
            local_guides = max(
                candidates,
                key=lambda values: (len(values), sum(item[2] for item in values)),
            )
        page_guides = [
            (page_box.x1 + x, "page_text_alignment", confidence)
            for x, _, confidence in local_guides
        ]
        page = _partition(
            binary,
            page_box,
            config,
            0,
            next_id,
            page_box.width,
            page_guides,
        )
        page.kind = "page"
        root.children.append(page)

    for order, leaf in enumerate(root.leaves(include_empty=False)):
        leaf.order = order
    return LayoutDocument(width=width, height=height, root=root)


def draw_layout(image: Image.Image, document: LayoutDocument) -> Image.Image:
    """Return an overlay showing hierarchy and final reading order."""
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    colors = {
        "spread": "#168aad",
        "page": "#1a759f",
        "region": "#7b2cbf",
        "text": "#f77f00",
        "empty": "#adb5bd",
    }

    def visit(node: LayoutNode, depth: int = 0) -> None:
        box = node.bbox
        draw.rectangle(
            box.as_list(),
            outline=colors[node.kind],
            width=max(1, 4 - min(depth, 3)),
        )
        if node.order is not None:
            draw.rectangle((box.x1 + 2, box.y1 + 2, box.x1 + 42, box.y1 + 24), fill="white")
            draw.text((box.x1 + 5, box.y1 + 4), str(node.order), fill="#d00000")
        for child in node.children:
            visit(child, depth + 1)

    visit(document.root)
    return output
