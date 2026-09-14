"""Bottom-up content tracks for vertical historical-document layouts.

The slicing layout tree starts from separators.  This companion graph starts
from ink and follows locally stable vertical streams, so a stream can begin or
end without forcing a cut across its whole parent region.  It deliberately
does not recognise characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .layout import Box, _binarize


@dataclass(frozen=True)
class LaneObservation:
    id: str
    bbox: Box
    level: int
    center_x: float
    ink: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "level": self.level,
            "center_x": round(self.center_x, 2),
            "ink": self.ink,
        }


@dataclass
class TextStream:
    id: str
    bbox: Box
    observation_ids: list[str]
    center_x: float
    median_width: float
    confidence: float
    role: str = "text"
    order: int | None = None
    neighbors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "observation_ids": self.observation_ids,
            "center_x": round(self.center_x, 2),
            "median_width": round(self.median_width, 2),
            "confidence": round(self.confidence, 4),
            "role": self.role,
            "order": self.order,
            "neighbors": self.neighbors,
        }


@dataclass(frozen=True)
class ContentEdge:
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
class StreamGroup:
    id: str
    bbox: Box
    stream_ids: list[str]
    order: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.as_list(),
            "stream_ids": self.stream_ids,
            "order": self.order,
        }


@dataclass
class ContentGraph:
    width: int
    height: int
    region: Box
    glyph_scale: float
    observations: list[LaneObservation]
    streams: list[TextStream]
    edges: list[ContentEdge]
    groups: list[StreamGroup]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "region": self.region.as_list(),
            "glyph_scale": round(self.glyph_scale, 2),
            "reading_order": [
                stream.id
                for stream in sorted(
                    (
                        stream
                        for stream in self.streams
                        if stream.role in {"text", "satellite"}
                    ),
                    key=lambda item: item.order if item.order is not None else 10**9,
                )
            ],
            "group_reading_order": [
                group.id for group in sorted(self.groups, key=lambda item: item.order)
            ],
            "observations": [item.to_dict() for item in self.observations],
            "streams": [item.to_dict() for item in self.streams],
            "edges": [item.to_dict() for item in self.edges],
            "groups": [item.to_dict() for item in self.groups],
        }


def _estimate_glyph_scale(ink: np.ndarray) -> float:
    mask = cv2.morphologyEx(
        (ink > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = []
    height, width = ink.shape
    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not (
            area >= 10
            and 3 <= component_width <= width * 0.18
            and 3 <= component_height <= height * 0.18
            and 0.12 <= component_width / component_height <= 8.0
        ):
            continue
        sizes.append(max(component_width, component_height))
    if not sizes:
        return float(max(12, min(height, width) * 0.035))
    # Individual radicals survive as components; the upper quartile is a more
    # stable proxy for character scale than the median on damaged scans.
    return float(np.percentile(sizes, 72))


def _remove_spanning_rules(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(width * 0.18)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(height * 0.18)))),
    )
    return cv2.subtract(binary, cv2.max(horizontal, vertical))


def _runs(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    merged: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def _horizontal_bands(binary: np.ndarray, scale: float) -> list[tuple[int, int]]:
    height, width = binary.shape
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(width * 0.42)), 1)),
    )
    rule_rows = (opened > 0).mean(axis=1) > 0.28
    rules = _runs(rule_rows, merge_gap=max(2, int(scale * 0.08)))
    bands = []
    cursor = 0
    for start, end in rules:
        if start - cursor >= scale * 1.4:
            bands.append((cursor, start))
        cursor = max(cursor, end)
    if height - cursor >= scale * 1.4:
        bands.append((cursor, height))
    return bands or [(0, height)]


def _observe_lanes(ink: np.ndarray, region: Box, scale: float) -> list[LaneObservation]:
    height, width = ink.shape
    window = min(height, max(28, int(scale * 2.4)))
    stride = max(12, window // 2)
    starts = list(range(0, max(1, height - window + 1), stride))
    if not starts or starts[-1] + window < height:
        starts.append(max(0, height - window))
    observations: list[LaneObservation] = []
    for level, start in enumerate(dict.fromkeys(starts)):
        end = min(height, start + window)
        tile = ink[start:end]
        bridge = max(2, int(scale * 0.16))
        joined = cv2.morphologyEx(
            tile,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)),
        )
        profile = (joined > 0).mean(axis=0).astype(np.float32)
        smooth_width = max(3, int(scale * 0.12) | 1)
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_width, 1), 0).ravel()
        positive = profile[profile > 0]
        if positive.size == 0:
            continue
        threshold = max(0.018, float(np.percentile(positive, 60)) * 0.28)
        for left, right in _runs(profile > threshold, merge_gap=max(1, int(scale * 0.12))):
            if right - left < max(3, int(scale * 0.16)):
                continue
            local = ink[start:end, left:right] > 0
            ys, xs = np.nonzero(local)
            if xs.size < max(8, int(scale * scale * 0.025)):
                continue
            x1 = left + int(xs.min())
            x2 = left + int(xs.max()) + 1
            y1 = start + int(ys.min())
            y2 = start + int(ys.max()) + 1
            bbox = Box(region.x1 + x1, region.y1 + y1, region.x1 + x2, region.y1 + y2)
            observations.append(
                LaneObservation(
                    f"observation_{len(observations):04d}",
                    bbox,
                    level,
                    (bbox.x1 + bbox.x2) / 2,
                    int(xs.size),
                )
            )
    return observations


def _link_streams(
    observations: list[LaneObservation], ink: np.ndarray, region: Box, scale: float
) -> tuple[list[TextStream], list[ContentEdge]]:
    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_level: dict[int, list[int]] = {}
    for index, observation in enumerate(observations):
        by_level.setdefault(observation.level, []).append(index)
    for level in sorted(by_level):
        current = by_level[level]
        following = by_level.get(level + 1, [])
        candidate_pairs = []
        for left in current:
            source = observations[left]
            for right in following:
                target = observations[right]
                width = max(source.bbox.width, target.bbox.width)
                distance = abs(source.center_x - target.center_x)
                width_ratio = max(source.bbox.width, target.bbox.width) / max(
                    1, min(source.bbox.width, target.bbox.width)
                )
                if (
                    distance <= max(scale * 0.45, width * 0.48)
                    and width_ratio <= 2.8
                ):
                    candidate_pairs.append(
                        (distance / max(width, scale), width_ratio, left, right)
                    )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, _, left, right in sorted(candidate_pairs):
            if left in used_left or right in used_right:
                continue
            union(left, right)
            used_left.add(left)
            used_right.add(right)

    groups: dict[int, list[int]] = {}
    for index in range(len(observations)):
        groups.setdefault(find(index), []).append(index)
    streams: list[TextStream] = []
    for members in groups.values():
        items = [observations[index] for index in members]
        levels = {item.level for item in items}
        if len(levels) < 2:
            continue
        bbox = Box(
            min(item.bbox.x1 for item in items),
            min(item.bbox.y1 for item in items),
            max(item.bbox.x2 for item in items),
            max(item.bbox.y2 for item in items),
        )
        centers = np.asarray([item.center_x for item in items])
        widths = np.asarray([item.bbox.width for item in items])
        stability = float(np.std(centers) / max(scale, float(np.median(widths)), 1.0))
        streams.append(
            TextStream(
                f"stream_{len(streams):04d}",
                bbox,
                [item.id for item in items],
                float(np.median(centers)),
                float(np.median(widths)),
                max(0.05, min(1.0, 1.0 - stability)),
            )
        )

    # Separated radicals and punctuation can each form a stable narrow trace.
    # If two traces are closer than one glyph and occupy the same vertical
    # windows, treat them as one compound writing stream.  Genuine parallel
    # columns normally have centres farther apart than the local glyph scale.
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(streams):
            left_levels = {
                observations[int(item.rsplit("_", 1)[1])].level
                for item in left.observation_ids
            }
            for right_index in range(left_index + 1, len(streams)):
                right = streams[right_index]
                distance = abs(left.center_x - right.center_x)
                combined_left = min(left.bbox.x1, right.bbox.x1)
                combined_right = max(left.bbox.x2, right.bbox.x2)
                if (
                    distance >= scale * 0.90
                    or combined_right - combined_left > scale * 1.48
                ):
                    continue
                right_levels = {
                    observations[int(item.rsplit("_", 1)[1])].level
                    for item in right.observation_ids
                }
                synchrony = len(left_levels & right_levels) / max(
                    1, min(len(left_levels), len(right_levels))
                )
                if synchrony < 0.72:
                    continue
                items = [
                    observations[int(item.rsplit("_", 1)[1])]
                    for item in left.observation_ids + right.observation_ids
                ]
                left.bbox = Box(
                    min(item.bbox.x1 for item in items),
                    min(item.bbox.y1 for item in items),
                    max(item.bbox.x2 for item in items),
                    max(item.bbox.y2 for item in items),
                )
                left.observation_ids = [item.id for item in items]
                left.center_x = float(np.median([item.center_x for item in items]))
                left.median_width = float(np.median([item.bbox.width for item in items]))
                left.confidence = min(left.confidence, right.confidence)
                del streams[right_index]
                changed = True
                break
            if changed:
                break

    for index, stream in enumerate(streams):
        stream.id = f"stream_{index:04d}"

    def row_centers(stream: TextStream) -> list[float]:
        x1 = max(0, stream.bbox.x1 - region.x1)
        x2 = min(ink.shape[1], stream.bbox.x2 - region.x1)
        y1 = max(0, stream.bbox.y1 - region.y1)
        y2 = min(ink.shape[0], stream.bbox.y2 - region.y1)
        rows = ((ink[y1:y2, x1:x2] > 0).sum(axis=1) >= 2).astype(np.uint8)
        rows = cv2.morphologyEx(
            rows.reshape(-1, 1),
            cv2.MORPH_CLOSE,
            np.ones((max(2, int(scale * 0.08)), 1), np.uint8),
        ).ravel() > 0
        return [y1 + (start + end) / 2 for start, end in _runs(rows, 0)]

    signatures = {stream.id: row_centers(stream) for stream in streams}
    for stream in streams:
        if (
            stream.median_width < scale * 0.28
            and stream.bbox.height > scale * 2.5
        ):
            stream.role = "rule_fragment"
        elif stream.bbox.height < scale * 1.35:
            stream.role = "noise"

    attachment_edges: list[ContentEdge] = []
    for satellite in streams:
        if satellite.role != "text":
            continue
        candidates = []
        for primary in streams:
            if primary is satellite or primary.role != "text":
                continue
            if satellite.median_width >= primary.median_width * 0.68:
                continue
            overlap = max(
                0,
                min(satellite.bbox.y2, primary.bbox.y2)
                - max(satellite.bbox.y1, primary.bbox.y1),
            )
            overlap_ratio = overlap / max(1, min(satellite.bbox.height, primary.bbox.height))
            horizontal_gap = max(
                0,
                max(satellite.bbox.x1, primary.bbox.x1)
                - min(satellite.bbox.x2, primary.bbox.x2),
            )
            if overlap_ratio < 0.72 or horizontal_gap > scale * 1.15:
                continue
            small_centers = signatures[satellite.id]
            large_centers = signatures[primary.id]
            if len(small_centers) < 3 or len(large_centers) < 3:
                continue
            count_ratio = len(small_centers) / len(large_centers)
            alignment = float(
                np.median(
                    [min(abs(y - other) for other in large_centers) for y in small_centers]
                )
            )
            narrow_companion = (
                satellite.median_width <= scale * 0.55
                and primary.median_width >= scale * 0.75
                and abs(satellite.center_x - primary.center_x) <= scale * 1.25
            )
            synchronized = (
                0.45 <= count_ratio <= 2.50 and alignment <= scale * 0.80
            )
            if narrow_companion or synchronized:
                direction_penalty = (
                    0.0 if satellite.center_x > primary.center_x else scale * 0.18
                )
                candidates.append(
                    (
                        direction_penalty
                        + abs(satellite.center_x - primary.center_x),
                        alignment,
                        primary,
                    )
                )
        if candidates:
            _, alignment, primary = min(candidates, key=lambda item: (item[0], item[1]))
            satellite.role = "satellite"
            attachment_edges.append(
                ContentEdge(
                    satellite.id,
                    primary.id,
                    "attaches_to",
                    max(0.05, 1.0 - alignment / max(scale, 1.0)),
                )
            )

    # Reading edges only encode high-confidence right-to-left peers.  Vertical
    # block transitions remain explicit instead of being guessed globally.
    edges: list[ContentEdge] = attachment_edges
    for source in streams:
        if source.role != "text":
            continue
        peers = []
        for target in streams:
            if (
                source is target
                or target.role != "text"
                or source.center_x <= target.center_x
            ):
                continue
            overlap = max(
                0,
                min(source.bbox.y2, target.bbox.y2)
                - max(source.bbox.y1, target.bbox.y1),
            )
            ratio = overlap / max(1, min(source.bbox.height, target.bbox.height))
            gap = source.bbox.x1 - target.bbox.x2
            if ratio >= 0.55 and gap <= scale * 2.2:
                peers.append((max(0, gap), target, ratio))
        if peers:
            _, target, ratio = min(peers, key=lambda item: item[0])
            edges.append(ContentEdge(source.id, target.id, "precedes", ratio))
            source.neighbors.append(target.id)
            target.neighbors.append(source.id)

    ordered = sorted(
        (stream for stream in streams if stream.role == "text"),
        key=lambda item: (-item.center_x, item.bbox.y1),
    )
    for order, stream in enumerate(ordered):
        stream.order = order
        stream.neighbors = sorted(set(stream.neighbors))
    return streams, edges


def _vertical_frame_bounds(binary: np.ndarray) -> tuple[int, int]:
    """Return conservative inner bounds when long left/right frame rules exist."""
    height, width = binary.shape
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(height * 0.55)))),
    )
    rule_columns = (opened > 0).mean(axis=0) > 0.18
    runs = _runs(rule_columns, merge_gap=2)
    left_candidates = [end for start, end in runs if (start + end) / 2 < width * 0.25]
    right_candidates = [start for start, end in runs if (start + end) / 2 > width * 0.75]
    left = max(left_candidates, default=0)
    right = min(right_candidates, default=width)
    if right - left < width * 0.45:
        return 0, width
    return left, right


def _regular_record_groups(
    streams: list[TextStream],
    binary: np.ndarray,
    region: Box,
    scale: float,
    bands: list[tuple[int, int]],
) -> tuple[list[StreamGroup], set[str]]:
    """Group local streams around a repeated sequence of primary text anchors.

    Many record pages alternate a broad primary name stream with narrower
    annotations.  The repeated primary-stream pitch is stronger evidence for
    record boundaries than any single whitespace cut.
    """
    frame_left, frame_right = _vertical_frame_bounds(binary)
    groups: list[StreamGroup] = []
    assigned: set[str] = set()
    for start, end in bands:
        band_height = end - start
        if band_height < scale * 6:
            continue
        absolute_top = region.y1 + start
        absolute_bottom = region.y1 + end
        candidates = []
        for stream in streams:
            overlap = max(
                0,
                min(absolute_bottom, stream.bbox.y2)
                - max(absolute_top, stream.bbox.y1),
            )
            if (
                stream.role == "text"
                and stream.median_width >= scale * 0.82
                and overlap / max(1, band_height) >= 0.42
                and region.x1 + frame_left <= stream.center_x <= region.x1 + frame_right
            ):
                candidates.append(stream)
        candidates.sort(key=lambda item: item.center_x)
        gaps = [
            right.center_x - left.center_x
            for left, right in zip(candidates, candidates[1:])
        ]
        pitch_gaps = [gap for gap in gaps if scale * 2.25 <= gap <= scale * 4.2]
        if len(pitch_gaps) < 3:
            continue
        pitch = float(np.median(pitch_gaps))

        clusters: list[list[TextStream]] = []
        for stream in candidates:
            if clusters and stream.center_x - clusters[-1][-1].center_x < pitch * 0.74:
                clusters[-1].append(stream)
            else:
                clusters.append([stream])
        anchors = [
            max(
                cluster,
                key=lambda item: (item.bbox.height, item.confidence, item.median_width),
            )
            for cluster in clusters
        ]
        if len(anchors) < 4:
            continue

        boundaries = [float(region.x1 + frame_left)]
        boundaries.extend(
            (left.center_x + right.center_x) / 2
            for left, right in zip(anchors, anchors[1:])
        )
        boundaries.append(float(region.x1 + frame_right))
        for index, anchor in enumerate(anchors):
            left = boundaries[index]
            right = boundaries[index + 1]
            members = []
            for stream in streams:
                overlap = max(
                    0,
                    min(absolute_bottom, stream.bbox.y2)
                    - max(absolute_top, stream.bbox.y1),
                )
                if (
                    stream.role in {"text", "satellite"}
                    and overlap / max(1, stream.bbox.height) >= 0.65
                    and left <= stream.center_x < right
                ):
                    members.append(stream.id)
            if anchor.id not in members:
                members.append(anchor.id)
            assigned.update(members)
            groups.append(
                StreamGroup(
                    f"record_{len(groups):04d}",
                    Box(
                        int(round(left)),
                        absolute_top,
                        int(round(right)),
                        absolute_bottom,
                    ),
                    sorted(set(members)),
                    0,
                )
            )
    return groups, assigned


def _group_streams(
    streams: list[TextStream],
    edges: list[ContentEdge],
    binary: np.ndarray,
    region: Box,
    scale: float,
    bands: list[tuple[int, int]],
) -> list[StreamGroup]:
    by_id = {stream.id: stream for stream in streams}
    attached: dict[str, list[str]] = {}
    for edge in edges:
        if edge.relation == "attaches_to":
            attached.setdefault(edge.target, []).append(edge.source)
    groups: list[StreamGroup] = []
    for stream in streams:
        if stream.role != "text":
            continue
        members = [stream.id, *sorted(attached.get(stream.id, []))]
        items = [by_id[item] for item in members]
        groups.append(
            StreamGroup(
                f"group_{len(groups):04d}",
                Box(
                    min(item.bbox.x1 for item in items),
                    min(item.bbox.y1 for item in items),
                    max(item.bbox.x2 for item in items),
                    max(item.bbox.y2 for item in items),
                ),
                members,
                0,
            )
        )
    record_groups, assigned = _regular_record_groups(
        streams, binary, region, scale, bands
    )
    groups = [
        group
        for group in groups
        if not any(stream_id in assigned for stream_id in group.stream_ids)
    ]
    groups.extend(record_groups)
    for index, group in enumerate(groups):
        group.id = f"group_{index:04d}"
    for order, group in enumerate(
        sorted(groups, key=lambda item: (-item.bbox.x2, item.bbox.y1))
    ):
        group.order = order
    return groups


def detect_content_graph(image: Image.Image, region: Box | None = None) -> ContentGraph:
    """Detect local vertical text streams without recognising characters."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    region = region or Box(0, 0, width, height)
    binary = _binarize(rgb)[region.y1 : region.y2, region.x1 : region.x2]
    ink = _remove_spanning_rules(binary)
    scale = _estimate_glyph_scale(ink)
    bands = _horizontal_bands(binary, scale)
    observations: list[LaneObservation] = []
    for band_index, (start, end) in enumerate(bands):
        band_region = Box(region.x1, region.y1 + start, region.x2, region.y1 + end)
        band_observations = _observe_lanes(ink[start:end], band_region, scale)
        for item in band_observations:
            observations.append(
                LaneObservation(
                    f"observation_{len(observations):04d}",
                    item.bbox,
                    item.level + band_index * 10_000,
                    item.center_x,
                    item.ink,
                )
            )
    streams, edges = _link_streams(observations, ink, region, scale)
    groups = _group_streams(streams, edges, binary, region, scale, bands)
    return ContentGraph(
        width, height, region, scale, observations, streams, edges, groups
    )


def draw_content_graph(
    image: Image.Image, graph: ContentGraph, *, show_observations: bool = False
) -> Image.Image:
    """Draw stream extents, centre tracks, and high-confidence order edges."""
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    if show_observations:
        for observation in graph.observations:
            draw.rectangle(observation.bbox.as_list(), outline="#b8e0d2", width=1)
    by_id = {stream.id: stream for stream in graph.streams}
    for stream in graph.streams:
        draw.rectangle(stream.bbox.as_list(), outline="#0077b6", width=2)
        draw.line(
            (stream.center_x, stream.bbox.y1, stream.center_x, stream.bbox.y2),
            fill="#00b4d8",
            width=1,
        )
        draw.text((stream.bbox.x1, stream.bbox.y1), str(stream.order), fill="#d00000")
    for edge in graph.edges:
        source, target = by_id[edge.source], by_id[edge.target]
        y = max(source.bbox.y1, target.bbox.y1) + 6
        draw.line((source.center_x, y, target.center_x, y), fill="#ff9f1c", width=2)
    for group in graph.groups:
        draw.rectangle(group.bbox.as_list(), outline="#8338ec", width=2)
        draw.text(
            (group.bbox.x1 + 2, group.bbox.y1 + 2),
            f"G{group.order}",
            fill="#6a00f4",
        )
    return output
