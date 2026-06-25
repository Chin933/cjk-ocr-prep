"""Hierarchical layout detection for vertical Chinese woodblock documents.

Pipeline:

1. Binarize and crop inside the dominant printed page frame.
2. Detect long horizontal rules with morphology.
3. Split the page into horizontal blocks.
4. Inside each block, find major vertical column groups.
5. Inside each major group, split optional readable subcolumns.

The goal is to follow the page's genealogy structure before detecting final
OCR regions, instead of treating every ink strip as one flat object class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ColumnType = Literal["text_column", "blank", "region"]
RegionType = Literal["frame", "separator", "block", "major_column"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Column:
    """One readable text column — the final output unit."""
    x1: int
    x2: int
    y1: int
    y2: int
    index: int = 0          # 0 = rightmost (first in reading order)
    col_type: ColumnType = "text_column"
    confidence: float = 1.0  # 1.0 = from detected line, 0.7 = from projection

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x1, self.y1, self.x2, self.y2))


@dataclass
class LayoutRegion:
    """Intermediate structural region used for debug and evaluation."""
    x1: int
    x2: int
    y1: int
    y2: int
    region_type: RegionType
    index: int = 0
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class LayoutResult:
    """Full hierarchical page layout."""
    frame: LayoutRegion
    separators: list[LayoutRegion]
    blocks: list[LayoutRegion]
    major_columns: list[LayoutRegion]
    columns: list[Column]


# ── Image utilities ────────────────────────────────────────────────────────────

def _binarise(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_mask(binary: np.ndarray, orientation: str, length: int) -> np.ndarray:
    if orientation == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    elif orientation == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.dilate(eroded, kernel, iterations=1)


def _runs_from_profile(
    profile: np.ndarray,
    threshold: float,
    min_width: int = 1,
    merge_gap: int = 0,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, value in enumerate(profile):
        if value >= threshold and not in_run:
            start = i
            in_run = True
        elif value < threshold and in_run:
            if i - start >= min_width:
                runs.append((start, i))
            in_run = False
    if in_run and len(profile) - start >= min_width:
        runs.append((start, len(profile)))

    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def detect_outer_frame(
    binary: np.ndarray,
    min_line_frac: float = 0.45,
    inset: int = 6,
) -> tuple[int, int, int, int]:
    """Return the inner crop of the dominant printed frame."""
    h, w = binary.shape
    h_len = max(80, int(w * 0.25))
    v_len = max(80, int(h * 0.25))
    h_mask = _line_mask(binary, "horizontal", h_len)
    v_mask = _line_mask(binary, "vertical", v_len)

    h_profile = h_mask.sum(axis=1) / 255.0
    v_profile = v_mask.sum(axis=0) / 255.0
    h_runs = _runs_from_profile(h_profile, w * min_line_frac, merge_gap=4)
    v_runs = _runs_from_profile(v_profile, h * min_line_frac, merge_gap=4)

    if len(h_runs) < 2 or len(v_runs) < 2:
        return (0, 0, w, h)

    top = h_runs[0][1] + inset
    bottom = h_runs[-1][0] - inset
    left = v_runs[0][1] + inset
    right = v_runs[-1][0] - inset

    if right - left < w * 0.35 or bottom - top < h * 0.35:
        return (0, 0, w, h)
    return (max(0, left), max(0, top), min(w, right), min(h, bottom))


# ── Level 1: major column groups ──────────────────────────────────────────────

def _vertical_projection(binary: np.ndarray) -> np.ndarray:
    """Sum ink pixels per image column → 1-D array over x."""
    return binary.sum(axis=0).astype(np.float64)


def _smooth(profile: np.ndarray, sigma: int = 5) -> np.ndarray:
    kernel = np.ones(sigma * 2 + 1) / (sigma * 2 + 1)
    return np.convolve(profile, kernel, mode="same")


def _find_valleys(
    profile: np.ndarray,
    threshold_frac: float,
    min_width: int,
) -> list[tuple[int, int]]:
    """Return (start, end) x-ranges where profile < threshold_frac * max."""
    if profile.max() == 0:
        return []
    threshold = profile.max() * threshold_frac
    below = profile < threshold

    valleys: list[tuple[int, int]] = []
    in_v, start = False, 0
    for i, b in enumerate(below):
        if b and not in_v:
            in_v, start = True, i
        elif not b and in_v:
            in_v = False
            if i - start >= min_width:
                valleys.append((start, i))
    if in_v and len(profile) - start >= min_width:
        valleys.append((start, len(profile)))
    return valleys


def _valleys_to_spans(
    valleys: list[tuple[int, int]],
    page_w: int,
    min_span: int,
) -> list[tuple[int, int]]:
    """Convert valley boundaries into content-span (x1, x2) intervals."""
    if not valleys:
        return [(0, page_w)] if page_w >= min_span else []

    spans = []
    cursor = 0
    for gap_start, gap_end in sorted(valleys):
        x1, x2 = cursor, gap_start
        if x2 - x1 >= min_span:
            spans.append((x1, x2))
        cursor = max(cursor, gap_end)
    if page_w - cursor >= min_span:
        spans.append((cursor, page_w))
    return spans


def detect_major_groups(
    binary: np.ndarray,
    wide_valley_frac: float = 0.02,
    min_group_width_frac: float = 0.05,
) -> list[tuple[int, int]]:
    """Detect major column groups via full-page vertical projection.

    Wide valleys (inter-block whitespace) separate major regions.
    Returns list of (x1, x2) spans in right-to-left order.
    """
    h, w = binary.shape
    profile = _smooth(_vertical_projection(binary), sigma=10)
    min_valley_w = max(8, int(w * 0.02))   # at least 2% of page width
    valleys = _find_valleys(profile, wide_valley_frac, min_valley_w)
    spans = _valleys_to_spans(valleys, w, int(w * min_group_width_frac))
    # filter out spans with no ink (blank halves of spread pages)
    result = []
    for x1, x2 in spans:
        strip = binary[:, x1:x2]
        ink_density = strip.sum() / (strip.size * 255 + 1)
        if ink_density > 0.002:
            result.append((x1, x2))
    log.debug("Major groups: %s", result)
    return result


# ── Level 2: horizontal section separators ─────────────────────────────────────

def detect_horizontal_separators(
    binary: np.ndarray,
    min_width_frac: float = 0.25,
    merge_gap: int = 8,
) -> list[int]:
    """Find y-positions of horizontal rule lines using morphological opening.

    Long horizontal strokes (thick rules between genealogy sections) are
    detected by opening with a wide horizontal kernel.

    Returns sorted list of y-coordinates (one per rule, centre of the line).
    """
    h, w = binary.shape
    min_w = max(80, int(w * min_width_frac))
    h_mask = _line_mask(binary, "horizontal", min_w)
    profile = h_mask.sum(axis=1) / 255.0
    runs = _runs_from_profile(
        profile,
        threshold=max(40, w * min_width_frac),
        min_width=1,
        merge_gap=merge_gap,
    )
    merged = [(start + end) // 2 for start, end in runs]

    log.debug("Horizontal separators at y=%s", merged)
    return merged


def _separators_to_sections(
    separators: list[int],
    page_h: int,
    min_section_h: int = 40,
    edge_margin_frac: float = 0.05,
) -> list[tuple[int, int]]:
    """Convert horizontal separator y-positions into (y1, y2) row bands.

    Separators within *edge_margin_frac* of the top/bottom are frame borders
    and are excluded — we only keep interior section rules.
    """
    edge = int(page_h * edge_margin_frac)
    interior = [y for y in separators if edge < y < page_h - edge]
    ys = [0] + interior + [page_h]
    sections = []
    for i in range(len(ys) - 1):
        y1, y2 = ys[i], ys[i + 1]
        if y2 - y1 >= min_section_h:
            sections.append((y1, y2))
    return sections


# ── Level 3: sub-columns within a block ────────────────────────────────────────

def _ruling_line_profile(block: np.ndarray, min_seg_frac: float = 0.40) -> np.ndarray:
    """Return vertical projection of ruling-line mask within a block."""
    h = block.shape[0]
    seg_h = max(5, int(h * min_seg_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, seg_h))
    opened = cv2.morphologyEx(block, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilated = cv2.dilate(opened, kernel2, iterations=1)
    return dilated.sum(axis=0).astype(np.float32)


def _peaks_to_boundaries(
    profile: np.ndarray,
    min_prominence_frac: float = 0.08,
    merge_radius: int = 6,
) -> list[int]:
    """Find x-positions of ruling lines as peaks in the line-mask profile."""
    if profile.max() == 0:
        return []
    threshold = profile.max() * min_prominence_frac
    above = profile >= threshold
    peaks: list[int] = []
    in_run, run_start = False, 0
    for i, v in enumerate(above):
        if v and not in_run:
            in_run, run_start = True, i
        elif not v and in_run:
            in_run = False
            seg = profile[run_start:i]
            peaks.append(run_start + int(np.argmax(seg)))
    if in_run:
        seg = profile[run_start:]
        peaks.append(run_start + int(np.argmax(seg)))
    merged: list[int] = []
    for x in sorted(peaks):
        if merged and x - merged[-1] <= merge_radius:
            merged[-1] = (merged[-1] + x) // 2
        else:
            merged.append(x)
    return merged


def _estimate_pitch(boundaries: list[int]) -> float | None:
    if len(boundaries) < 2:
        return None
    gaps = sorted([boundaries[i+1] - boundaries[i]
                   for i in range(len(boundaries)-1)])
    return float(np.median(gaps[:max(1, len(gaps)//2)]))


def _fill_gaps(boundaries: list[int], w: int, pitch: float, max_fill: float = 2.5) -> list[int]:
    all_b = sorted(boundaries)
    filled: list[int] = list(all_b)
    for i in range(len(all_b) - 1):
        gap = all_b[i+1] - all_b[i]
        if gap > max_fill * pitch:
            n = round(gap / pitch) - 1
            for k in range(1, n+1):
                filled.append(int(all_b[i] + k * pitch))
    return sorted(set(filled))


def _vertical_line_dominance(strip: np.ndarray) -> float:
    """Fraction of ink explained by very long vertical rules/frame lines."""
    if strip.sum() == 0:
        return 0.0
    seg_h = max(5, int(strip.shape[0] * 0.55))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, seg_h))
    lines = cv2.morphologyEx(strip, cv2.MORPH_OPEN, kernel)
    return float(lines.sum() / (strip.sum() + 1))


def _projection_subcolumns(
    block: np.ndarray,
    min_col_width: int,
    valley_frac: float = 0.20,
    min_gap_width: int = 8,
    min_ink_density: float = 0.03,
) -> list[tuple[int, int]]:
    """Detect text lanes as dense projection islands separated by local valleys."""
    profile = _smooth(_vertical_projection(block), sigma=6)
    valleys = _find_valleys(profile, valley_frac, min_gap_width)
    spans = _valleys_to_spans(valleys, block.shape[1], min_col_width)

    result: list[tuple[int, int]] = []
    for x1, x2 in spans:
        strip = block[:, x1:x2]
        density = strip.sum() / (strip.size * 255 + 1)
        if density < min_ink_density:
            continue
        if _vertical_line_dominance(strip) > 0.35:
            continue
        result.append((x1, x2))
    return result


def _ink_density(binary: np.ndarray) -> float:
    if binary.size == 0:
        return 0.0
    return float(binary.sum() / (binary.size * 255 + 1))


def _is_text_like_region(
    binary: np.ndarray,
    min_density: float = 0.12,
    min_vertical_aspect: float = 1.5,
) -> bool:
    h, w = binary.shape
    if h < w * min_vertical_aspect:
        return False
    if _ink_density(binary) < min_density:
        return False
    if _vertical_line_dominance(binary) > 0.45:
        return False
    return True


def detect_subcolumns(
    block: np.ndarray,
    min_col_width: int,
    min_ink_density: float = 0.003,
) -> list[tuple[int, int]]:
    """Detect sub-columns using ruling-line peaks (not projection valleys).

    In woodblock text, column boundaries ARE the ruling lines, which appear
    as peaks (high ink) in the vertical projection — not as valleys.
    """
    h, w = block.shape
    if w < min_col_width * 2:
        return [(0, w)]

    projected = _projection_subcolumns(
        block,
        min_col_width=min_col_width,
        min_ink_density=max(0.01, min_ink_density),
    )
    if projected:
        return projected

    profile = _ruling_line_profile(block)
    boundaries = _peaks_to_boundaries(profile)

    boundaries = sorted(set([0] + boundaries + [w]))

    spans: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        x1, x2 = boundaries[i], boundaries[i+1]
        if x2 - x1 < min_col_width:
            continue
        strip = block[:, x1:x2]
        if strip.sum() / (strip.size * 255 + 1) < min_ink_density:
            continue
        if _vertical_line_dominance(strip) > 0.35:
            continue
        spans.append((x1, x2))

    return spans if spans else [(0, w)]


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_layout(
    image: Image.Image,
    dpi: int = 200,
    wide_valley_frac: float = 0.05,
    narrow_valley_frac: float = 0.02,
    h_sep_width_frac: float = 0.30,
    margin_trim_frac: float = 0.01,
) -> LayoutResult:
    """Detect page frame, separators, blocks, and text columns.

    Args:
        image: RGB PIL image of the page.
        dpi: Resolution of the image.
        wide_valley_frac: Ink threshold for major-group valley detection.
        narrow_valley_frac: Ink threshold for sub-column valley detection.
        h_sep_width_frac: Minimum width (as page fraction) for a horizontal
                          separator line to be detected.
        margin_trim_frac: Strip to ignore at top/bottom.

    Returns:
        List of Column objects sorted right-to-left (index 0 = rightmost).
    """
    min_col_width = max(18, int(dpi * 0.12))

    img_np = np.array(image.convert("L"))
    h, w = img_np.shape
    margin_top = int(h * margin_trim_frac)
    margin_bot = int(h * (1 - margin_trim_frac))
    full_binary = _binarise(img_np[margin_top:margin_bot, :])
    fx1, fy1, fx2, fy2 = detect_outer_frame(full_binary)
    binary = full_binary[fy1:fy2, fx1:fx2]
    bh, bw = binary.shape
    x_offset = fx1
    y_offset = margin_top + fy1
    log.info("Frame crop: x=[%d:%d], y=[%d:%d]", fx1, fx2, y_offset, margin_top + fy2)
    frame_region = LayoutRegion(
        x1=fx1,
        x2=fx2,
        y1=y_offset,
        y2=margin_top + fy2,
        region_type="frame",
    )

    # ── Level 1: major groups ─────────────────────────────────────────────────
    # Horizontal separators define genealogy blocks. Detect them before
    # vertical columns so mixed upper/lower structures do not share one profile.

    # ── Level 2: horizontal separators (on full binary, not per-group) ────────
    h_seps = detect_horizontal_separators(binary, min_width_frac=h_sep_width_frac)
    internal_h_seps = [sep for sep in h_seps if 25 < sep < bh - 25]
    sections = _separators_to_sections(internal_h_seps, bh)
    log.info("L1: %d horizontal blocks (separators at y=%s)", len(sections), internal_h_seps)

    separators = [
        LayoutRegion(
            x1=x_offset,
            x2=x_offset + bw,
            y1=max(y_offset, y_offset + sep - 2),
            y2=min(y_offset + bh, y_offset + sep + 2),
            region_type="separator",
            index=i,
        )
        for i, sep in enumerate(internal_h_seps)
    ]
    blocks = [
        LayoutRegion(
            x1=x_offset,
            x2=x_offset + bw,
            y1=y_offset + sy1,
            y2=y_offset + sy2,
            region_type="block",
            index=i,
        )
        for i, (sy1, sy2) in enumerate(sections)
    ]
    major_columns: list[LayoutRegion] = []

    # ── Level 3: sub-columns in each (group × section) block ─────────────────
    all_columns: list[Column] = []
    for sy1, sy2 in sections:
        section = binary[sy1:sy2, :]
        if _ink_density(section) < 0.002:
            continue

        groups = detect_major_groups(section, wide_valley_frac=wide_valley_frac)
        if not groups:
            groups = [(0, bw)]
        log.debug("Section y=[%d:%d] major groups: %s", sy1, sy2, groups)

        for group_index, (gx1, gx2) in enumerate(groups):
            major_columns.append(
                LayoutRegion(
                    x1=x_offset + gx1,
                    x2=x_offset + gx2,
                    y1=y_offset + sy1,
                    y2=y_offset + sy2,
                    region_type="major_column",
                    index=group_index,
                    confidence=0.8,
                )
            )
            block = binary[sy1:sy2, gx1:gx2]
            if _ink_density(block) < 0.002:
                continue
            sub_spans = detect_subcolumns(block, min_col_width, narrow_valley_frac)
            for sx1, sx2 in sub_spans:
                candidate = block[:, sx1:sx2]
                if not _is_text_like_region(candidate):
                    continue
                col = Column(
                    x1=x_offset + gx1 + sx1,
                    x2=x_offset + gx1 + sx2,
                    y1=y_offset + sy1,
                    y2=y_offset + sy2,
                    confidence=0.8,
                )
                all_columns.append(col)

    if not all_columns:
        log.warning("No columns detected; returning full page.")
        all_columns = [Column(x1=0, x2=w, y1=0, y2=h, confidence=0.3)]

    # Sort right-to-left and assign reading-order index
    all_columns.sort(key=lambda c: (-c.x1, c.y1))
    for i, c in enumerate(all_columns):
        c.index = i

    log.info("L3: %d sub-columns total", len(all_columns))
    return LayoutResult(
        frame=frame_region,
        separators=separators,
        blocks=blocks,
        major_columns=major_columns,
        columns=all_columns,
    )


def detect_columns(
    image: Image.Image,
    dpi: int = 200,
    wide_valley_frac: float = 0.05,
    narrow_valley_frac: float = 0.02,
    h_sep_width_frac: float = 0.30,
    margin_trim_frac: float = 0.01,
) -> list[Column]:
    """Detect final OCR text columns."""
    return detect_layout(
        image=image,
        dpi=dpi,
        wide_valley_frac=wide_valley_frac,
        narrow_valley_frac=narrow_valley_frac,
        h_sep_width_frac=h_sep_width_frac,
        margin_trim_frac=margin_trim_frac,
    ).columns


def draw_columns(image: Image.Image, columns: list[Column]) -> Image.Image:
    """Draw column bounding boxes on a copy of *image*."""
    from PIL import ImageDraw, ImageFont
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for col in columns:
        colour = "#00cc44" if col.confidence >= 1.0 else "#ff6600"
        draw.rectangle([col.x1, col.y1, col.x2, col.y2], outline=colour, width=2)
    return out


def draw_layout_debug(image: Image.Image, layout: LayoutResult) -> Image.Image:
    """Draw hierarchical layout regions on a copy of *image*."""
    from PIL import ImageDraw

    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    draw.rectangle(
        [layout.frame.x1, layout.frame.y1, layout.frame.x2, layout.frame.y2],
        outline="#1f77b4",
        width=4,
    )
    for block in layout.blocks:
        draw.rectangle([block.x1, block.y1, block.x2, block.y2], outline="#2ca02c", width=3)
    for major in layout.major_columns:
        draw.rectangle([major.x1, major.y1, major.x2, major.y2], outline="#9467bd", width=2)
    for separator in layout.separators:
        draw.rectangle(
            [separator.x1, separator.y1, separator.x2, separator.y2],
            outline="#d62728",
            width=4,
        )
    for col in layout.columns:
        draw.rectangle([col.x1, col.y1, col.x2, col.y2], outline="#ff7f0e", width=2)
    return out
