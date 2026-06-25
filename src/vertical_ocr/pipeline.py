"""Main OCR pipeline: PDF → columns → OCR → structured output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from .layout import Column, detect_columns, draw_columns
from .ocr_engine import LineResult, ocr_image
from .postprocess import PageResult, assemble
from .pdf_utils import pdf_to_images

log = logging.getLogger(__name__)


def process_page(
    page_index: int,
    image: Image.Image,
    dpi: int = 300,
    low_conf_threshold: float = 0.80,
    gap_threshold_frac: float = 0.02,
    debug_dir: Path | None = None,
) -> PageResult:
    """Run the full pipeline on one page image.

    Args:
        page_index: 0-based page number (used for logging/output).
        image: RGB PIL image of the page.
        dpi: Resolution of *image*.
        low_conf_threshold: OCR confidence below which chars are flagged.
        gap_threshold_frac: Sensitivity for gap detection in layout.
        debug_dir: If set, save debug images here (column overlays).

    Returns:
        PageResult with columns in right-to-left reading order.
    """
    log.info("Page %d: detecting columns…", page_index + 1)
    columns: list[Column] = detect_columns(
        image,
        dpi=dpi,
        wide_valley_frac=gap_threshold_frac,
    )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        annotated = draw_columns(image, columns)
        annotated.save(debug_dir / f"page_{page_index + 1:04d}_layout.jpg", quality=85)

    log.info("Page %d: %d columns → running OCR…", page_index + 1, len(columns))
    ocr_results: list[list[LineResult]] = []
    for col in columns:
        crop = col.crop(image)
        # Rotate 90° CCW: vertical top→bottom text becomes horizontal left→right.
        # Rightmost sub-column (read first) becomes topmost row after rotation.
        crop = crop.rotate(90, expand=True)
        lines = ocr_image(crop, low_conf_threshold=low_conf_threshold)
        ocr_results.append(lines)
        log.debug(
            "  col #%d (%s): %d lines, mean conf=%.3f",
            col.index, col.col_type, len(lines),
            sum(l.mean_confidence for l in lines) / max(len(lines), 1),
        )

    return assemble(page_index, columns, ocr_results)


def process_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
    dpi: int = 300,
    low_conf_threshold: float = 0.80,
    gap_threshold_frac: float = 0.02,
    debug_dir: Path | None = None,
) -> Iterator[PageResult]:
    """Yield PageResult for each page in *pdf_path*.

    Args:
        pdf_path: Path to the input PDF.
        page_range: Optional (start, end) page indices (0-based, exclusive end).
        dpi: Rendering resolution.
        low_conf_threshold: Confidence threshold for flagging.
        gap_threshold_frac: Layout gap detection sensitivity.
        debug_dir: Directory for debug layout-overlay images.

    Yields:
        PageResult in page order.
    """
    pdf_path = Path(pdf_path)
    for page_idx, image in pdf_to_images(pdf_path, dpi=dpi, page_range=page_range):
        yield process_page(
            page_index=page_idx,
            image=image,
            dpi=dpi,
            low_conf_threshold=low_conf_threshold,
            gap_threshold_frac=gap_threshold_frac,
            debug_dir=debug_dir,
        )


def process_image_file(
    image_path: str | Path,
    dpi: int = 300,
    low_conf_threshold: float = 0.80,
    gap_threshold_frac: float = 0.02,
    debug_dir: Path | None = None,
) -> PageResult:
    """Run the pipeline on a single image file."""
    img = Image.open(str(image_path)).convert("RGB")
    return process_page(
        page_index=0,
        image=img,
        dpi=dpi,
        low_conf_threshold=low_conf_threshold,
        gap_threshold_frac=gap_threshold_frac,
        debug_dir=debug_dir,
    )
