"""Convert PDF pages to PIL images using PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image

log = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
    page_range: tuple[int, int] | None = None,
) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page_number, PIL.Image) for each page in *pdf_path*.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution (300 recommended for OCR).
        page_range: Optional (start, end) page indices (0-based, exclusive end).
                    None means all pages.

    Yields:
        (page_index, PIL.Image in RGB mode)
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    start = 0
    end = len(doc)
    if page_range is not None:
        start, end = page_range
        end = min(end, len(doc))

    log.info("Rendering pages %d–%d of '%s' at %d dpi", start, end - 1, pdf_path.name, dpi)
    for page_idx in range(start, end):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield page_idx, img

    doc.close()


def page_count(pdf_path: str | Path) -> int:
    """Return total number of pages in the PDF."""
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)