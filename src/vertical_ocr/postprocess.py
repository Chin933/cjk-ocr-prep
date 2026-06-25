"""Post-processing: assemble column OCR results into ordered text.

Reading order for vertical Traditional Chinese:
  - Columns are read right-to-left.
  - Within each column, characters are read top-to-bottom.

PaddleOCR returns bounding boxes per detected line segment within a column.
This module sorts those segments top-to-bottom within each column, then
concatenates columns right-to-left, producing the final reading-order text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TextIO

from .layout import Column, ColumnType
from .ocr_engine import LineResult

log = logging.getLogger(__name__)


@dataclass
class ColumnText:
    """All OCR text for one detected column."""

    column_index: int        # 0 = rightmost (first in reading order)
    col_type: ColumnType
    lines: list[LineResult] = field(default_factory=list)
    x1: int = 0
    x2: int = 0
    y1: int = 0
    y2: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.mean_confidence for l in self.lines) / len(self.lines)

    @property
    def has_low_confidence(self) -> bool:
        return any(
            ch.low_confidence
            for line in self.lines
            for ch in line.chars
        )

    def flagged_text(self, marker: str = "⚠") -> str:
        return "\n".join(line.flagged_text(marker) for line in self.lines)


@dataclass
class PageResult:
    """OCR result for one page, columns in right-to-left reading order."""

    page_index: int
    columns: list[ColumnText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Plain text, columns separated by blank line."""
        return "\n\n".join(col.text for col in self.columns if col.text.strip())

    @property
    def flagged_text(self) -> str:
        """Text with ⚠ before every low-confidence character."""
        return "\n\n".join(
            col.flagged_text() for col in self.columns if col.text.strip()
        )

    @property
    def low_confidence_columns(self) -> list[ColumnText]:
        return [c for c in self.columns if c.has_low_confidence]

    def to_dict(self) -> dict:
        """Serialisable dict representation."""
        return {
            "page_index": self.page_index,
            "columns": [
                {
                    "index": col.column_index,
                    "type": col.col_type,
                    "x1": col.x1, "x2": col.x2, "y1": col.y1, "y2": col.y2,
                    "mean_confidence": round(col.mean_confidence, 4),
                    "has_low_confidence": col.has_low_confidence,
                    "lines": [
                        {
                            "text": line.text,
                            "confidence": round(line.mean_confidence, 4),
                            "chars": [
                                {"char": ch.char, "conf": round(ch.confidence, 4),
                                 "low": ch.low_confidence}
                                for ch in line.chars
                            ],
                        }
                        for line in col.lines
                    ],
                }
                for col in self.columns
            ],
        }


def assemble(
    page_index: int,
    columns: list[Column],
    ocr_results: list[list[LineResult]],
) -> PageResult:
    """Merge layout + OCR results into a PageResult.

    Args:
        page_index: 0-based page number.
        columns: Column objects (already sorted right-to-left, index 0 = right).
        ocr_results: Parallel list – ocr_results[i] is the OCR for columns[i].

    Returns:
        PageResult with columns in right-to-left reading order.
    """
    col_texts: list[ColumnText] = []
    for col, lines in zip(columns, ocr_results):
        # Sort lines top-to-bottom by their top y-coordinate
        def top_y(line: LineResult) -> float:
            if line.box:
                return min(pt[1] for pt in line.box)
            return 0.0

        sorted_lines = sorted(lines, key=top_y)
        col_texts.append(ColumnText(
            column_index=col.index,
            col_type=col.col_type,
            lines=sorted_lines,
            x1=col.x1, x2=col.x2, y1=col.y1, y2=col.y2,
        ))

    # Ensure right-to-left reading order (index 0 = rightmost = first)
    col_texts.sort(key=lambda c: c.column_index)
    return PageResult(page_index=page_index, columns=col_texts)


# ── Output writers ─────────────────────────────────────────────────────────────

def write_plain(results: list[PageResult], out: TextIO, separator: str = "\n\n---\n\n") -> None:
    """Write plain text to *out*, pages separated by *separator*."""
    for page in results:
        out.write(f"[Page {page.page_index + 1}]\n")
        out.write(page.full_text)
        out.write(separator)


def write_flagged(results: list[PageResult], out: TextIO, separator: str = "\n\n---\n\n") -> None:
    """Write text with ⚠ markers for low-confidence characters."""
    for page in results:
        out.write(f"[Page {page.page_index + 1}]\n")
        out.write(page.flagged_text)
        out.write(separator)


def write_json(results: list[PageResult], out: TextIO) -> None:
    """Write structured JSON with full confidence information."""
    payload = [r.to_dict() for r in results]
    json.dump(payload, out, ensure_ascii=False, indent=2)


def write_tsv(results: list[PageResult], out: TextIO) -> None:
    """Write tab-separated values: page, col_index, col_type, conf, text."""
    out.write("page\tcol\ttype\tconfidence\tlow_conf\ttext\n")
    for page in results:
        for col in page.columns:
            out.write(
                f"{page.page_index + 1}\t{col.column_index}\t{col.col_type}\t"
                f"{col.mean_confidence:.4f}\t{col.has_low_confidence}\t"
                f"{col.text.replace(chr(10), '|')}\n"
            )
