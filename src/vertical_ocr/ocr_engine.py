"""OCR engine wrapper using RapidOCR (ONNX runtime backend).

RapidOCR uses the same model weights as PaddleOCR but runs inference through
ONNX Runtime, which is stable on Windows CPU without the oneDNN bug present
in PaddlePaddle v3.x.

Confidence
----------
RapidOCR returns one score per detected text line.  We distribute it uniformly
across the characters in that line (best available granularity without custom
model surgery).  Lines below `low_conf_threshold` are flagged for review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_ocr_instance: Any = None


def _get_ocr() -> Any:
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        log.info("Loading RapidOCR (ONNX) model…")
        _ocr_instance = RapidOCR()
        log.info("RapidOCR ready.")
    return _ocr_instance


@dataclass
class CharResult:
    char: str
    confidence: float
    low_confidence: bool = False


@dataclass
class LineResult:
    text: str
    chars: list[CharResult] = field(default_factory=list)
    mean_confidence: float = 0.0
    box: list[list[float]] = field(default_factory=list)

    def flagged_text(self, marker: str = "?") -> str:
        if not self.chars:
            return self.text
        out = []
        for ch in self.chars:
            if ch.low_confidence:
                out.append(marker)
            out.append(ch.char)
        return "".join(out)


def _parse_rapid_result(
    result: list | None,
    low_conf_threshold: float,
) -> list[LineResult]:
    if not result:
        return []
    lines: list[LineResult] = []
    for item in result:
        # item = [box, text, score]
        box, text, score = item[0], item[1], float(item[2])
        char_results = [
            CharResult(
                char=ch,
                confidence=score,
                low_confidence=score < low_conf_threshold,
            )
            for ch in text
        ]
        lines.append(LineResult(
            text=text,
            chars=char_results,
            mean_confidence=score,
            box=box,
        ))
    return lines


def ocr_image(
    image: Image.Image,
    low_conf_threshold: float = 0.80,
    **_kwargs,
) -> list[LineResult]:
    """Run OCR on a PIL image, return structured results with confidence."""
    ocr = _get_ocr()
    img_np = np.array(image.convert("RGB"))
    try:
        result, _ = ocr(img_np)
    except Exception as exc:
        log.error("RapidOCR failed: %s", exc)
        return []
    lines = _parse_rapid_result(result, low_conf_threshold)
    if lines:
        avg = sum(l.mean_confidence for l in lines) / len(lines)
        log.debug("OCR: %d lines, mean_conf=%.3f", len(lines), avg)
    return lines
