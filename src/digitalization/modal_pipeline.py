"""Shared inference pipeline for the Modal API and precomputed examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .box_detector import BoxSegmenter
from .page_structure import PAGE_KINDS, decode_page, exam_rule_bands
from .ruling_geometry import refine_tracks


VERSION = "layout-tree-v1@afba6c8"


def load_segmenter(checkpoint: str | Path) -> tuple[BoxSegmenter, dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    channels = saved["state_dict"]["out.weight"].shape[0]
    model = BoxSegmenter(channels=channels)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _public_element(element: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": element["id"],
        "role": element["role"],
        "bbox": [int(value) for value in element["boxes"][0]],
        "band": element.get("band"),
        "lane": element.get("lane"),
    }
    if element.get("polygon") is not None:
        record["polygon"] = _native(element["polygon"])
    for key in ("score", "source"):
        if element.get(key) is not None:
            record[key] = _native(element[key])
    return record


def analyze_image(
    image: Image.Image,
    page_kind: str,
    model: BoxSegmenter,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if page_kind not in PAGE_KINDS:
        raise ValueError(f"Unknown page_kind: {page_kind}")

    # Import here so the training module remains optional for library users.
    from training.restore_rules import restore

    image = image.convert("RGB")
    _, _, bands = restore(image)
    bands = refine_tracks(image, bands)

    pitches = [band["pitch"] for band in bands if "pitch" in band]
    base_scale = float(checkpoint.get("scale", 0.75))
    scale = base_scale * 51 / float(np.median(pitches)) if pitches else base_scale
    gray = np.asarray(image.convert("L"))
    reduced = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy((1 - reduced.astype(np.float32) / 255)[None, None])
    with torch.inference_mode():
        probabilities = model(tensor).sigmoid()[0].numpy()

    note_probability = cv2.resize(probabilities[1], image.size)
    primary_probability = cv2.resize(probabilities[0], image.size)
    if page_kind == "exam":
        bands = exam_rule_bands(image, bands)

    notes, primary, _ = decode_page(
        image,
        bands,
        note_probability,
        page_kind,
        primary_probability=primary_probability,
    )
    for prefix, elements in (("N", notes), ("M", primary)):
        elements.sort(key=lambda item: (item.get("band", 0), -item["boxes"][0][0], item["boxes"][0][1]))
        for index, element in enumerate(elements, 1):
            element["id"] = f"{prefix}{index:04d}"

    return {
        "version": VERSION,
        "page_kind": page_kind,
        "image": {"width": image.width, "height": image.height},
        "bands": _native(bands),
        "elements": [_public_element(element) for element in notes + primary],
    }
