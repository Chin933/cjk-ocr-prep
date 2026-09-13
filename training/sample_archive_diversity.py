"""Select structurally novel pages from the 420-volume source archive.

The first pass renders sparse, low-resolution pages in memory and compares
their detected layout-tree features with the current corpus. Only a small
farthest-first subset is rendered to disk at review resolution. No OCR or
manual boxes are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from analyze_layout_families import _features
from digitalization import detect_layout


def _render(document: fitz.Document, page_index: int, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = document[page_index].get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _page_indices(page_count: int) -> list[int]:
    return sorted(
        {
            min(page_count - 1, max(0, int(round((page_count - 1) * fraction))))
            for fraction in (0.12, 0.50, 0.88)
        }
    )


def _content_metrics(image: Image.Image) -> tuple[float, int]:
    """Estimate non-ruling content so blank forms do not dominate novelty."""
    gray = np.asarray(image.convert("L"))
    binary = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, int(width * 0.10)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, int(height * 0.10)))),
    )
    rules = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), dtype=np.uint8)
    )
    content = cv2.bitwise_and(binary, cv2.bitwise_not(rules))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        (content > 0).astype(np.uint8)
    )
    areas = stats[1:component_count, cv2.CC_STAT_AREA]
    glyph_like = int(((areas >= 8) & (areas <= 2000)).sum())
    return float((content > 0).mean()), glyph_like


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pdf-step", type=int, default=14)
    parser.add_argument("--select", type=int, default=24)
    parser.add_argument("--screen-dpi", type=int, default=96)
    parser.add_argument("--output-dpi", type=int, default=160)
    parser.add_argument("--min-content-density", type=float, default=0.015)
    parser.add_argument("--min-text-regions", type=int, default=8)
    parser.add_argument(
        "--exclude-list",
        type=Path,
        default=PROJECT_ROOT / "training" / "archive_sample_exclusions.txt",
        help="Optional PDF-stem,page-number lines to omit from sampling.",
    )
    args = parser.parse_args()

    exclusions: set[tuple[str, int]] = set()
    if args.exclude_list.exists():
        for line in args.exclude_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pdf_stem, page_number = (part.strip() for part in line.split(",", 1))
            exclusions.add((pdf_stem, int(page_number)))

    pdfs = sorted(args.archive_dir.glob("*.pdf"), key=lambda path: int(path.stem))
    sampled_pdfs = pdfs[:: args.pdf_step]
    if pdfs and pdfs[-1] not in sampled_pdfs:
        sampled_pdfs.append(pdfs[-1])

    existing_source = json.loads(args.existing.read_text(encoding="utf-8"))
    existing_vectors = np.asarray(
        [_features(record)[0] for record in existing_source["records"]],
        dtype=np.float64,
    )

    candidates = []
    vectors = []
    total = len(sampled_pdfs) * 3
    completed = 0
    for pdf_path in sampled_pdfs:
        with fitz.open(pdf_path) as document:
            for page_index in _page_indices(document.page_count):
                completed += 1
                if (pdf_path.stem, page_index + 1) in exclusions:
                    print(
                        f"[{completed:03d}/{total:03d}] {pdf_path.name} "
                        f"p{page_index + 1} excluded",
                        flush=True,
                    )
                    continue
                image = _render(document, page_index, args.screen_dpi)
                layout = detect_layout(image).to_dict()
                content_density, component_count = _content_metrics(image)
                vector, description = _features(
                    {"image": "", "layout": layout}
                )
                candidates.append(
                    {
                        "pdf": pdf_path.name,
                        "page": page_index + 1,
                        "page_count": document.page_count,
                        "features": description,
                        "content_density": round(content_density, 4),
                        "glyph_like_components": component_count,
                    }
                )
                vectors.append(vector)
                print(
                    f"[{completed:03d}/{total:03d}] {pdf_path.name} p{page_index + 1}",
                    flush=True,
                )

    candidate_vectors = np.asarray(vectors, dtype=np.float64)
    combined = np.vstack([existing_vectors, candidate_vectors])
    scale = np.std(combined, axis=0)
    scale[scale < 1e-8] = 1.0
    center = np.mean(combined, axis=0)
    existing_scaled = (existing_vectors - center) / scale
    candidate_scaled = (candidate_vectors - center) / scale

    reference_distance = np.sqrt(
        ((candidate_scaled[:, None, :] - existing_scaled[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    eligible = {
        index
        for index, item in enumerate(candidates)
        if item["content_density"] >= args.min_content_density
        and item["glyph_like_components"] >= 100
        and item["features"]["text_regions"] >= args.min_text_regions
    }
    selected: list[int] = []
    remaining = set(eligible)
    while remaining and len(selected) < args.select:
        if not selected:
            chosen = max(remaining, key=lambda index: reference_distance[index])
        else:
            selected_matrix = candidate_scaled[selected]
            chosen = max(
                remaining,
                key=lambda index: min(
                    reference_distance[index],
                    float(
                        np.sqrt(
                            ((selected_matrix - candidate_scaled[index]) ** 2).sum(axis=1)
                        ).min()
                    ),
                ),
            )
        selected.append(chosen)
        remaining.remove(chosen)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in args.output_dir.glob("archive_*.jpg"):
        obsolete.unlink()
    selected_records = []
    for rank, index in enumerate(selected, start=1):
        item = candidates[index]
        pdf_path = args.archive_dir / item["pdf"]
        with fitz.open(pdf_path) as document:
            image = _render(document, item["page"] - 1, args.output_dpi)
        output_name = f"archive_{int(pdf_path.stem):03d}_p{item['page']:04d}.jpg"
        image.save(args.output_dir / output_name, quality=90)
        selected_records.append(
            {
                **item,
                "image": output_name,
                "novelty_to_existing": round(float(reference_distance[index]), 4),
                "selection_rank": rank,
            }
        )

    report = {
        "archive_pdf_count": len(pdfs),
        "sampled_pdf_count": len(sampled_pdfs),
        "screened_page_count": len(candidates),
        "eligible_page_count": len(eligible),
        "existing_page_count": len(existing_vectors),
        "selected_page_count": len(selected_records),
        "screen_dpi": args.screen_dpi,
        "output_dpi": args.output_dpi,
        "excluded_page_count": len(exclusions),
        "eligibility": {
            "min_content_density": args.min_content_density,
            "min_glyph_like_components": 100,
            "min_text_regions": args.min_text_regions,
        },
        "selected": selected_records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
