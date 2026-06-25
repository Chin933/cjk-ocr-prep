"""Extract selected PDF pages as high-quality training images."""

from pathlib import Path

import fitz
from PIL import Image


PDF = Path(r"D:\Programs\Keju\Zhujuanjicheng\《清代硃卷集成》曾氏选抄.pdf")
OUT_DIR = Path("training/images/all")
DPI = 200

# 1-based PDF page numbers.
PAGE_RANGES = [
    range(3, 9),
    range(13, 17),
    range(27, 33),
    range(37, 44),
    range(44, 60),
    range(95, 119),
    range(170, 183),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted({p for r in PAGE_RANGES for p in r})
    mat = fitz.Matrix(DPI / 72.0, DPI / 72.0)

    print(f"Extracting {len(pages)} pages: {pages}")
    doc = fitz.open(str(PDF))
    saved: list[int] = []
    try:
        for pg in pages:
            if pg > len(doc):
                print(f"  skip p{pg}: beyond end of PDF ({len(doc)} pages)")
                continue
            out = OUT_DIR / f"page_{pg:04d}.jpg"
            if out.exists():
                saved.append(pg)
                print(f"  keep page {pg:4d}  ({out.name} already exists)")
                continue
            pix = doc[pg - 1].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(str(out), quality=92)
            saved.append(pg)
            print(f"  saved page {pg:4d}  ({img.width}x{img.height})")
    finally:
        doc.close()

    print(f"\nDone: {len(saved)} images in {OUT_DIR}/")
    print("Next: generate projection pseudo-labels with training/prepare_projection_dataset.py")


if __name__ == "__main__":
    main()
