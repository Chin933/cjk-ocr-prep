"""Image preprocessing to improve OCR accuracy on historical Chinese documents.

Historical woodblock prints have:
- Uneven ink density (faded strokes, bleed-through from back of page)
- Slight skew / warping from scan
- Low contrast in some regions
- Noise from paper texture

Improvements applied before OCR:
1. CLAHE – adaptive histogram equalization (boosts local contrast)
2. Unsharp masking – sharpen character edges
3. Binarization (optional, Sauvola local threshold) – clean binary image
4. Deskew – correct small rotation
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def enhance(
    image: Image.Image,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    sharpen: bool = True,
    sharpen_sigma: float = 1.0,
    sharpen_amount: float = 1.5,
    deskew: bool = True,
    max_skew_deg: float = 2.0,
) -> Image.Image:
    """Apply preprocessing to improve OCR accuracy.

    Args:
        image: Input RGB PIL image.
        clahe_clip: CLAHE clip limit (higher = more contrast boost).
        clahe_tile: CLAHE tile grid size in pixels (smaller = more local).
        sharpen: Apply unsharp masking.
        sharpen_sigma: Gaussian sigma for unsharp mask.
        sharpen_amount: Strength of sharpening.
        deskew: Correct small rotational skew.
        max_skew_deg: Maximum angle to correct (beyond this, skip).

    Returns:
        Enhanced RGB PIL image.
    """
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. CLAHE on grayscale
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tile, clahe_tile),
    )
    gray_eq = clahe.apply(gray)

    # 2. Unsharp masking
    if sharpen:
        blurred = cv2.GaussianBlur(gray_eq, (0, 0), sharpen_sigma)
        gray_eq = cv2.addWeighted(gray_eq, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
        gray_eq = np.clip(gray_eq, 0, 255).astype(np.uint8)

    # 3. Deskew using projection profile
    if deskew:
        gray_eq = _deskew(gray_eq, max_deg=max_skew_deg)

    # Back to RGB for OCR engine
    rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _deskew(gray: np.ndarray, max_deg: float = 2.0) -> np.ndarray:
    """Correct small rotational skew via projection profile maximisation."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0
    for angle in np.linspace(-max_deg, max_deg, 41):
        h, w = binary.shape
        centre = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        profile = rotated.sum(axis=1).astype(np.float64)
        # Score = variance of the projection (maximised when text rows are aligned)
        score = float(profile.var())
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.1:
        return gray  # no meaningful skew

    h, w = gray.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, best_angle, 1.0)
    corrected = cv2.warpAffine(gray, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    return corrected
