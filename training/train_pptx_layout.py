"""Train a small YOLO detector on PPTX-derived hierarchical layout labels.

This is intended as a smoke-test model for the annotation/training pipeline,
not as a production model. With only two labeled pages, the goal is to verify
that labels, classes, training, and prediction artifacts all work end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "pptx_layout_dataset" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"
BASE_WEIGHTS = ROOT.parent / "models" / "yolo" / "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(BASE_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="pptx_layout_2p_smoke")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"Missing {DATA_YAML}. Run: python training/prepare_pptx_layout_dataset.py"
        )
    if not Path(args.weights).exists():
        raise SystemExit(f"Missing YOLO weights: {args.weights}")

    model = YOLO(args.weights)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR.resolve()),
        name=args.name,
        exist_ok=True,
        patience=max(10, args.epochs // 2),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        degrees=0.5,
        translate=0.02,
        scale=0.10,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        close_mosaic=0,
    )
    print(f"Best weights: {RUNS_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
