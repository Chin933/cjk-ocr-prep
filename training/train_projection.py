"""Train YOLO on the generated projection-column dataset.

Run prepare_projection_dataset.py first. The resulting weights are written to
training/runs/projection_layout/weights/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "generated_projection" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"
BASE_WEIGHTS = ROOT.parent / "models" / "yolo" / "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(BASE_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="projection_layout")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"Missing {DATA_YAML}. Run: python training/prepare_projection_dataset.py"
        )

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
        patience=max(10, args.epochs // 4),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=1.0,
        translate=0.03,
        scale=0.20,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.1,
    )

    print(f"Best weights: {RUNS_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
