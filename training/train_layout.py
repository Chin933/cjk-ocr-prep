"""Fine-tune YOLOv8 to detect individual text columns in vertical Chinese documents.

Workflow
--------
1. Annotate images in LabelImg (YOLO format, class 0 = text_column).
2. Run split_dataset.py to create train/val splits.
3. Run this script to fine-tune YOLOv8n on the annotated data.
4. Best weights -> training/runs/layout/weights/best.pt
"""
from pathlib import Path
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "dataset.yaml"
RUNS_DIR  = ROOT / "runs"
BASE_WEIGHTS = ROOT.parent / "models" / "yolo" / "yolov8n.pt"

dataset_cfg = {
    "path": str(ROOT.resolve()),
    "train": "images/train",
    "val":   "images/val",
    "nc":    1,
    "names": ["text_column"],
}
with open(DATA_YAML, "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False, allow_unicode=True)
print(f"Dataset config written to {DATA_YAML}")

model = YOLO(str(BASE_WEIGHTS))

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=1024,
    batch=4,
    device="cpu",
    workers=2,
    project=str(RUNS_DIR),
    name="layout",
    exist_ok=True,
    patience=20,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
    degrees=2.0, translate=0.05, scale=0.3,
    flipud=0.0, fliplr=0.0, mosaic=0.3,
)

best = RUNS_DIR / "layout" / "weights" / "best.pt"
print(f"\nTraining complete. Best weights: {best}")
