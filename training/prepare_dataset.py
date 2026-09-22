"""Split annotated images into train/val sets and write dataset YAML.

Run this AFTER annotating all images with LabelImg (YOLO format).

Usage:
    python training/prepare_dataset.py [--val-ratio 0.15]
"""
import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    all_imgs = sorted(Path("training/images/all").glob("*.jpg"))
    all_labels = Path("training/labels/all")

    # Only keep images that have a corresponding annotation file
    annotated = [p for p in all_imgs if (all_labels / (p.stem + ".txt")).exists()]
    print(f"Annotated images: {len(annotated)} / {len(all_imgs)}")

    random.seed(42)
    random.shuffle(annotated)
    n_val = max(1, int(len(annotated) * args.val_ratio))
    val_set = set(p.name for p in annotated[:n_val])

    for split in ("train", "val"):
        Path(f"training/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"training/labels/{split}").mkdir(parents=True, exist_ok=True)

    for img_path in annotated:
        split = "val" if img_path.name in val_set else "train"
        shutil.copy(img_path, f"training/images/{split}/{img_path.name}")
        lbl = all_labels / (img_path.stem + ".txt")
        shutil.copy(lbl, f"training/labels/{split}/{img_path.stem}.txt")

    n_train = len(annotated) - n_val
    print(f"Split: {n_train} train, {n_val} val")

    # Write dataset YAML for YOLOv8
    yaml_content = f"""path: {Path('training').resolve().as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: text_column
"""
    Path("training/dataset.yaml").write_text(yaml_content)
    print("Wrote training/dataset.yaml")

if __name__ == "__main__":
    main()
