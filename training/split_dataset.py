"""Split annotated images into train/val (80/20) and write dataset.yaml."""
import random, shutil
from pathlib import Path

SEED = 42; VAL_FRAC = 0.2
IMG_ALL = Path("training/images/all")
LBL_ALL = Path("training/labels/all")

for split in ["train", "val"]:
    Path(f"training/images/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"training/labels/{split}").mkdir(parents=True, exist_ok=True)

labeled = [p for p in sorted(IMG_ALL.glob("*.jpg"))
           if (LBL_ALL / (p.stem + ".txt")).exists()
           and (LBL_ALL / (p.stem + ".txt")).stat().st_size > 0]

random.seed(SEED); random.shuffle(labeled)
n_val = max(1, int(len(labeled) * VAL_FRAC))
val_set, train_set = labeled[:n_val], labeled[n_val:]

for img_path in train_set:
    shutil.copy2(img_path, f"training/images/train/{img_path.name}")
    shutil.copy2(LBL_ALL / (img_path.stem + ".txt"),
                 f"training/labels/train/{img_path.stem}.txt")
for img_path in val_set:
    shutil.copy2(img_path, f"training/images/val/{img_path.name}")
    shutil.copy2(LBL_ALL / (img_path.stem + ".txt"),
                 f"training/labels/val/{img_path.stem}.txt")

print(f"Split: {len(train_set)} train / {len(val_set)} val")
yaml_path = Path("training/dataset.yaml")
yaml_path.write_text(f"""path: {Path('training').resolve()}
train: images/train
val:   images/val
nc: 1
names: [text_column]
""")
print(f"Dataset config: {yaml_path}")
print("Ready. Run:  modal run training/train_modal.py")