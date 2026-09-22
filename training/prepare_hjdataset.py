"""Download HJDataset and convert from COCO JSON → YOLO txt format.

HJDataset: Historical Japanese document layout dataset (part of LayoutParser).
Classes: 0=Page, 1=Frame, 2=Zone, 3=Background, 4=Headline, 5=Drop-Capital, 6=Caption

Outputs to training/hjdataset/{images,labels}/{train,val}/
"""
import json, shutil, urllib.request
from pathlib import Path

OUT = Path("training/hjdataset")
(OUT / "images/train").mkdir(parents=True, exist_ok=True)
(OUT / "images/val").mkdir(parents=True, exist_ok=True)
(OUT / "labels/train").mkdir(parents=True, exist_ok=True)
(OUT / "labels/val").mkdir(parents=True, exist_ok=True)

# HJDataset download URL (LayoutParser public release)
HJ_URL = "https://huggingface.co/datasets/layouts/HJDataset/resolve/main/HJDataset.zip"
ZIP = OUT / "HJDataset.zip"

if not ZIP.exists():
    print(f"Downloading HJDataset from {HJ_URL} ...")
    try:
        urllib.request.urlretrieve(HJ_URL, ZIP)
        print(f"Downloaded: {ZIP.stat().st_size/1e6:.1f} MB")
    except Exception as e:
        # Fallback: LayoutParser GitHub release
        alt = "https://github.com/Layout-Parser/layout-parser/releases/download/v0.0/HJDataset.tar.gz"
        print(f"First URL failed ({e}), trying {alt} ...")
        urllib.request.urlretrieve(alt, OUT / "HJDataset.tar.gz")
        import tarfile
        with tarfile.open(OUT / "HJDataset.tar.gz") as t:
            t.extractall(OUT)
        print("Extracted.")
else:
    print(f"Already downloaded: {ZIP}")

# Extract zip if needed
import zipfile
extracted = OUT / "HJDataset"
if not extracted.exists() and ZIP.exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(OUT)
    print("Extracted ZIP.")


def coco_to_yolo(ann_file: Path, img_dir: Path, out_img: Path, out_lbl: Path):
    """Convert one COCO JSON split to YOLO format."""
    data = json.loads(ann_file.read_text())

    # Map image_id → {file_name, width, height}
    id2img = {img["id"]: img for img in data["images"]}

    # Collect annotations per image
    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Class id mapping
    cat_ids = sorted({c["id"] for c in data["categories"]})
    cat_map = {cid: i for i, cid in enumerate(cat_ids)}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    print(f"  Classes: { {i: cats[cid] for cid, i in cat_map.items()} }")

    for img_info in data["images"]:
        iid = img_info["id"]
        fname = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]

        # Copy image
        src = img_dir / fname
        if src.exists():
            shutil.copy2(src, out_img / fname)

        # Write YOLO label
        lines = []
        for ann in img_anns[iid]:
            cls = cat_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]           # COCO: top-left x,y + w,h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl_name = Path(fname).stem + ".txt"
        (out_lbl / lbl_name).write_text("\n".join(lines))

    print(f"  {len(data['images'])} images → {out_img}")


# Find annotation files
ann_dir = extracted / "annotations" if (extracted / "annotations").exists() else extracted
train_ann = next(ann_dir.glob("*train*.json"), None)
val_ann   = next(ann_dir.glob("*val*.json"),   None)
img_dir   = next((p for p in [extracted/"images", extracted/"imgs", extracted] if (p).is_dir()), extracted)

if train_ann:
    print("\nConverting train split...")
    coco_to_yolo(train_ann, img_dir, OUT/"images/train", OUT/"labels/train")
if val_ann:
    print("Converting val split...")
    coco_to_yolo(val_ann, img_dir, OUT/"images/val", OUT/"labels/val")

# Write dataset YAML for YOLO training
yaml = OUT / "hjdataset.yaml"
yaml.write_text(f"""path: {OUT.resolve()}
train: images/train
val: images/val

nc: 7
names: [Page, Frame, Zone, Background, Headline, Drop-Capital, Caption]
""")
print(f"\nDataset YAML saved: {yaml}")
print("Next: run  python training/train_modal.py")
