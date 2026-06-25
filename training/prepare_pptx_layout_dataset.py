"""Convert PPTX rectangle annotations into a small YOLO layout dataset.

The PowerPoint annotation convention is:
  red    (#FF0000) -> block
  yellow (#FFFF00) -> major_column
  green  (#00B050) -> subcolumn

Each slide should contain one page image plus colored rectangle outlines.
Coordinates are mapped from PowerPoint EMUs back to the embedded image pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_PPTX = ROOT / "演示文稿1.pptx"
DEFAULT_IMAGES = ROOT / "images" / "all"
DEFAULT_OUT = ROOT / "pptx_layout_dataset"

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

COLOR_TO_CLASS = {
    "FF0000": (0, "block"),
    "FFFF00": (1, "major_column"),
    "00B050": (2, "subcolumn"),
}


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def file_md5(path: Path) -> str:
    return md5_bytes(path.read_bytes())


def read_xml(zf: ZipFile, name: str) -> ET.Element:
    return ET.fromstring(zf.read(name))


def slide_paths(zf: ZipFile) -> list[str]:
    return sorted(
        name
        for name in zf.namelist()
        if name.startswith("ppt/slides/slide") and name.endswith(".xml")
    )


def slide_relationships(zf: ZipFile, slide_path: str) -> dict[str, str]:
    rel_path = slide_path.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
    root = read_xml(zf, rel_path)
    rels: dict[str, str] = {}
    for rel in root.findall("rel:Relationship", NS):
        rel_id = rel.attrib["Id"]
        target = rel.attrib["Target"]
        if target.startswith("../"):
            target = "ppt/" + target[3:]
        rels[rel_id] = target
    return rels


def shape_geometry(shape: ET.Element) -> tuple[int, int, int, int] | None:
    xfrm = shape.find(".//a:xfrm", NS)
    if xfrm is None:
        return None
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is None or ext is None:
        return None
    x = int(off.attrib["x"])
    y = int(off.attrib["y"])
    w = int(ext.attrib["cx"])
    h = int(ext.attrib["cy"])
    return (x, y, x + w, y + h)


def outline_color(shape: ET.Element) -> str | None:
    line = shape.find(".//a:ln", NS)
    if line is None:
        return None
    srgb = line.find(".//a:srgbClr", NS)
    if srgb is None:
        return None
    return srgb.attrib.get("val", "").upper()


def find_slide_picture(slide: ET.Element) -> tuple[str, tuple[int, int, int, int]]:
    for pic in slide.findall(".//p:pic", NS):
        blip = pic.find(".//a:blip", NS)
        geom = shape_geometry(pic)
        if blip is not None and geom is not None:
            rid = blip.attrib.get(f"{{{NS['r']}}}embed")
            if rid:
                return rid, geom
    raise ValueError("slide has no embedded picture")


def map_rect_to_pixels(
    rect: tuple[int, int, int, int],
    picture_rect: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    px1, py1, px2, py2 = picture_rect
    iw, ih = image_size
    x1, y1, x2, y2 = rect
    rx1 = round((x1 - px1) / (px2 - px1) * iw)
    ry1 = round((y1 - py1) / (py2 - py1) * ih)
    rx2 = round((x2 - px1) / (px2 - px1) * iw)
    ry2 = round((y2 - py1) / (py2 - py1) * ih)
    rx1 = max(0, min(iw, rx1))
    rx2 = max(0, min(iw, rx2))
    ry1 = max(0, min(ih, ry1))
    ry2 = max(0, min(ih, ry2))
    if rx2 - rx1 < 3 or ry2 - ry1 < 3:
        return None
    return (rx1, ry1, rx2, ry2)


def yolo_line(cls_id: int, rect: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    x1, y1, x2, y2 = rect
    w, h = size
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def draw_debug(image_path: Path, annotations: list[dict], output_path: Path) -> None:
    colors = {"block": "red", "major_column": "yellow", "subcolumn": "lime"}
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        box = ann["bbox"]
        label = ann["class_name"]
        draw.rectangle(box, outline=colors[label], width=4 if label == "block" else 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def build_dataset(pptx: Path, image_dir: Path, out_dir: Path) -> None:
    image_hashes = {file_md5(path): path for path in sorted(image_dir.glob("*.jpg"))}
    all_records: list[dict] = []

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)

    with ZipFile(pptx) as zf:
        for slide_index, slide_path in enumerate(slide_paths(zf), start=1):
            slide = read_xml(zf, slide_path)
            rels = slide_relationships(zf, slide_path)
            rid, picture_rect = find_slide_picture(slide)
            image_part = rels[rid]
            embedded_bytes = zf.read(image_part)
            image_hash = md5_bytes(embedded_bytes)
            if image_hash not in image_hashes:
                raise ValueError(f"slide {slide_index}: embedded image not found in {image_dir}")

            image_path = image_hashes[image_hash]
            with Image.open(image_path) as im:
                image_size = im.size

            annotations: list[dict] = []
            for shape in slide.findall(".//p:spTree/*", NS):
                if not shape.tag.endswith("}sp"):
                    continue
                color = outline_color(shape)
                if color not in COLOR_TO_CLASS:
                    continue
                geom = shape_geometry(shape)
                if geom is None:
                    continue
                bbox = map_rect_to_pixels(geom, picture_rect, image_size)
                if bbox is None:
                    continue
                cls_id, cls_name = COLOR_TO_CLASS[color]
                annotations.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "bbox": list(bbox),
                    }
                )

            split = "train" if slide_index == 1 else "val"
            shutil.copy2(image_path, out_dir / "images" / split / image_path.name)
            label_path = out_dir / "labels" / split / f"{image_path.stem}.txt"
            label_path.write_text(
                "\n".join(yolo_line(a["class_id"], tuple(a["bbox"]), image_size) for a in annotations)
                + "\n",
                encoding="utf-8",
            )
            draw_debug(image_path, annotations, out_dir / "debug" / image_path.name)
            all_records.append(
                {
                    "slide": slide_index,
                    "image": image_path.name,
                    "split": split,
                    "width": image_size[0],
                    "height": image_size[1],
                    "annotations": annotations,
                }
            )

    names = [name for _, name in sorted((v for v in COLOR_TO_CLASS.values()), key=lambda x: x[0])]
    (out_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "annotations.json").write_text(
        json.dumps(
            {
                "classes": [{"id": i, "name": name} for i, name in enumerate(names)],
                "records": all_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir}")
    for record in all_records:
        counts: dict[str, int] = {}
        for ann in record["annotations"]:
            counts[ann["class_name"]] = counts.get(ann["class_name"], 0) + 1
        print(f"{record['image']} -> {record['split']} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pptx", type=Path, default=DEFAULT_PPTX)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_dataset(args.pptx, args.image_dir, args.out_dir)


if __name__ == "__main__":
    main()
