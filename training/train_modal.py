"""Fine-tune DocLayout-YOLO on Chinese woodblock pages (auto-annotated).

Run with:
    modal token new            # one-time login
    modal run training/train_modal.py

Prerequisites:
    python training/auto_annotate.py   # generates labels
    python training/split_dataset.py   # creates train/val split + dataset.yaml

The trained model is saved to training/woodblock_best.pt locally.
Estimated cost: ~$0.30-0.60 on T4 GPU (20-40 min).
"""

import modal
from pathlib import Path

DATASET_LOCAL = Path("training")
YAML_LOCAL    = DATASET_LOCAL / "dataset.yaml"

app = modal.App("doclayout-woodblock-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "doclayout-yolo",
        "huggingface_hub",
    )
)

volume = modal.Volume.from_name("woodblock-training", create_if_missing=True)
VOL = Path("/vol")


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={VOL: volume},
)
def train(epochs: int = 80, imgsz: int = 1024, batch: int = 4):
    import huggingface_hub
    from doclayout_yolo import YOLOv10

    # Download base DocLayout-YOLO weights (pre-trained on diverse doc layouts)
    print("Downloading base DocLayout-YOLO weights...")
    base_weights = huggingface_hub.hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        cache_dir=str(VOL / "hf_cache"),
    )
    print(f"Base weights: {base_weights}")

    yaml_path = VOL / "dataset.yaml"
    runs_dir  = VOL / "runs"

    model = YOLOv10(base_weights)
    print(f"Fine-tuning for {epochs} epochs (batch={batch}, imgsz={imgsz})...")

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(runs_dir),
        name="woodblock",
        exist_ok=True,
        patience=20,
        lr0=0.001,        # lower LR for fine-tuning
        lrf=0.01,
        save=True,
        plots=True,
        verbose=True,
    )

    best = runs_dir / "woodblock/weights/best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    volume.commit()
    return str(best)


@app.local_entrypoint()
def main(epochs: int = 80):
    if not YAML_LOCAL.exists():
        print("ERROR: Run  python training/split_dataset.py  first.")
        return

    # Fix absolute path in YAML for the remote Linux container
    yaml_text = YAML_LOCAL.read_text()
    yaml_remote = yaml_text.replace(
        str(DATASET_LOCAL.resolve()),
        str(VOL)
    ).replace("\\", "/")

    print("Uploading dataset to Modal volume...")
    vol = modal.Volume.from_name("woodblock-training", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        # YAML with Linux paths
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(yaml_remote); tmp.close()
        batch.put_file(tmp.name, "/dataset.yaml")
        os.unlink(tmp.name)

        # Images and labels
        for split in ["train", "val"]:
            for folder in ["images", "labels"]:
                src = DATASET_LOCAL / folder / split
                if src.exists():
                    for f in src.iterdir():
                        batch.put_file(str(f), f"/{folder}/{split}/{f.name}")

    print(f"Upload complete. Starting training (epochs={epochs})...")
    best_path = train.remote(epochs=epochs)
    print(f"Remote training done → {best_path}")

    # Download model back
    out = DATASET_LOCAL / "woodblock_best.pt"
    with vol.batch_download() as batch:
        remote_rel = best_path.replace("/vol/", "")
        batch.get_file(remote_rel, str(out))
    print(f"Model saved: {out}")
    print(f"\nTest: update model_path in test_doclayout.py to '{out}'")
