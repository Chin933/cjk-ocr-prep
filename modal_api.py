"""Modal deployment for historical Chinese document layout analysis."""

from pathlib import Path

import modal


APP_NAME = "cjk-layout-api"
VOLUME_NAME = "cjk-layout-assets"
ASSET_ROOT = Path("/assets")
MODEL_PATH = ASSET_ROOT / "model" / "box_segmenter.pt"
EXAMPLE_ROOT = ASSET_ROOT / "examples"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0"
    )
    .pip_install(
        "fastapi==0.115.8",
        "python-multipart==0.0.20",
        "opencv-python-headless==4.10.0.84",
        "Pillow==11.1.0",
        "numpy==2.1.3",
    )
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("training", "/app/training", copy=True)
)

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=image,
    volumes={str(ASSET_ROOT): assets},
    cpu=2.0,
    memory=4096,
    timeout=60,
    scaledown_window=300,
    max_containers=2,
)
class LayoutService:
    @modal.enter()
    def load_model(self):
        import os
        import sys

        import cv2
        import torch

        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        os.environ["OMP_NUM_THREADS"] = "2"
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        from digitalization.modal_pipeline import load_segmenter

        self.model, self.checkpoint = load_segmenter(MODEL_PATH)

    @modal.asgi_app()
    def web(self):
        import io
        import json
        import random
        import sys

        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from PIL import Image, ImageOps, UnidentifiedImageError

        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/src")
        from digitalization.modal_pipeline import PAGE_KINDS, VERSION, analyze_image

        api = FastAPI(title="Historical Chinese Layout API", version=VERSION)
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["https://chin933.github.io"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

        @api.get("/health")
        def health():
            return {"status": "ok", "version": VERSION, "page_kinds": list(PAGE_KINDS)}

        @api.post("/analyze")
        async def analyze(file: UploadFile = File(...), page_kind: str = Form(...)):
            if page_kind not in PAGE_KINDS:
                raise HTTPException(422, detail={"page_kind": f"Expected one of {PAGE_KINDS}"})
            content_type = (file.content_type or "").lower()
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(415, detail="Supported formats: JPEG, PNG, WebP")
            payload = await file.read(10 * 1024 * 1024 + 1)
            if len(payload) > 10 * 1024 * 1024:
                raise HTTPException(413, detail="Image exceeds the 10 MB limit")
            try:
                source = Image.open(io.BytesIO(payload))
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, detail="Supported formats: JPEG, PNG, WebP")
                source.load()
                source = ImageOps.exif_transpose(source).convert("RGB")
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(400, detail="Invalid image") from error
            return analyze_image(source, page_kind, self.model, self.checkpoint)

        @api.get("/examples/random")
        def random_example():
            index_path = EXAMPLE_ROOT / "index.json"
            if not index_path.exists():
                raise HTTPException(503, detail="Precomputed examples are not installed")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entry = random.choice(index["examples"])
            result = json.loads((EXAMPLE_ROOT / entry["result"]).read_text(encoding="utf-8"))
            result["example"] = {
                "id": entry["id"],
                "case": entry.get("case"),
                "image_url": f"/examples/{entry['id']}/image",
                "precomputed": True,
            }
            return result

        @api.get("/examples/{example_id}/image")
        def example_image(example_id: str):
            index_path = EXAMPLE_ROOT / "index.json"
            if not index_path.exists():
                raise HTTPException(404, detail="Example not found")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entry = next((item for item in index["examples"] if item["id"] == example_id), None)
            if entry is None:
                raise HTTPException(404, detail="Example not found")
            return FileResponse(EXAMPLE_ROOT / entry["image"], media_type="image/webp")

        return api
