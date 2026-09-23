# Modal layout API

The deployed service exposes live layout inference for uploaded images and a
precomputed random-example path for the personal website.

## Deploy

Run these commands from commit `afba6c815d2e9e693670fcd3f3b816b7251672a0`:

```powershell
python -m modal volume create cjk-layout-assets
python -m modal volume put cjk-layout-assets "D:\Programs\Digitalization\runs\context_roles\box_segmenter.pt" /model/box_segmenter.pt

python scripts/precompute_modal_examples.py `
  --checkpoint "D:\Programs\Digitalization\runs\context_roles\box_segmenter.pt" `
  --images "D:\Programs\Digitalization\training\images\archive_diverse"
python -m modal volume put -f cjk-layout-assets .modal-examples /examples

python -m modal deploy modal_api.py
```

The model is 397,089 bytes (about 388 KiB), so uploading it is effectively
instant on a normal connection. The example bundle is also small; it contains
WebP copies plus precomputed JSON, not model predictions generated per click.

## Personal-site integration

Live inference requires the user to choose the page type before submitting:

```js
const API = "https://YOUR-MODAL-ENDPOINT.modal.run";

async function analyzeUpload(file, pageKind) {
  const body = new FormData();
  body.append("file", file);
  body.append("page_kind", pageKind);

  const response = await fetch(`${API}/analyze`, { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
```

The random-example button uses the precomputed path:

```js
async function loadRandomExample() {
  const response = await fetch(`${API}/examples/random`);
  if (!response.ok) throw new Error(await response.text());
  const result = await response.json();
  const imageUrl = new URL(result.example.image_url, API).href;
  return { result, imageUrl };
}
```

Draw `element.polygon` when present, otherwise draw `element.bbox`. Use purple
for `primary` and orange for `annotation`.

```js
function drawElements(ctx, elements) {
  ctx.lineWidth = 2;
  for (const element of elements) {
    ctx.strokeStyle = element.role === "primary" ? "#8745c4" : "#ec7900";
    ctx.beginPath();
    if (element.polygon?.length) {
      ctx.moveTo(element.polygon[0][0], element.polygon[0][1]);
      for (const [x, y] of element.polygon.slice(1)) ctx.lineTo(x, y);
      ctx.closePath();
    } else {
      const [x1, y1, x2, y2] = element.bbox;
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
    }
    ctx.stroke();
  }
}
```
