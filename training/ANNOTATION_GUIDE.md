# How to Annotate — Column Bounding Boxes

## Launch LabelImg

```
cd D:\Programs\OCR
labelimg training\images\all\ training\labels\all\ training\predefined_classes.txt
```

## Settings (do once)

1. **View → Auto Save Mode** — turn ON (saves on every Next/Prev)
2. **Change Save Format** → select **YOLO** (top toolbar button, make sure it says "YOLO" not "PascalVOC")
3. Zoom to ~80% so you can see individual columns clearly

## What to label

Draw one bounding box per **text column** — meaning each individual narrow
vertical strip of characters, including:

| What to box | What NOT to box |
|---|---|
| Each narrow vertical character column | Page border / frame lines |
| Columns in the main text area | Blank white margins |
| Header/commentary columns (smaller text) | Decorative ornaments |
| Title blocks with single large characters | Seal stamps |

**One box = one column of characters**, even if it only contains 3-4 characters.

## Tips

- Use **W** key to start drawing a box
- **D** / **A** to move to next/previous image
- Hold **Ctrl** and scroll to zoom in on dense areas
- It's OK if boxes slightly overlap at boundaries
- For commentary columns (smaller text at top/page edge), still draw individual boxes
- Aim for tight fits — box should contain the text but not much whitespace

## Label format

YOLO format (auto-saved as `.txt` beside each image):
```
0  x_center  y_center  width  height    ← all values 0–1, normalised
```
`0` = class index for `text_column`

## How many to annotate

- Annotate ALL 23 images fully for best results
- Minimum viable: at least 15 images with complete annotations

## After annotation

```
cd D:\Programs\OCR
python training\prepare_dataset.py
python training\train_layout.py
```
