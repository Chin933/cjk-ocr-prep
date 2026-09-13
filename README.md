# Digitalization

Digitalization detects hierarchical page layout and reading order in scanned
historical Chinese documents.  It focuses on pages with vertical writing,
ruling lines, nested sections, annotations, large titles, and two-page spreads.

The core package does **not** recognize characters.  OCR engines can consume the
ordered leaf regions afterward, but OCR is outside the package's main task.

## Output

For each image, Digitalization returns a tree:

```text
spread
├── right page
│   ├── upper region
│   │   └── columns in right-to-left order
│   └── lower region
└── left page
    └── columns in right-to-left order
```

Every node records its bounding box, split direction, children, structural
evidence, confidence, and reading-order index.  The tree is retained because
some historical layouts cannot be represented faithfully by a flat list of
rectangles.

## Installation

```bash
pip install -e .
```

Only Pillow, NumPy, and OpenCV are required for layout detection.

Development tools are installed explicitly:

```bash
pip install -e ".[dev]"
```

## Python API

```python
from PIL import Image
from digitalization import detect_layout, detect_rule_graph

image = Image.open("page.jpg")
document = detect_layout(image)

for region in document.reading_order:
    print(region.order, region.bbox)

# Optional: inspect the 2-D ruling graph and T-junctions directly.
rule_graph = detect_rule_graph(image)
print(rule_graph.junctions)
```

## Command line

```bash
digitalization-layout page.jpg \
  --output page.layout.json \
  --overlay page.layout.jpg \
  --graph-output page.rules.json \
  --graph-overlay page.rules.jpg
```

The JSON contains both the hierarchy and a flattened list of ordered leaf IDs.
The overlay numbers the detected reading regions for review.

## Method

The current development version combines:

- printed-frame and center-gutter detection
- multi-scale ruling-line extraction
- local 2-D ruling segments with preserved endpoints and junctions
- rejection of partial rules incorrectly promoted to page-wide separators
- whitespace evidence
- inferred column pitch when scan damage removes alternating rules
- page-wide alignment guides shared across upper and lower sections
- local detection of passages that change from one major column to two small columns
- glyph-support intersection and connected-component topology for short local subcolumns
- page-shape gating so a portrait centre rule is not mistaken for a two-page gutter
- content classification that separates thin spanning rules from actual text ink
- strict local horizontal separators inside narrow columns
- sparse-layout pitch recovery for unusually wide two- or three-column bands
- recursive horizontal and vertical partitioning
- direction-aware traversal for reading order

The earlier two-page YOLO experiment is retained only as annotation history.
It is not part of the production detector.

## Development status

Version `1.1.0.dev0` is an active layout-tree prototype. The two-page seed
benchmark currently reaches 98.79% pairwise reading-order accuracy. See
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for metrics, limitations, and the
pseudo-label/review loop used to expand the benchmark without drawing every
box manually.

## Author

Qinnan Zhou
