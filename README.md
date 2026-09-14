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
from digitalization import (
    detect_content_graph,
    detect_glyph_graph,
    detect_layout,
    detect_region_graph,
    detect_rule_graph,
)

image = Image.open("page.jpg")
document = detect_layout(image)

for region in document.reading_order:
    print(region.order, region.bbox)

# Optional: inspect the 2-D ruling graph and T-junctions directly.
rule_graph = detect_rule_graph(image)
print(rule_graph.junctions)

# Bounded faces preserve T-junctions that cannot be expressed by a flat cut.
# Each cell includes its true polygon, bounding box, area, neighbours and order.
region_graph = detect_region_graph(image)
print(region_graph.cells)

# Local ink observations are linked into bounded vertical text streams.
content_graph = detect_content_graph(image)
print(content_graph.streams, content_graph.edges)

# Mixed-size fields expose glyph instances and typed structural relations.
glyph_graph = detect_glyph_graph(image)
print(glyph_graph.nodes, glyph_graph.chains, glyph_graph.relations)
```

## Command line

```bash
digitalization-layout page.jpg \
  --output page.layout.json \
  --overlay page.layout.jpg \
  --graph-output page.rules.json \
  --graph-overlay page.rules.jpg \
  --region-graph-output page.regions.json \
  --region-graph-overlay page.regions.jpg \
  --content-graph-output page.content.json \
  --content-graph-overlay page.content.jpg \
  --glyph-graph-output page.glyphs.json \
  --glyph-graph-overlay page.glyphs.jpg
```

The JSON contains both the hierarchy and a flattened list of ordered leaf IDs.
The optional region-graph JSON is deliberately conservative: it emits bounded
faces only where detected rules support them, retaining non-rectangular polygon
geometry instead of expanding every local divider into a page-wide cut.
The overlay numbers the detected reading regions for review.

## Method

The current development version combines:

- printed-frame and center-gutter detection
- multi-scale ruling-line extraction
- local 2-D ruling segments with preserved endpoints and junctions
- planar region faces and adjacency for non-uniform T-junction layouts
- rejection of partial rules incorrectly promoted to page-wide separators
- whitespace evidence
- inferred column pitch when scan damage removes alternating rules
- page-wide alignment guides shared across upper and lower sections
- local detection of passages that change from one major column to two small columns
- glyph-support intersection and connected-component topology for short local subcolumns
- bottom-up multi-scale lane observations linked into bounded vertical text streams
- repeated local primary-stream pitch for grouping mixed-size record cells
- local glyph reconstruction and within-band multi-scale classification
- separate continuation, annotation, and next-record relations
- cumulative-drift constraints that stop tracks migrating into adjacent columns
- horizontal-rule barriers that prevent content tracks from crossing major sections
- content-stream adjacency, attachment relations, and explicit reading edges
- page-shape gating so a portrait centre rule is not mistaken for a two-page gutter
- content classification that separates thin spanning rules from actual text ink
- strict local horizontal separators inside narrow columns
- sparse-layout pitch recovery for unusually wide two- or three-column bands
- recursive horizontal and vertical partitioning
- direction-aware traversal for reading order

The earlier two-page YOLO experiment is retained only as annotation history.
It is not part of the production detector.

## Development status

Version `1.5.0.dev0` is an active layout-tree/content-graph prototype. The two-page seed
benchmark currently reaches 98.57% pairwise reading-order accuracy, 82.86%
major-column recall, and 50.00% local-subcolumn recall at IoU 0.50. See
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for metrics, limitations, and the
pseudo-label/review loop used to expand the benchmark without drawing every
box manually.

On the two visually reviewed archive regressions, the content graph recovers
all 14 reviewed streams on the regular page and all six reviewed upper record
groups on the mixed-size page, each at 100% one-to-one precision and recall.
The irregular lower record field remains the active segmentation target.

## Author

Qinnan Zhou
