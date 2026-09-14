# Layout benchmark and annotation loop

The benchmark measures page structure, not character recognition. OCR text is
intentionally excluded.

## Seed truth

`training/pptx_layout_dataset/annotations.json` preserves the two PowerPoint
pages drawn by hand. The three nested labels are `block`, `major_column`, and
`subcolumn`. They are used as a seed benchmark, not as sufficient training
data, a fixed ontology, or evidence of generalization. The production output
is an arbitrary-depth partition tree. New archive page types may introduce
additional nesting rather than being forced into those three labels.

Run the benchmark with:

```bash
python training/evaluate_pptx_benchmark.py --output runs/pptx_benchmark_current.json
```

The evaluator reports best-node IoU recall for every structural level and
pairwise reading-order accuracy among peers in the same block.

## Scaling without manual box drawing

The detector creates tree pseudo-labels for all available pages:

```bash
python training/generate_pseudo_labels.py training/images/all \
  --output runs/pseudo_labels_current.json
python training/audit_layout_dataset.py \
  --pseudo-labels runs/pseudo_labels_current.json \
  --output runs/layout_audit_current.json
```

The audit ranks a 12-page review queue using structural warnings, unusually
wide regions, and reliance on inferred evidence. Review effort is spent only
on this queue. Corrected trees become benchmark truth; the detector is rerun,
and the queue is rebuilt. This is an active-error-correction loop rather than
manual annotation of every image.

The pseudo-labelled corpus is also clustered by tree depth, split axes, region
geometry, and evidence types:

```bash
python training/analyze_layout_families.py runs/pseudo_labels_current.json \
  --output runs/layout_families_current.json --clusters 8
```

One representative from every family must be reviewed, so a large family of
easy pages cannot crowd out rare shapes.

The source archive is sampled separately with a low-resolution screening pass
and farthest-first structural novelty selection:

```bash
python training/sample_archive_diversity.py D:/Programs/Keju/Zhujuanjicheng/1_420 \
  --existing runs/pseudo_labels_current.json \
  --output-dir training/images/archive_diverse \
  --report runs/archive_diversity_current.json
```

PyMuPDF is needed only for this archive-development command, never by the core
package.

## Programmatic shape stress test

`training/synthetic_benchmark.py` generates exact trees for ten layout
grammars. Besides regular columns, it covers unequal upper/lower bands, side
titles, bands with different column counts, local subcolumns, asymmetric
nested regions, spanning headings, columns that change structure at different
heights, narrow heading/colophon bands, and multi-level unequal columns. Rules
may be faint or broken. This supplies unlimited geometry examples without
pretending that synthetic marks are OCR training data.

```bash
python training/synthetic_benchmark.py \
  --output runs/synthetic_benchmark_current.json --cases 40 \
  --min-region-recall 0.65 --min-order-accuracy 0.90 \
  --min-template-recall 0.45 --min-template-order 0.88
```

## Current seed result

On the two PowerPoint pages, version `1.4.0.dev0` currently reaches 98.57%
pairwise reading-order accuracy. IoU@0.50 recall is 90.00% for blocks, 82.86%
for major columns, and 50.00% for local subcolumns. Recovering parent columns
around narrow aligned half-lanes raises major-column IoU@0.75 recall from
28.57% to 55.71% without changing leaf order. On 40 unseen generated layouts
spanning ten grammars, region IoU@0.50 recall is 76.25% and pairwise order
accuracy is 96.52%. The local 2-D ruling graph raises the independent
within-column grammar from 51.04% to 64.58% on the same fixed seeds. Its
segments retain endpoints and junctions, preventing a short local rule from
being silently extended across the page. This per-grammar result remains a
development target and cannot be hidden by easier regular-column cases.

The planar region graph is a companion to the recursive tree, not a replacement
score hidden inside these figures. It converts high-confidence local rules into
bounded faces, preserves each face's polygon and adjacency, and inherits content
order from the tree. A T-junction test verifies that a divider on one side of a
page does not split its neighbour. Pages without enough explicit ruling evidence
remain the responsibility of the content-alignment tree.

Across all 76 current real images, the detector now averages 50.58 text regions
per image, with no automatic audit flags. A second target-only archive sample
contains 40 pages selected from 31 volumes and eight detected layout families.
The archive sample is a diversity and failure-discovery set, not ground truth:
its current mean is 23.25 text regions per page, and its 25 wide-region warnings
show that dense mixed-size pages remain the next major under-segmentation target.

Error diagnosis on the seed truth separates failures by geometry. Of the 52
remaining subcolumn misses, 28 are unsplit horizontal merges and 15 have an
overlong vertical extent; these require different evidence. The current local
2-D glyph-support model improved subcolumn recall from 38.46% to 50.00%
without OCR, while a mechanical adjacent-column grouping experiment was
rejected because it reduced independent synthetic reading-order accuracy.

The next-stage content graph is evaluated separately on visually reviewed
archive pages. It starts from overlapping local ink observations, estimates
glyph scale, links observations one-to-one into bounded vertical streams, and
treats long horizontal rules as barriers that tracks cannot cross. The first
two regression pages require recovery of marginal headings, independent upper
and lower stream populations, and zero tracks crossing major section rules.
The regular page currently matches all 14 manually reviewed streams with 100%
one-to-one precision and recall. On the mixed-size page, repeated local
primary-stream pitch groups the six reviewed upper records with 100% precision
and recall while keeping the horizontal section rule uncrossed. The lower
record field is deliberately not assigned a headline accuracy yet: it still
contains overlapping large-name streams, smaller annotation columns, and
streams that begin and end at different heights. Its current regression guards
against reintroducing gross multi-column merges while the 2-D grouping model is
developed.
