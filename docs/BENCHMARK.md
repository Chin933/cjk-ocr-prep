# Layout benchmark and annotation loop

The benchmark measures page structure, not character recognition. OCR text is
intentionally excluded.

## Seed truth

`training/pptx_layout_dataset/annotations.json` preserves the two PowerPoint
pages drawn by hand. The three nested labels are `block`, `major_column`, and
`subcolumn`. They are used as a seed benchmark, not as sufficient training
data or evidence of generalization.

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
python training/audit_layout_dataset.py training/images/all \
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

On the two PowerPoint pages, version `1.1.0.dev0` currently reaches 97.58%
pairwise reading-order accuracy. IoU@0.50 recall is 90.00% for blocks, 64.29%
for major columns, and 38.46% for local subcolumns. On 40 unseen generated
layouts spanning ten grammars, region IoU@0.50 recall is 72.30% and pairwise
order accuracy is 96.17%. The harder benchmark deliberately lowered aggregate
recall: independent within-column structure is currently the weakest grammar
at 51.04% region recall. This per-grammar result is a development target and
cannot be hidden by easier regular-column cases.
