# Digitalization

Digitalization is a Python program for layout detection and OCR preparation in historical Chinese documents.

The program targets pages written in vertical Traditional Chinese columns. These pages often contain title blocks, page frames, horizontal section rules, large text blocks, and nested columns. General OCR systems often fail on this material because they detect text strokes but miss the page structure.

Digitalization starts from the structure of the page. It uses projection profiles and morphology to detect frames, separators, blocks, and columns before OCR.

## Author

Qinnan Zhou

## What the program does

Digitalization prepares scanned historical Chinese pages for OCR.

It can:

- detect the printed page frame
- detect horizontal section separators
- split a page into large blocks
- detect major vertical columns inside each block
- detect smaller text lanes when needed
- create debug images for checking layout detection
- prepare page regions for downstream OCR

## Why this project exists

Historical Chinese archives often use complex vertical layouts. A single page may contain main text, titles, annotations, nested columns, and genealogy-style divisions.

Most document layout models are trained on modern pages. They work well for paragraphs, titles, tables, and figures. They work less well for archive pages with vertical text and nested reading order.

This project uses the geometry of the page before OCR. It reads whitespace, ruling lines, and column structure as layout signals.

## Current version

It includes:

- a hierarchical projection-based layout detector
- scripts for extracting pages from PDFs
- scripts for generating layout debug images
- scripts for preparing experimental training data
- early YOLO training code for later experiments

The projection-based detector is the main baseline in this version. The small YOLO experiment trained from two labeled pages is kept as an experiment, not as the main model.

## Basic use

Install dependencies:

```bash
pip install -r requirements.txt
```

Preview layout detection:

```bash
python cli.py preview input.jpg 1 --layout-debug --output output/layout_debug.jpg
```

Run OCR on a PDF:

```bash
python cli.py ocr input.pdf -p 1-5 -o output.txt
```

Run OCR on one image:

```bash
python cli.py ocr scan.jpg -f text -o result.txt
```

Save debug images during OCR:

```bash
python cli.py ocr input.pdf --debug-dir output/debug -o output.txt
```

## Project structure

```text
src/vertical_ocr/
  layout.py        Layout detection
  pdf_utils.py     PDF to image conversion
  ocr_engine.py    OCR wrapper
  postprocess.py   Text assembly
  pipeline.py      Full pipeline

training/
  extract_pages.py
  prepare_projection_dataset.py
  train_projection.py
  prepare_pptx_layout_dataset.py
  train_pptx_layout.py

docs/
  Notes and annotation instructions

models/
  Local model files

runs/
  Local experiment outputs
```

## Training data

The current project includes scripts for preparing training data. The recommended label set is small and hierarchical:

- block
- major column
- subcolumn
- separator

The goal is not to label every strip of ink. The goal is to label the reading structure of the page.

## Status

This repository is a research prototype. It is designed for historical Chinese documents with vertical text and complex page structure.

The main contribution is the layout pipeline. The OCR engine can be replaced as better OCR systems become available.
