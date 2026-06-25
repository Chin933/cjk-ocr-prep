#!/usr/bin/env python3
"""Command-line interface for the vertical OCR pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # suppress paddle noise
    for noisy in ("ppocr", "paddle", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@click.group()
@click.version_option("0.1.0")
def cli() -> None:
    """Vertical OCR – digitise traditional Chinese columnar documents."""


@cli.command("ocr")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["text", "flagged", "json", "tsv"]),
              default="text", show_default=True,
              help="Output format.")
@click.option("--pages", "-p", default=None,
              help="Page range, e.g. '1-5' (1-based, PDF only).")
@click.option("--dpi", default=300, show_default=True,
              help="Rendering resolution for PDFs.")
@click.option("--conf-threshold", default=0.80, show_default=True,
              help="Confidence below which characters are flagged (0–1).")
@click.option("--gap-sensitivity", default=0.05, show_default=True,
              help="Major whitespace threshold fraction (higher = more splits).")
@click.option("--debug-dir", default=None,
              help="Save column-layout debug images to this directory.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def ocr_cmd(
    input_path: str,
    output: str | None,
    fmt: str,
    pages: str | None,
    dpi: int,
    conf_threshold: float,
    gap_sensitivity: float,
    debug_dir: str | None,
    verbose: bool,
) -> None:
    """Digitise INPUT_PATH (PDF or image) and output structured text.

    \b
    Examples:
      # Plain text from a PDF, pages 1-10
      vertical-ocr ocr document.pdf -p 1-10 -o output.txt

      # JSON with confidence scores
      vertical-ocr ocr scan.jpg -f json -o result.json

      # Flag low-confidence chars and save debug layout images
      vertical-ocr ocr document.pdf -f flagged --debug-dir debug/
    """
    _setup_logging(verbose)

    from src.vertical_ocr.pipeline import process_pdf, process_image_file
    from src.vertical_ocr.postprocess import write_plain, write_flagged, write_json, write_tsv
    from src.vertical_ocr.pdf_utils import page_count

    inp = Path(input_path)
    suffix = inp.suffix.lower()

    # Parse page range
    page_range: tuple[int, int] | None = None
    if pages and suffix == ".pdf":
        parts = pages.split("-")
        start = int(parts[0]) - 1  # convert to 0-based
        end = int(parts[1]) if len(parts) > 1 else int(parts[0])
        page_range = (start, end)

    debug_path = Path(debug_dir) if debug_dir else None
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        if suffix == ".pdf":
            total = page_count(inp)
            if page_range:
                total = page_range[1] - page_range[0]
            task = progress.add_task(f"Processing {inp.name}…", total=total)
            for page_result in process_pdf(
                inp,
                page_range=page_range,
                dpi=dpi,
                low_conf_threshold=conf_threshold,
                gap_threshold_frac=gap_sensitivity,
                debug_dir=debug_path,
            ):
                results.append(page_result)
                progress.advance(task)
        else:
            progress.add_task(f"Processing {inp.name}…", total=None)
            results.append(process_image_file(
                inp,
                dpi=dpi,
                low_conf_threshold=conf_threshold,
                gap_threshold_frac=gap_sensitivity,
                debug_dir=debug_path,
            ))

    # Write output
    writers = {
        "text": write_plain,
        "flagged": write_flagged,
        "json": write_json,
        "tsv": write_tsv,
    }
    writer = writers[fmt]

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            writer(results, f)
        console.print(f"[green]OK[/green] Saved to [bold]{out_path}[/bold]")
    else:
        import io
        buf = io.StringIO()
        writer(results, buf)
        click.echo(buf.getvalue())

    # Summary table
    low_conf_pages = [r for r in results if r.low_confidence_columns]
    if low_conf_pages:
        table = Table(title="Low-confidence pages", show_lines=True)
        table.add_column("Page", style="cyan")
        table.add_column("Columns flagged", style="yellow")
        for r in low_conf_pages:
            table.add_row(str(r.page_index + 1), str(len(r.low_confidence_columns)))
        console.print(table)


@cli.command("preview")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("page", type=int, default=1)
@click.option("--dpi", default=150, show_default=True)
@click.option("--output", "-o", default="preview.jpg", show_default=True,
              help="Output image path.")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--layout-debug", is_flag=True, default=False,
              help="Draw frame, separators, blocks, major columns, and columns.")
def preview_cmd(input_path: str, page: int, dpi: int, output: str, verbose: bool,
                layout_debug: bool) -> None:
    """Save a column-layout debug image for PAGE (1-based) of INPUT_PATH."""
    _setup_logging(verbose)

    from src.vertical_ocr.pdf_utils import pdf_to_images
    from src.vertical_ocr.layout import (
        detect_columns,
        detect_layout,
        draw_columns,
        draw_layout_debug,
    )
    from PIL import Image

    inp = Path(input_path)
    suffix = inp.suffix.lower()
    page_idx = page - 1

    if suffix == ".pdf":
        image = None
        for idx, img in pdf_to_images(inp, dpi=dpi, page_range=(page_idx, page_idx + 1)):
            image = img
            break
    else:
        image = Image.open(str(inp)).convert("RGB")

    if image is None:
        console.print(f"[red]Could not load page {page}[/red]")
        sys.exit(1)

    if layout_debug:
        layout = detect_layout(image, dpi=dpi)
        columns = layout.columns
        annotated = draw_layout_debug(image, layout)
    else:
        columns = detect_columns(image, dpi=dpi)
        annotated = draw_columns(image, columns)
    out_path = Path(output)
    annotated.save(str(out_path), quality=85)
    console.print(
        f"[green]OK[/green] {len(columns)} columns detected -- saved to [bold]{out_path}[/bold]"
    )
    for col in columns:
        console.print(
            f"  #{col.index:2d}  {col.col_type:12s}  "
            f"x=[{col.x1}:{col.x2}]  y=[{col.y1}:{col.y2}]  w={col.width}px"
        )


if __name__ == "__main__":
    cli()
