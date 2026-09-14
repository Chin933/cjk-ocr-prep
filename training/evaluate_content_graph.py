"""Run structural invariants on visually reviewed archive pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from digitalization import detect_content_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "training/content_graph_regressions.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "training/images/archive_diverse",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pages = []
    failures = []
    for record in spec["records"]:
        with Image.open(args.images / record["image"]) as source:
            graph = detect_content_graph(source.convert("RGB"))
        streams = [
            stream
            for stream in graph.streams
            if stream.role in {"text", "satellite"}
        ]
        checks = {}
        minimum, maximum = record["text_stream_count"]
        checks["text_stream_count"] = minimum <= len(streams) <= maximum
        if "header" in record:
            header = record["header"]
            count = sum(stream.bbox.y2 <= header["end_y"] for stream in streams)
            checks["header_streams"] = count >= header["min_streams"]
        if "barrier" in record:
            barrier = record["barrier"]
            count = sum(
                stream.bbox.y1 < barrier["top_y"]
                and stream.bbox.y2 > barrier["bottom_y"]
                for stream in streams
            )
            checks["barrier_crossings"] = count <= barrier["max_crossing_streams"]
        if "upper" in record:
            upper = record["upper"]
            count = sum(stream.bbox.y1 < upper["end_y"] for stream in streams)
            checks["upper_streams"] = count >= upper["min_streams"]
        if "lower" in record:
            lower = record["lower"]
            count = sum(stream.bbox.y2 > lower["start_y"] for stream in streams)
            checks["lower_streams"] = count >= lower["min_streams"]
        pages.append(
            {
                "image": record["image"],
                "glyph_scale": round(graph.glyph_scale, 2),
                "text_stream_count": len(streams),
                "group_count": len(graph.groups),
                "checks": checks,
            }
        )
        failures.extend(
            f"{record['image']}:{name}" for name, passed in checks.items() if not passed
        )

    result = {"passed": not failures, "failures": failures, "pages": pages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Content-graph regression failed")


if __name__ == "__main__":
    main()
