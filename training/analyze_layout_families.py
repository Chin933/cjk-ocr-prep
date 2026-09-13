"""Cluster pseudo-labelled pages into structural layout families.

This intentionally uses only the generated layout trees. It helps the review
set cover distinct shapes instead of selecting twelve near-duplicate pages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _walk(root: dict) -> list[tuple[dict, int]]:
    result: list[tuple[dict, int]] = []
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        result.append((node, depth))
        stack.extend((child, depth + 1) for child in node["children"])
    return result


def _features(record: dict) -> tuple[list[float], dict]:
    layout = record["layout"]
    nodes = _walk(layout["root"])
    leaves = [node for node, _ in nodes if not node["children"]]
    text = [node for node in leaves if node["kind"] != "empty"]
    evidence = Counter(item for node, _ in nodes for item in node["evidence"])
    axes = Counter(node["split_axis"] for node, _ in nodes)
    width = layout["width"]
    heights = [
        (node["bbox"][3] - node["bbox"][1]) / layout["height"] for node in text
    ]
    widths = [(node["bbox"][2] - node["bbox"][0]) / width for node in text]
    page_leaf_counts = []
    for page in layout["root"]["children"]:
        page_leaf_counts.append(
            sum(
                not node["children"] and node["kind"] != "empty"
                for node, _ in _walk(page)
            )
        )
    count = max(1, len(nodes))
    vector = [
        len(text),
        len(leaves) - len(text),
        axes["x"],
        axes["y"],
        max(depth for _, depth in nodes),
        float(np.mean(widths)) if widths else 0.0,
        float(np.max(widths)) if widths else 0.0,
        float(np.std(widths)) if widths else 0.0,
        float(np.mean(heights)) if heights else 0.0,
        abs(page_leaf_counts[0] - page_leaf_counts[-1]) if page_leaf_counts else 0,
        evidence["ruling_line"] / count,
        evidence["whitespace"] / count,
        (evidence["text_alignment"] + evidence["page_text_alignment"]) / count,
        evidence["partial_subcolumn_alignment"] / count,
        evidence["inferred_pitch"] / count,
    ]
    description = {
        "text_regions": len(text),
        "empty_regions": len(leaves) - len(text),
        "horizontal_splits": axes["y"],
        "vertical_splits": axes["x"],
        "max_depth": max(depth for _, depth in nodes),
        "partial_subcolumn_nodes": evidence["partial_subcolumn_alignment"],
        "alignment_nodes": evidence["text_alignment"]
        + evidence["page_text_alignment"],
        "inferred_pitch_nodes": evidence["inferred_pitch"],
    }
    return vector, description


def _cluster(matrix: np.ndarray, cluster_count: int) -> tuple[np.ndarray, list[int]]:
    median = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - median), axis=0)
    scale[scale < 1e-6] = np.std(matrix[:, scale < 1e-6], axis=0) + 1e-6
    normalized = (matrix - median) / scale

    medoids = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(medoids) < cluster_count:
        distance = np.min(
            np.stack(
                [np.linalg.norm(normalized - normalized[index], axis=1) for index in medoids]
            ),
            axis=0,
        )
        medoids.append(int(np.argmax(distance)))

    assignments = np.zeros(len(matrix), dtype=int)
    for _ in range(20):
        distances = np.stack(
            [np.linalg.norm(normalized - normalized[index], axis=1) for index in medoids],
            axis=1,
        )
        new_assignments = np.argmin(distances, axis=1)
        new_medoids = []
        for cluster in range(cluster_count):
            members = np.flatnonzero(new_assignments == cluster)
            if not len(members):
                new_medoids.append(medoids[cluster])
                continue
            pairwise = np.linalg.norm(
                normalized[members, None, :] - normalized[None, members, :], axis=2
            )
            new_medoids.append(int(members[int(np.argmin(pairwise.sum(axis=1)))]))
        assignments = new_assignments
        if new_medoids == medoids:
            break
        medoids = new_medoids
    return assignments, medoids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pseudo_labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    source = json.loads(args.pseudo_labels.read_text(encoding="utf-8"))
    records = source["records"]
    extracted = [_features(record) for record in records]
    matrix = np.asarray([item[0] for item in extracted], dtype=np.float64)
    cluster_count = min(max(2, args.clusters), len(records))
    assignments, medoids = _cluster(matrix, cluster_count)

    families = []
    for cluster in range(cluster_count):
        members = np.flatnonzero(assignments == cluster).tolist()
        medoid = medoids[cluster]
        families.append(
            {
                "family": cluster,
                "representative": records[medoid]["image"],
                "size": len(members),
                "representative_features": extracted[medoid][1],
                "members": [records[index]["image"] for index in members],
            }
        )
    families.sort(key=lambda item: item["family"])
    payload = {"page_count": len(records), "family_count": cluster_count, "families": families}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
