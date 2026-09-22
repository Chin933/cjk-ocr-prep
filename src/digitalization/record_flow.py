"""Infer local ownership and column wraps from record-envelope geometry.

Input elements supply only id, role and boxes. Human parent labels, relations,
transcriptions and review states are deliberately not inputs to inference.
Scores express geometric support, not calibrated probabilities.
"""

from __future__ import annotations

from statistics import median


def _area(boxes):
    """Area of a rectangle union, including overlapping multipart envelopes."""
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        end = float('-inf')
        height = 0.0
        for top, bottom in intervals:
            height += max(0, bottom - max(top, end))
            end = max(end, bottom)
        area += (right - left) * height
    return area


def _intersection(left, right):
    result = []
    for a in left:
        for b in right:
            box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)
    return result


def _bounds(element):
    boxes = element['boxes']
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _cx(element):
    box = _bounds(element)
    return (box[0] + box[2]) / 2


def _coverage(element, container):
    return _area(_intersection(element['boxes'], container['boxes'])) / _area(element['boxes'])


def _lanes(elements, width):
    groups = []
    for element in sorted(elements, key=_cx, reverse=True):
        if not groups or abs(_cx(element) - median(_cx(e) for e in groups[-1])) > width * .45:
            groups.append([])
        groups[-1].append(element)
    return [sorted(group, key=lambda e: _bounds(e)[1]) for group in groups]


def infer_record_flow(elements: list[dict]) -> dict:
    """Infer relationships in manually or automatically detected envelopes.

    Requires explicit region elements and record-envelope primary boxes.
    Incomplete or ambiguous geometry remains unresolved. It does not perform
    page detection, OCR, or infer exact character order within annotation boxes.
    """
    geometry = []
    seen = set()
    for element in elements:
        identifier, role, boxes = element['id'], element['role'], element['boxes']
        if identifier in seen or not boxes or any(len(b) != 4 or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError('Element ids must be unique and boxes must have positive area')
        seen.add(identifier)
        geometry.append({'id': identifier, 'role': role, 'boxes': [list(b) for b in boxes]})
    regions = [e for e in geometry if e['role'] == 'region']
    primaries = [e for e in geometry if e['role'] == 'primary']
    annotations = [e for e in geometry if e['role'] == 'annotation']
    by_id = {e['id']: e for e in geometry}
    region_of = {}
    for e in primaries + annotations:
        candidates = sorted(((_coverage(e, r), r['id']) for r in regions), reverse=True)
        if candidates and candidates[0][0] >= .65:
            region_of[e['id']] = candidates[0][1]

    ownership = []
    for note in annotations:
        candidates = []
        for primary in primaries:
            if note['id'] not in region_of or region_of.get(primary['id']) != region_of[note['id']]:
                continue
            coverage = _coverage(note, primary)
            if coverage >= .65:
                candidates.append({'parent_id': primary['id'], 'coverage': round(coverage, 4)})
        candidates.sort(key=lambda c: (-c['coverage'], _area(by_id[c['parent_id']]['boxes']), c['parent_id']))
        margin = candidates[0]['coverage'] - candidates[1]['coverage'] if len(candidates) > 1 else 1
        accepted = bool(candidates and margin >= .12)
        ownership.append({'source': note['id'], 'parent_id': candidates[0]['parent_id'] if accepted else None,
                          'candidates': candidates, 'status': 'inferred' if accepted else 'ambiguous' if candidates else 'unresolved'})

    duplicates = []
    for index, left in enumerate(primaries):
        for right in primaries[index + 1:]:
            intersection = _area(_intersection(left['boxes'], right['boxes']))
            iou = intersection / (_area(left['boxes']) + _area(right['boxes']) - intersection)
            if iou >= .8:
                duplicates.append({'left': left['id'], 'right': right['id'], 'iou': round(iou, 4)})

    continuations, primary_candidates, lane_summaries = [], [], []
    for region in regions:
        notes = [e for e in annotations if region_of.get(e['id']) == region['id']]
        mains = [e for e in primaries if region_of.get(e['id']) == region['id']]
        if not notes:
            continue
        width = median(b[2] - b[0] for e in notes for b in e['boxes'])
        bounds = _bounds(region)
        lanes = _lanes(notes, width)
        lane_summaries.append({'region': region['id'], 'annotation_lane_centers': [round(median(_cx(e) for e in g), 2) for g in lanes]})
        for right, left in zip(lanes, lanes[1:]):
            source = max(right, key=lambda e: _bounds(e)[3])
            target = min(left, key=lambda e: _bounds(e)[1])
            delta = _cx(source) - _cx(target)
            bottom_gap = bounds[3] - _bounds(source)[3]
            top_gap = _bounds(target)[1] - bounds[1]
            if .55 * width <= delta <= 1.65 * width and bottom_gap <= .7 * width and top_gap <= 1.2 * width:
                continuations.append({'source': source['id'], 'target': target['id'], 'type': 'annotation_continues',
                                      'region': region['id'], 'evidence': {'column_step': round(delta / width, 3),
                                      'bottom_gap_in_column_widths': round(bottom_gap / width, 3),
                                      'top_gap_in_column_widths': round(top_gap / width, 3)}, 'status': 'inferred'})
        # A short main fragment at a column end may continue at the next head.
        # Shape alone cannot establish that it is an unfinished name: keep a candidate.
        for right, left in zip(_lanes(mains, width), _lanes(mains, width)[1:]):
            source = max(right, key=lambda e: _bounds(e)[3])
            target = min(left, key=lambda e: _bounds(e)[1])
            a, b = _bounds(source), _bounds(target)
            if not (.55 * width <= _cx(source) - _cx(target) <= 1.65 * width
                    and bounds[3] - a[3] <= .7 * width and b[1] - bounds[1] <= 1.2 * width
                    and a[3] - a[1] <= 1.3 * width):
                continue
            if any(_coverage(note, source) >= .65 for note in notes):
                continue
            head = {'boxes': [[b[0], b[1], b[2], min(b[3], b[1] + .6 * width)]]}
            if any(_coverage(head, note) >= .65 for note in notes):
                continue
            primary_candidates.append({'source': source['id'], 'target': target['id'], 'type': 'primary_continues',
                                       'region': region['id'], 'status': 'candidate',
                                       'reason': 'short_main_at_column_end_to_main_at_next_head'})

    # Local envelopes on two sides of a confirmed geometric wrap can be fragments
    # of the same logical record. Preserve every input box and id in the result.
    local_owner = {o['source']: o['parent_id'] for o in ownership}
    roots = {e['id']: e['id'] for e in primaries}

    def root(identifier):
        while roots[identifier] != identifier:
            identifier = roots[identifier]
        return identifier

    for edge in continuations:
        source, target = local_owner[edge['source']], local_owner[edge['target']]
        if source and target:
            roots[root(target)] = root(source)
    records = {}
    for p in primaries:
        records.setdefault(root(p['id']), []).append(p['id'])
    return {'stage': 'relationships_given_record_envelopes', 'region_of': region_of,
            'ownership': ownership, 'annotation_continuations': continuations,
            'primary_continuation_candidates': primary_candidates,
            'logical_record_candidates': list(records.values()), 'near_duplicate_envelopes': duplicates,
            'lanes': lane_summaries}
