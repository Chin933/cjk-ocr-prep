from PIL import Image, ImageDraw

from digitalization import (
    Box,
    LayoutConfig,
    LayoutNode,
    detect_layout,
    detect_region_graph,
    detect_rule_graph,
)
from digitalization.layout import _binarize, _split_partial_subcolumns


def _synthetic_page() -> Image.Image:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 590, 390), outline="black", width=5)
    draw.line((300, 10, 300, 390), fill="black", width=5)
    draw.line((300, 200, 590, 200), fill="black", width=4)
    for x in (360, 430, 500):
        draw.line((x, 12, x, 388), fill="black", width=3)
    for x in (70, 140, 210):
        draw.line((x, 12, x, 388), fill="black", width=3)
    for x in range(330, 570, 35):
        draw.rectangle((x, 40, x + 10, 150), fill="black")
    for x in range(330, 570, 35):
        draw.rectangle((x, 240, x + 10, 350), fill="black")
    return image


def test_detects_pages_and_stable_reading_order() -> None:
    result = detect_layout(_synthetic_page())
    assert result.root.kind == "spread"
    assert len(result.root.children) == 2
    assert result.root.children[0].bbox.x1 >= result.root.children[1].bbox.x2
    orders = [node.order for node in result.reading_order]
    assert orders == list(range(len(orders)))
    # The right page is traversed before the left page.
    assert result.reading_order[0].bbox.x1 >= 300
    # The ruled but content-free left page contributes no reading item.
    assert all(node.bbox.x1 >= 300 for node in result.reading_order)


def test_serialization_contains_tree_and_order() -> None:
    result = detect_layout(_synthetic_page()).to_dict()
    assert result["root"]["children"]
    assert isinstance(result["reading_order"], list)


def test_layout_core_has_no_ocr_import() -> None:
    import digitalization.layout as layout

    assert "paddleocr" not in layout.__dict__
    assert "rapidocr_onnxruntime" not in layout.__dict__


def test_recovers_unruled_vertical_text_lanes() -> None:
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 890, 690), outline="black", width=4)
    for page_start in (30, 470):
        for column in range(4):
            x = page_start + column * 95
            for y in range(70, 640, 48):
                draw.rectangle((x, y, x + 35, y + 30), fill="black")

    result = detect_layout(image)
    evidence = [item for node in result.reading_order for item in node.evidence]
    assert len(result.root.children) == 2
    assert len(result.reading_order) >= 8
    assert any(item in evidence for item in ("text_alignment", "page_text_alignment"))


def test_rule_graph_retains_partial_line_extents_and_junctions() -> None:
    graph = detect_rule_graph(_synthetic_page())
    assert graph.horizontal
    assert graph.vertical
    assert graph.junctions
    # The horizontal divider exists only on the right-hand page.  A 1-D
    # projection would lose this fact and silently extend it across the spread.
    assert any(
        segment.end - segment.start < graph.frame.width * 0.75
        for segment in graph.horizontal
    )
    assert graph.to_dict()["junctions"]


def test_planar_region_graph_keeps_t_junction_cells_local() -> None:
    graph = detect_region_graph(_synthetic_page())
    left = [cell for cell in graph.cells if cell.bbox.x2 <= 300]
    right = [cell for cell in graph.cells if cell.bbox.x1 >= 300]

    assert len(left) == 4
    assert len(right) == 8
    assert all(cell.bbox.height > 300 for cell in left)
    assert all(cell.bbox.height < 250 for cell in right)
    assert all(cell.area > 0 and len(cell.polygon) >= 4 for cell in graph.cells)
    assert all(
        cell.id in next(item for item in graph.cells if item.id == neighbor).neighbors
        for cell in graph.cells
        for neighbor in cell.neighbors
    )
    assert len(graph.to_dict()["reading_order"]) == 8


def test_planar_region_graph_preserves_concave_face_geometry() -> None:
    image = Image.new("RGB", (420, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 410, 410), outline="black", width=5)
    draw.line((210, 10, 210, 210), fill="black", width=4)
    draw.line((210, 210, 410, 210), fill="black", width=4)

    graph = detect_region_graph(image)
    concave = [
        cell
        for cell in graph.cells
        if cell.area < cell.bbox.width * cell.bbox.height * 0.80
    ]
    assert len(graph.cells) == 2
    assert len(concave) == 1
    assert len(concave[0].polygon) >= 6


def test_local_2d_support_recovers_short_nested_subcolumns() -> None:
    image = Image.new("RGB", (100, 500), "white")
    draw = ImageDraw.Draw(image)
    # A normal centered passage above and below a short, genuinely two-lane
    # passage.  The local structure must not be averaged over the full column.
    for y in (25, 75, 375, 425):
        draw.rectangle((32, y, 68, y + 28), fill="black")
    for x in (12, 60):
        for y in (155, 200, 245):
            draw.rectangle((x, y, x + 27, y + 27), fill="black")

    binary = _binarize(image)
    node = LayoutNode("region_test", "region", Box(0, 0, 100, 500))
    counter = iter(range(100))
    split = _split_partial_subcolumns(
        node,
        binary,
        binary,
        node.bbox,
        LayoutConfig(),
        900,
        lambda prefix: f"{prefix}_{next(counter)}",
    )

    assert split
    nested = [child for band in node.children for child in band.children]
    assert len(nested) >= 2
    assert any("partial_subcolumn_alignment" in child.evidence for child in nested)


def test_portrait_center_rule_is_not_mistaken_for_a_book_gutter() -> None:
    image = Image.new("RGB", (500, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 470, 770), outline="black", width=5)
    draw.line((250, 30, 250, 770), fill="black", width=5)
    for x in (80, 160, 310, 390):
        for y in range(80, 720, 55):
            draw.rectangle((x, y, x + 24, y + 30), fill="black")

    result = detect_layout(image)

    assert len(result.root.children) == 1
    assert result.root.children[0].kind == "page"
