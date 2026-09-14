from PIL import Image, ImageDraw

from digitalization import Box, detect_glyph_graph


def test_glyph_graph_separates_local_large_and_small_scales() -> None:
    image = Image.new("RGB", (260, 420), "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, 360, 55):
        draw.rectangle((165, y, 197, y + 35), fill="black")
    for x in (110, 135):
        for y in range(35, 360, 28):
            draw.rectangle((x, y, x + 14, y + 17), fill="black")

    graph = detect_glyph_graph(image, Box(60, 0, 220, 420))

    assert sum(node.scale_class == "large" for node in graph.nodes) >= 5
    assert sum(node.scale_class == "small" for node in graph.nodes) >= 12
    assert any(relation.relation == "annotates" for relation in graph.relations)


def test_glyph_graph_keeps_continuation_and_record_edges_distinct() -> None:
    image = Image.new("RGB", (220, 480), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 80, 130, 260, 310, 360):
        draw.rectangle((125, y, 158, y + 34), fill="black")
    for y in range(35, 390, 27):
        draw.rectangle((90, y, 104, y + 16), fill="black")

    graph = detect_glyph_graph(image)
    relations = {relation.relation for relation in graph.relations}

    assert "continues_to" in relations
    assert "annotates" in relations
    assert "next_record" in relations
    payload = graph.to_dict()
    assert payload["nodes"]
    assert payload["chains"]
    assert len(payload["records"]) == 2
    assert payload["reading_order"] == ["record_00000", "record_00001"]


def test_glyph_graph_splits_a_horizontally_fused_multi_glyph_blob() -> None:
    image = Image.new("RGB", (240, 360), "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, 330, 48):
        draw.rectangle((165, y, 194, y + 31), fill="black")
    draw.rectangle((55, 120, 81, 149), fill="black")
    draw.rectangle((93, 120, 119, 149), fill="black")
    draw.line((81, 134, 93, 134), fill="black", width=1)

    graph = detect_glyph_graph(image)
    fused_area = [
        node for node in graph.nodes if 45 <= node.center_x <= 130 and 110 <= node.center_y <= 160
    ]

    assert len(fused_area) >= 2
    assert max(node.bbox.width for node in fused_area) < 50
