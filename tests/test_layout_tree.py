from PIL import Image, ImageDraw

from digitalization import detect_layout


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
    assert result.reading_order[-1].bbox.x2 <= 300


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
