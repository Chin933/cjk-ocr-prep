from PIL import Image, ImageDraw

from digitalization import Box, detect_content_graph


def test_content_graph_tracks_independent_local_vertical_streams() -> None:
    image = Image.new("RGB", (360, 500), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, 440, 55):
        draw.rectangle((250, y, 292, y + 34), fill="black")
    for x in (120, 165):
        for y in range(160, 390, 35):
            draw.rectangle((x, y, x + 18, y + 22), fill="black")

    graph = detect_content_graph(image)
    centers = sorted(stream.center_x for stream in graph.streams)

    assert len(graph.streams) == 3
    assert centers[0] < 140
    assert 160 < centers[1] < 190
    assert centers[2] > 250
    assert graph.streams[0].bbox.height != graph.streams[-1].bbox.height
    assert graph.edges


def test_content_graph_keeps_split_radicals_in_one_large_glyph_stream() -> None:
    image = Image.new("RGB", (180, 500), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, 440, 58):
        draw.rectangle((70, y, 80, y + 35), fill="black")
        draw.rectangle((91, y, 103, y + 35), fill="black")

    graph = detect_content_graph(image, Box(30, 0, 140, 500))

    assert len(graph.streams) == 1
    assert graph.streams[0].bbox.x1 <= 70
    assert graph.streams[0].bbox.x2 >= 103


def test_content_graph_serializes_observations_streams_and_order_edges() -> None:
    image = Image.new("RGB", (240, 360), "white")
    draw = ImageDraw.Draw(image)
    for x in (80, 145):
        for y in range(30, 330, 42):
            draw.rectangle((x, y, x + 24, y + 27), fill="black")

    payload = detect_content_graph(image).to_dict()

    assert payload["observations"]
    assert len(payload["streams"]) == 2
    assert len(payload["reading_order"]) == 2
    assert len(payload["groups"]) == 2
    assert len(payload["group_reading_order"]) == 2
    assert payload["edges"][0]["relation"] == "precedes"
