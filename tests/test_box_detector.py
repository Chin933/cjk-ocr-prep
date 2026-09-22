import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')
from digitalization.box_detector import BoxSegmenter, predict_boxes, separate_instances


def test_boundary_separates_touching_records_without_erasing_nested_annotation():
    class FixedPrediction(torch.nn.Module):
        def forward(self, image):
            logits = torch.full((1, 3, 80, 80), -10.)
            logits[:, 0, 8:72, 10:70] = 10
            logits[:, 1, 20:50, 17:30] = 10
            logits[:, 2, 8:72, 37:43] = 10
            return logits

    boxes, _ = predict_boxes(Image.new('L', (80, 80), 255), FixedPrediction(), scale=1, constrain=False)
    records = [e['boxes'][0] for e in boxes if e['role'] == 'primary']
    notes = [e['boxes'][0] for e in boxes if e['role'] == 'annotation']
    assert len(records) == 2
    assert len(notes) == 1
    left = min(records, key=lambda b: b[0])
    assert left[0] <= notes[0][0] < notes[0][2] <= left[2]


def test_segmenter_handles_image_dimensions_not_divisible_by_eight():
    model = BoxSegmenter().eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(np.zeros((1, 1, 65, 73), np.float32)))
    assert prediction.shape == (1, 3, 65, 73)


def test_thin_cross_column_bridges_do_not_produce_one_spanning_box():
    support=np.zeros((100,210),np.float32)
    for x in (10,50,90,130,170):
        support[10:90,x:x+28]=.95
    support[65:69,10:198]=.8
    parts=separate_instances(support)
    assert len(parts)==5
    assert all(width<40 for x,y,width,height,area in parts)


def test_genuine_wide_rectangular_instance_is_not_forced_into_equal_columns():
    support=np.zeros((100,210),np.float32)
    support[10:90,10:198]=.95
    parts=separate_instances(support)
    assert len(parts)==1
    assert parts[0][2]>170
