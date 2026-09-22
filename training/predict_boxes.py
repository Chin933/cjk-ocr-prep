"""Run the learned box detector on an image without annotation inputs."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'src'))
from digitalization.box_detector import BoxSegmenter, predict_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/box_detector_current/box_segmenter.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/box_detector_current')
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, weights_only=True)
    model = BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0])
    model.load_state_dict(saved['state_dict'])
    image = Image.open(args.image).convert('RGB')
    boxes, _ = predict_boxes(image, model, saved['scale'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/(args.image.stem+'.predicted.json')).write_text(json.dumps(boxes,ensure_ascii=False,indent=2),encoding='utf-8')
    draw = ImageDraw.Draw(image)
    for e in boxes:
        draw.rectangle(e['boxes'][0],outline='#8745c4' if e['role']=='primary' else '#e27a10',width=1)
    output = args.output/(args.image.stem+'.predicted.png')
    image.save(output)
    print(json.dumps({'image':str(args.image),'primary':sum(e['role']=='primary' for e in boxes),
                      'annotation':sum(e['role']=='annotation' for e in boxes),'overlay':str(output)}))


if __name__=='__main__':
    main()
