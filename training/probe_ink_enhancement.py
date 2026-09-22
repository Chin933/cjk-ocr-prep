"""Compare mild grayscale enhancement with the fixed layout checkpoint."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from digitalization.box_detector import BoxSegmenter


def main():
    output=ROOT/'runs/ink_enhancement_probe';output.mkdir(parents=True,exist_ok=True)
    stem='archive_155_p0396'
    gray=np.asarray(Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('L'))
    data=json.loads((ROOT/'training/layout_review/batches/typed_feedback_20260918'/f'{stem}.json').read_text())
    saved=torch.load(ROOT/'runs/box_detector_v2_final/box_segmenter.pt',weights_only=True)
    model=BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0]).eval()
    model.load_state_dict(saved['state_dict']);torch.set_num_threads(2)
    scale=saved['scale']*51/np.median([b['pitch'] for b in data['bands'] if 'pitch' in b])
    versions={'original':gray,'local_contrast':cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(gray),
              'darken_faint':np.uint8(255*(gray.astype(np.float32)/255)**1.5)}
    regions={'main_zhuang':[236,582,284,609],'note_hua_lin_sheng':[236,616,284,668],
             'main_yong':[292,677,340,702],'empty_box':[236,556,284,578]}
    canvas=Image.new('RGB',(360*3,540),'white');draw=ImageDraw.Draw(canvas)
    report={}
    for i,(name,pixels) in enumerate(versions.items()):
        reduced=cv2.resize(pixels,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            p=model(torch.from_numpy((1-reduced.astype(np.float32)/255)[None,None])).sigmoid()[0,1].numpy()
        p=cv2.resize(p,(gray.shape[1],gray.shape[0]))
        report[name]={key:round(float(p[y1:y2,x1:x2].mean()),4) for key,(x1,y1,x2,y2) in regions.items()}
        crop=Image.fromarray(pixels).convert('RGB').crop((228,550,348,720)).resize((360,510))
        canvas.paste(crop,(i*360,30));draw.text((i*360+5,6),name,fill='black')
    canvas.save(output/'comparison.png')
    (output/'response.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
