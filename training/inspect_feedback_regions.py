"""Produce enlarged image/probability crops around reviewed layout elements."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
targets={'archive_127_p0056':['N0004','N0006','N0007'],
         'archive_113_p0054':['M0039'],'archive_029_p0237':['M0003','M0007'],
         'archive_155_p0396':['N0003','N0035','N0028','N0037']}
out=ROOT/'runs/feedback_regions';out.mkdir(parents=True,exist_ok=True)
for stem,ids in targets.items():
    data=json.loads((ROOT/'training/layout_review/batches/typed_20260918'/f'{stem}.json').read_text())
    image=Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('RGB')
    p=np.load(ROOT/'training/layout_review/batches/notes_100'/f'{stem}.probabilities.npz')['note']
    for e in data['notes']+data['primary_candidates']:
        if e['id'] not in ids:continue
        x1,y1,x2,y2=e['boxes'][0]
        box=(max(0,x1-10),max(0,y1-65),min(image.width,x2+10),min(image.height,y2+65))
        crop=image.crop(box);w,h=crop.size
        plot=Image.new('RGB',(w*6,h*3+28),'#eee')
        plot.paste(crop.resize((w*3,h*3)),(0,28))
        heat=Image.fromarray(np.uint8(p[box[1]:box[3],box[0]:box[2]]*255)).convert('RGB')
        plot.paste(heat.resize((w*3,h*3)),(w*3,28))
        draw=ImageDraw.Draw(plot);draw.text((4,4),stem+' '+e['id'],fill='black')
        draw.rectangle(((x1-box[0])*3,(y1-box[1])*3+28,(x2-box[0])*3,(y2-box[1])*3+28),outline='#f07800',width=2)
        plot.save(out/f'{stem}.{e["id"]}.png')
