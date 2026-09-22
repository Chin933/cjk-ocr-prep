"""Run extracted instruction figures as explicitly cropped structural fixtures."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'src'))
from digitalization.page_structure import decode_page
from digitalization.box_detector import BoxSegmenter
from restore_rules import fit_lattice

out=ROOT/'training/layout_review/batches/instruction_20260918';out.mkdir(parents=True,exist_ok=True)
figures=[(6,'personal_biography'),(7,'personal_biography'),(23,'exam'),(27,'exam'),(35,'poem')]
torch.set_num_threads(2)
saved=torch.load(ROOT/'runs/box_detector_v2_final/box_segmenter.pt',weights_only=True)
model=BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0]).eval();model.load_state_dict(saved['state_dict'])
reports=[]
for number,kind in figures:
    image=Image.open(ROOT/'runs/instruction_review'/f'image{number}.png').convert('RGB')
    gray=np.asarray(image.convert('L'));h,w=gray.shape;ink=(gray<180).astype(np.uint8)
    bands=[]
    if kind=='exam':
        # These source figures are individual examiner records, not whole
        # ruled pages. Their crop is the supplied record region.
        bands=[{'band':[0,0,w,h],'boundaries':[0.,float(w)],'pitch':float(w),'pitch_source':'source_record_crop'}]
    else:
        lines=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(20,round(w*.65))),np.uint8))
        ys=np.flatnonzero(lines.sum(1)>w*.5);cuts=[0]
        for y in ys:
            if y-cuts[-1]>12 and h-y>12:cuts.append(int(y))
        cuts.append(h)
        for top,bottom in zip(cuts,cuts[1:]):
            if bottom-top<h*.15:continue
            fit=fit_lattice(ink[top:bottom],1,w-2)
            if fit is None:continue
            origin,pitch,count,agreement=fit
            bands.append({'band':[1,top,w-2,bottom],'pitch':pitch,'boundaries':np.linspace(origin,origin+count*pitch,count+1).tolist(),
                          'pitch_source':'cropped_figure_corridors','fit_contrast':agreement})
    pitches=[b['pitch'] for b in bands]
    scale=min(2.,saved['scale']*51/np.median(pitches)) if pitches else saved['scale']
    reduced=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    with torch.inference_mode():p=model(torch.from_numpy((1-reduced.astype(np.float32)/255)[None,None])).sigmoid()[0,1].numpy()
    p=cv2.resize(p,image.size)
    notes,primary,headings=decode_page(image,bands,p,kind)
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    for prefix,elements,color in [('N',notes,'#ec7900'),('M',primary,'#8745c4')]:
        for i,e in enumerate(elements,1):
            e['id']=f'{prefix}{i:04d}'
            if e.get('polygon'):draw.polygon([tuple(p) for p in e['polygon']],outline=color,width=1)
            else:draw.rectangle(e['boxes'][0],outline=color,width=1)
    overlay.save(out/f'image{number}.result.png');image.save(out/f'image{number}.source.png')
    report={'figure':number,'kind':kind,'bands':bands,'notes':notes,'primary':primary,'headings':headings,
            'scope':'source_figure_crop_not_full_page','validation':'assistant_visual_probe_no_gold'}
    (out/f'image{number}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    reports.append({'figure':number,'kind':kind,'bands':len(bands),'notes':len(notes),'primary':len(primary),'headings':len(headings)})
    print(reports[-1],flush=True)
(out/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
html='''<!doctype html><meta charset="utf-8"><title>说明文档特殊版式测试</title><style>body{font:16px system-ui;background:#eee;padding:20px}section{background:white;margin:20px 0;padding:16px}img{max-width:46%;vertical-align:top;margin:1%}h2{font-size:18px}</style><h1>说明文档特殊版式测试</h1><p>左侧原始插图，右侧识别结果。紫色主文，橙色附注。图中裁切范围来自文档，不代表已自动定位整页；这是压力测试，尚未全部通过。</p>'''
for r in reports:html+=f'<section><h2>原文插图 {r["figure"]} · {r["kind"]}</h2><img src="image{r["figure"]}.source.png"><img src="image{r["figure"]}.result.png"></section>'
(out/'index.html').write_text(html,encoding='utf-8')
