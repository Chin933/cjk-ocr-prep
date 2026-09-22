"""Multi-page ruled-lane / note-first experiment, with untouched user labels."""

import argparse
import json
from pathlib import Path
import sys
import hashlib

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from digitalization.box_detector import BoxSegmenter
from digitalization.ruled_notes import decode_notes
from digitalization.ruling_geometry import refine_tracks,lane_corners
from restore_rules import restore
from train_box_detector import score_boxes

DEFAULT_PAGES=['archive_113_p0054','archive_113_p0221','archive_029_p0237',
               'archive_071_p0056','archive_155_p0396','archive_365_p0048']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pages',nargs='+',default=DEFAULT_PAGES)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/note_first')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/box_detector_v2_final/box_segmenter.pt')
    parser.add_argument('--slanted-rules',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.manifest:
        args.pages=[p['image'] for p in json.loads(args.manifest.read_text(encoding='utf-8'))['pages']]
    fingerprint=hashlib.sha256(args.checkpoint.read_bytes())
    for path in [Path(__file__),ROOT/'training/restore_rules.py',ROOT/'src/digitalization/ruled_notes.py',ROOT/'src/digitalization/ruling_geometry.py']:
        fingerprint.update(path.read_bytes())
    fingerprint.update(str(args.slanted_rules).encode())
    fingerprint=fingerprint.hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,weights_only=True)
    model=BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0]).eval()
    model.load_state_dict(saved['state_dict'])
    reports=[]
    sheet=Image.new('RGB',(420*3,640*2),'#eeeeea')
    draw=ImageDraw.Draw(sheet)
    for page_index,stem in enumerate(args.pages):
        result_path=args.output/f'{stem}.json'
        if args.resume and result_path.exists():
            previous=json.loads(result_path.read_text(encoding='utf-8'))
            if previous.get('pipeline_fingerprint')==fingerprint:
                reports.append({'image':stem,'pitches':[round(b['pitch'],2) for b in previous['bands'] if 'pitch' in b],
                                'bands':sum('pitch' in b for b in previous['bands']),'notes':len(previous['notes']),
                                'primary_candidates':len(previous['primary_candidates']),
                                'matched':previous.get('fit_check',{}).get('matched_iou50')})
                (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
                print(json.dumps({'completed':page_index+1,'total':len(args.pages),'resumed':stem}),flush=True)
                if page_index<6:
                    thumbnail=Image.open(args.output/f'{stem}.notes.png');thumbnail.thumbnail((410,600))
                    x,y=(page_index%3)*420,(page_index//3)*640
                    sheet.paste(thumbnail,(x,y+25));draw.text((x+5,y+5),stem,fill='#111111')
                continue
        image=Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('RGB')
        _,rule_overlay,bands=restore(image)
        if args.slanted_rules:
            bands=refine_tracks(image,bands)
            rule_overlay=image.copy();rules=ImageDraw.Draw(rule_overlay)
            for band in bands:
                for lane in range(len(band.get('boundaries',[]))-1):
                    points=[tuple(p) for p in lane_corners(band,lane).tolist()]
                    rules.line(points+[points[0]],fill='#168aad',width=1)
        rule_overlay.save(args.output/f'{stem}.rules.png')
        pitches=[b['pitch'] for b in bands if 'pitch' in b]
        # Normalize font size to the training lane width, without painting lines.
        scale=saved['scale']*51/float(np.median(pitches)) if pitches else saved['scale']
        gray=np.asarray(image.convert('L'))
        reduced=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            probabilities=model(torch.from_numpy((1-reduced.astype(np.float32)/255)[None,None])).sigmoid()[0].numpy()
        note=cv2.resize(probabilities[1],image.size)
        edge=cv2.resize(probabilities[3],image.size) if len(probabilities)>3 else None
        np.savez_compressed(args.output/f'{stem}.probabilities.npz',note=note,edge=edge)
        notes,primary=decode_notes(image,bands,note,edge)
        overlay=image.copy()
        pen=ImageDraw.Draw(overlay)
        for e in notes:
            if 'polygon' in e:
                pen.polygon([tuple(p) for p in e['polygon']],outline='#ed7900',width=2)
            else:
                pen.rectangle(e['boxes'][0],outline='#ed7900',width=2)
        overlay.save(args.output/f'{stem}.notes.png')
        Image.fromarray((note*255).astype(np.uint8)).save(args.output/f'{stem}.note_probability.png')
        result={'image':stem,'size':list(image.size),'bands':bands,'scale':scale,
                'pipeline_fingerprint':fingerprint,
                'notes':notes,'primary_candidates':primary,
                'primary_semantics':'local_ink_intervals_not_record_envelopes'}
        labels=ROOT/'training/layout_review/annotations'/f'{stem}.user-linked.json'
        if labels.exists():
            gold=json.loads(labels.read_text(encoding='utf-8-sig'))
            result['fit_check']=score_boxes(gold['elements'],notes,0,image.width)['annotation']
        else:
            result['validation']='unseen_page_human_review'
        (args.output/f'{stem}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        thumbnail=overlay.copy();thumbnail.thumbnail((410,600))
        if page_index<6:
            x,y=(page_index%3)*420,(page_index//3)*640
            sheet.paste(thumbnail,(x,y+25));draw.text((x+5,y+5),stem,fill='#111111')
        reports.append({'image':stem,'pitches':[round(p,2) for p in pitches],
                        'bands':len(pitches),'notes':len(notes),'primary_candidates':len(primary),
                        'matched':result.get('fit_check',{}).get('matched_iou50')})
        (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps({'completed':page_index+1,'total':len(args.pages),**reports[-1]}),flush=True)
    sheet.save(args.output/'contact.png')
    (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
