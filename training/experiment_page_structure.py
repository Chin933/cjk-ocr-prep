"""Frozen-probability, type-adaptive geometry regression against the 100-page run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from digitalization.page_structure import decode_page,exam_rule_bands
from digitalization.ruling_geometry import refine_tracks,lane_corners
from restore_rules import restore
from train_box_detector import score_boxes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pages',nargs='+')
    parser.add_argument('--types',type=Path,default=ROOT/'training/layout_review/page_types_20260918.json')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reuse-bands',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cache',type=Path,default=ROOT/'training/layout_review/batches/notes_100')
    parser.add_argument('--extra-cache',type=Path,default=ROOT/'training/layout_review/batches/special_baseline_20260917')
    parser.add_argument('--output',type=Path,default=ROOT/'training/layout_review/batches/typed_20260918')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    entries=json.loads(args.types.read_text(encoding='utf-8'))['pages']
    types={e['image']:e for e in entries}
    if args.pages is None:args.pages=list(types)
    fingerprint=hashlib.sha256()
    for path in [Path(__file__),ROOT/'src/digitalization/page_structure.py',ROOT/'src/digitalization/ruled_notes.py',ROOT/'src/digitalization/ruling_geometry.py',ROOT/'training/restore_rules.py']:
        fingerprint.update(path.read_bytes())
    fingerprint.update(args.types.read_bytes())
    model=None
    if args.checkpoint:
        import torch
        import cv2
        cv2.setNumThreads(1)
        from digitalization.box_detector import BoxSegmenter
        torch.set_num_threads(2)
        saved=torch.load(args.checkpoint,weights_only=True)
        model=BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0]).eval()
        model.load_state_dict(saved['state_dict'])
        fingerprint.update(args.checkpoint.read_bytes())
    reports=[]
    completed={}
    if args.resume and (args.output/'summary.json').exists():
        completed={r['image']:r for r in json.loads((args.output/'summary.json').read_text(encoding='utf-8'))}
    for stem in args.pages:
        if stem in completed:
            reports.append(completed[stem])
            continue
        image=Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('RGB')
        cache=args.cache if (args.cache/f'{stem}.json').exists() else args.extra_cache
        prior=json.loads((cache/f'{stem}.json').read_text(encoding='utf-8'))
        probability=np.load(cache/f'{stem}.probabilities.npz')['note']
        primary_probability=None
        geometry_path=args.baseline/f'{stem}.json' if args.baseline else None
        if args.reuse_bands:
            geometry=json.loads(geometry_path.read_text(encoding='utf-8')) if geometry_path and geometry_path.exists() else prior
            bands=geometry['bands']
        else:
            _,_,bands=restore(image);bands=refine_tracks(image,bands)
        if model is not None:
            pitches=[b['pitch'] for b in bands if 'pitch' in b]
            scale=saved['scale']*51/float(np.median(pitches)) if pitches else saved['scale']
            gray=np.asarray(image.convert('L'))
            reduced=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            with torch.inference_mode():
                probabilities=model(torch.from_numpy((1-reduced.astype(np.float32)/255)[None,None])).sigmoid()[0].numpy()
            probability=cv2.resize(probabilities[1],image.size)
            primary_probability=cv2.resize(probabilities[0],image.size)
            np.savez_compressed(args.output/f'{stem}.probabilities.npz',note=probability,
                                primary=cv2.resize(probabilities[0],image.size))
        kind=types[stem]['kind']
        if kind=='exam':bands=exam_rule_bands(image,bands)
        valid=[b for b in bands if 'boundaries' in b]
        notes,primary,headings=decode_page(image,bands,probability,kind,primary_probability=primary_probability)
        for prefix,elements in [('N',notes),('M',primary)]:
            elements.sort(key=lambda e:(e.get('band',0),-e['boxes'][0][0],e['boxes'][0][1]))
            for i,e in enumerate(elements,1):e['id']=f'{prefix}{i:04d}'
        rules=image.copy();pen=ImageDraw.Draw(rules)
        for bi,band in enumerate(bands):
            if 'boundaries' not in band:continue
            # Virtual rules stop at merged cells; they are not painted on text.
            for li in range(len(band['boundaries'])-1):
                points=lane_corners(band,li)
                for a,b in [(points[0],points[3]),(points[1],points[2])]:
                    for y in range(round(a[1]),round(b[1])):
                        t=(y-a[1])/max(1,b[1]-a[1]);x=a[0]+t*(b[0]-a[0])
                        if not any(e['boxes'][0][0]<x<e['boxes'][0][2] and e['boxes'][0][1]-4<=y<=e['boxes'][0][3]+4 for e in headings):
                            pen.point((round(x),y),fill='#168aad')
        rules.save(args.output/f'{stem}.rules.png')
        overlay=image.copy();pen=ImageDraw.Draw(overlay)
        for e in notes+primary:
            color='#ec7900' if e['role']=='annotation' else '#8745c4'
            if 'polygon' in e:pen.polygon([tuple(p) for p in e['polygon']],outline=color,width=2)
            else:pen.rectangle(e['boxes'][0],outline=color,width=2)
        overlay.save(args.output/f'{stem}.notes.png')
        result={'image':stem,'size':list(image.size),'bands':bands,'notes':notes,'primary_candidates':primary,
                'layout_mode':kind,'page_kind':kind,'case':types[stem]['case'],
                'type_source':'reviewed_page_manifest_not_automatic_classifier',
                'merged_headings':headings,'pipeline_fingerprint':fingerprint.hexdigest(),
                'probability_source_fingerprint':prior.get('pipeline_fingerprint'),
                'baseline_file':str((args.baseline if args.baseline and (args.baseline/f'{stem}.json').exists() else cache).resolve()/f'{stem}.json'),
                'checkpoint':str(args.checkpoint) if args.checkpoint else None,
                'primary_semantics':'local_text_streams_not_record_ownership',
                'validation':('feedback_development_page_not_independent_test' if cache==args.cache
                              else 'new_page_visual_check_without_human_gold')}
        labels=ROOT/'training/layout_review/annotations'/f'{stem}.user-linked.json'
        if labels.exists():
            gold=json.loads(labels.read_text(encoding='utf-8-sig'))
            result['fit_check']=score_boxes(gold['elements'],notes,0,image.width)['annotation']
        (args.output/f'{stem}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        row={'image':stem,'mode':result['layout_mode'],'notes':len(notes),'primary':len(primary),
             'merged_headings':len(headings),'bands':len(valid),'fit':result.get('fit_check')}
        reports.append(row);print(json.dumps(row),flush=True)
        (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':main()
