"""Image-only ruled-lattice reconstruction and frozen-model input ablation."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))


def fit_lattice(ink, left, right, positions=(), pitch_hint=None):
    """Select a frame-anchored lattice that follows inter-column corridors."""
    profile = np.convolve(ink.mean(axis=0),np.ones(5)/5,mode='same')
    positions=np.asarray(positions,dtype=float)
    use_rules=len(positions)>=4 and np.ptp(positions)>(right-left)*.4
    options=[]
    for count in range(7,23):
        best=None
        for start in np.arange(left-5,left+5.1,1):
            for stop in np.arange(right-5,right+5.1,1):
                xs=np.linspace(start,stop,count+1)[1:-1]
                cost=float(np.interp(xs,np.arange(len(profile)),profile).mean())
                cost+=.001*(abs(start-left)+abs(stop-right))
                if use_rules:
                    pitch=(stop-start)/count
                    ids=np.rint((positions-start)/pitch)
                    error=np.abs(positions-start-ids*pitch)/pitch
                    coverage=len(set(ids.astype(int)))/max(1,count-1)
                    cost=.25*cost+float(error.mean())+.1*max(0,1-coverage)
                if best is None or cost<best[0]:
                    best=(cost,start,(stop-start)/count,count)
        options.append(best)
    options.sort()
    cost,origin,pitch,count=options[0]
    # A generic lattice is not evidence: require a distinctive corridor fit.
    typical=float(np.median([o[0] for o in options]))
    if pitch_hint is not None and not use_rules:
        matching=[o for o in options if abs(o[2]/pitch_hint-1)<.08]
        if matching and matching[0][0]<typical*.85:
            cost,origin,pitch,count=matching[0]
            return origin,pitch,count,1-cost/max(typical,1e-6)
    if typical<=0 or cost>typical*.8 or options[1][0]-cost<.015:
        return None
    return origin,pitch,count,1-cost/typical


def restore(image):
    from digitalization.layout import detect_rule_graph, _local_rule_segments
    gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    ink = (gray < 180).astype(np.uint8)
    # Existing frame geometry anchors the repeated internal ruled lanes.
    graph=detect_rule_graph(image)
    vertical_segments=graph.vertical+_local_rule_segments(ink,'x')
    vertical_frame=[s.position for s in vertical_segments if s.end-s.start>height*.6]
    bands = [np.array([s.start,s.position,s.end]) for s in graph.horizontal+_local_rule_segments(ink,'y')
             if s.end-s.start>width*.6 and height*.05<s.position<height*.98]
    bands.sort(key=lambda s:s[1])
    merged=[]
    for band in bands:
        if merged and band[1]-merged[-1][1]<8:
            continue
        merged.append(band)
    bands=merged
    mask = np.zeros_like(ink)
    observed = np.zeros_like(ink)
    report = []
    left_edges=[x for x in vertical_frame if width*.03<x<width*.25]
    right_edges=[x for x in vertical_frame if width*.75<x<width*.97]
    if not left_edges and 0<graph.frame.x1<width*.25:
        left_edges=[graph.frame.x1]
    if not right_edges and width*.75<graph.frame.x2<width:
        right_edges=[graph.frame.x2]
    # A scanned edge may be open or interrupted. Long horizontal rules still
    # provide independent endpoints for that side of the printed frame.
    if len(bands)>=2:
        starts=[s[0] for s in bands]
        stops=[s[2] for s in bands]
        if not left_edges and width*.02<np.median(starts)<width*.25:
            left_edges=[float(np.median(starts))]
        if not right_edges and width*.75<np.median(stops)<width*.98:
            right_edges=[float(np.median(stops))]
    if not left_edges or not right_edges:
        return image.copy(),image.copy(),[{'status':'insufficient_frame_evidence'}]
    frame_left,frame_right=max(left_edges),min(right_edges)
    candidates=[]
    for upper, lower in zip(bands,bands[1:]):
        top, bottom = int(upper[1])+5, int(lower[1])-5
        left, right = int(frame_left),int(frame_right)
        if bottom-top < height*.1 or right-left < width*.5:
            continue
        positions=[]
        for s in sorted(vertical_segments,key=lambda s:s.position):
            overlap=max(0,min(bottom,s.end)-max(top,s.start))
            if (left+10<s.position<right-10 and overlap>(bottom-top)*.3
                    and (not positions or s.position-positions[-1]>6)):
                positions.append(s.position)
        fit = fit_lattice(ink[top:bottom],left,right,positions)
        candidates.append({'band':[left,top,right,bottom],'fit':fit,'positions':positions})
    fitted=[c['fit'][1] for c in candidates if c['fit'] is not None]
    page_pitch=max(fitted) if fitted else None
    for candidate in candidates:
        left,top,right,bottom=candidate['band']
        fit=candidate['fit']
        source='local_rules' if len(candidate['positions'])>=4 else 'local_corridors'
        # Another band can resolve a half-pitch alias, but only if this band's
        # own corridors also support it and actual rulings do not contradict it.
        if page_pitch and (fit is None or abs(fit[1]*2/page_pitch-1)<.12):
            hinted=fit_lattice(ink[top:bottom],left,right,candidate['positions'],page_pitch)
            if hinted is not None:
                fit=hinted;source='cross_band_pitch_with_local_check'
        if fit is None:
            report.append({'band':[left,top,right,bottom],'status':'insufficient_lattice_evidence'})
            continue
        crop = ink[top:bottom,left:right]
        joined = cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
        vertical = cv2.morphologyEx(joined,cv2.MORPH_OPEN,np.ones((25,1),np.uint8))
        origin,pitch,count,agreement = fit
        boundaries=np.linspace(origin,origin+pitch*count,count+1)
        anchors={0:float(left),count:float(right)}
        strengths={0:float('inf'),count:float('inf')}
        for segment in vertical_segments:
            overlap=max(0,min(bottom,segment.end)-max(top,segment.start))
            index=int(round((segment.position-origin)/pitch))
            if (0<index<count and overlap>max(pitch,(bottom-top)*.18)
                    and abs(segment.position-boundaries[index])<pitch*.2
                    and overlap>strengths.get(index,0)):
                anchors[index]=float(segment.position)
                strengths[index]=overlap
        if len(anchors)>2:
            indices=sorted(anchors)
            boundaries=np.interp(np.arange(count+1),indices,[anchors[i] for i in indices])
        tracks=[]
        for index in range(1,count):
            center = boundaries[index]
            xa=xb=int(round(center))
            cv2.line(mask,(xa,top),(xb,bottom),1,1)
            tracks.append([xa,top,xb,bottom])
        observed[top:bottom,left:right] |= vertical & cv2.dilate(mask[top:bottom,left:right],np.ones((1,5),np.uint8))
        report.append({'band':[left,top,right,bottom],'pitch':float(pitch),
                       'fit_contrast':agreement,'pitch_source':source,'tracks':tracks,
                       'boundaries':[float(x) for x in boundaries]})
    restored = np.asarray(image.convert('RGB')).copy()
    restored[mask>0] = np.minimum(restored[mask>0],40)
    overlay = np.asarray(image.convert('RGB')).copy()
    overlay[cv2.dilate(observed,np.ones((1,3),np.uint8))>0] = [0,170,190]
    added = mask & (1-cv2.dilate(observed,np.ones((1,5),np.uint8)))
    overlay[cv2.dilate(added,np.ones((1,3),np.uint8))>0] = [230,70,70]
    return Image.fromarray(restored), Image.fromarray(overlay), report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/rule_restoration')
    args=parser.parse_args()
    image=Image.open(args.image).convert('RGB')
    restored,overlay,report=restore(image)
    args.output.mkdir(parents=True,exist_ok=True)
    stem=args.image.stem
    restored.save(args.output/f'{stem}.restored.png')
    overlay.save(args.output/f'{stem}.rules.png')
    (args.output/f'{stem}.rules.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.checkpoint:
        import torch
        from digitalization.box_detector import BoxSegmenter,predict_boxes
        from train_box_detector import score_boxes
        torch.set_num_threads(2)
        saved=torch.load(args.checkpoint,weights_only=True)
        model=BoxSegmenter(channels=saved['state_dict']['out.weight'].shape[0])
        model.load_state_dict(saved['state_dict'])
        results={}
        annotation=ROOT/'training/layout_review/annotations'/f'{stem}.user-linked.json'
        truth=json.loads(annotation.read_text(encoding='utf-8-sig')) if annotation.exists() else None
        for name,input_image in [('original',image),('restored',restored)]:
            boxes,_=predict_boxes(input_image,model,saved['scale'])
            (args.output/f'{stem}.{name}.predicted.json').write_text(json.dumps(boxes,indent=2),encoding='utf-8')
            plotted=input_image.copy()
            draw=ImageDraw.Draw(plotted)
            for e in boxes:
                draw.rectangle(e['boxes'][0],outline='#8745c4' if e['role']=='primary' else '#e27a10',width=1)
            plotted.save(args.output/f'{stem}.{name}.predicted.png')
            results[name]=score_boxes(truth['elements'],boxes,0,image.width) if truth else {'prediction_count':len(boxes),'human_review_required':True}
        (args.output/f'{stem}.comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'image':stem,'bands':report},indent=2))


if __name__=='__main__':
    main()
