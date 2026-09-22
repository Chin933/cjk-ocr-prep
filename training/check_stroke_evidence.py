"""Inspect stroke-width evidence in the frozen feedback lanes."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'src'))
from digitalization.ruling_geometry import lane_corners,sample_lane
from digitalization.ruled_notes import large_glyph_rows,runs,close_gaps,compound_glyph_rows
targets={'archive_127_p0056':[(1,5),(1,3),(1,2)],'archive_113_p0054':[(1,1)],'archive_029_p0237':[(1,10),(1,7)],'archive_155_p0396':[(0,4),(1,5),(1,3),(1,2)]}
for stem,lanes in targets.items():
    data=json.loads((ROOT/'training/layout_review/batches/typed_20260918'/f'{stem}.json').read_text())
    gray=np.asarray(Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('L'))
    prob=np.load(ROOT/'training/layout_review/batches/notes_100'/f'{stem}.probabilities.npz')['note']
    for bi,li in lanes:
        band=data['bands'][bi];corners=lane_corners(band,li);w=round(band['boundaries'][li+1]-band['boundaries'][li]);h=round((corners[2,1]+corners[3,1]-corners[0,1]-corners[1,1])/2)
        tile,p=sample_lane(gray,prob,corners,w,h);m=max(2,round(w*.08));ink=(tile[:,m:-m]<180).astype('uint8');p=p[:,m:-m]
        dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);k=np.ones(max(5,round(w*.3)))
        stroke=np.convolve(dt.sum(1),k,'same')/np.maximum(1,np.convolve(ink.sum(1),k,'same'))
        wide=large_glyph_rows(ink,w,p);valid=np.convolve(ink.sum(1),k,'same')>w*2
        main=stroke[wide&valid];note=stroke[(p.mean(1)>.75)&valid]
        print(stem,bi,li,'main/note',np.median(main) if len(main) else 0,np.median(note) if len(note) else 0)
        print('groups',[(round(float(corners[0,1]+a)),b-a,round(float(dt[a:b].sum()/max(1,ink[a:b].sum())),2)) for a,b in runs(close_gaps(ink.sum(1)>=2,2))])
        print('compound',runs(compound_glyph_rows(ink.astype(bool),w,p,wide)))
        if (stem,li) in [('archive_127_p0056',5),('archive_029_p0237',10)]:
            for a,b in runs(ink.sum(1)>=max(2,w*.08)):
                if not 710<corners[0,1]+a<850:continue
                xs=np.flatnonzero(ink[a:b].any(0));mid=(xs[0]+xs[-1]+1)//2
                weights=[float(dt[a:b,l:r][ink[a:b,l:r].astype(bool)].mean()) for l,r in [(xs[0],mid),(mid,xs[-1]+1)]]
                print('glyph',round(float(corners[0,1]+a)),b-a,weights,'note',float(dt[(p>.8)&ink.astype(bool)&~wide[:,None]].mean()))
        print([(round(float(corners[0,1]+y)),round(float(stroke[y]),2),round(float(p[y].mean()),2)) for y in range(0,h,15) if valid[y]][:40])
