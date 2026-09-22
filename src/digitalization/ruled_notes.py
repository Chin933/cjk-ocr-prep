"""Decode note regions within physical ruled lanes, then propose preceding text.

The note probability map is the only learned input. Primary text candidates
are local ink intervals, not record envelopes or cross-column ownership.
"""

from __future__ import annotations

import numpy as np
import cv2
from .ruling_geometry import lane_corners, sample_lane, interpolate


def runs(mask):
    changes=np.diff(np.pad(np.asarray(mask,dtype=np.int8),(1,1)))
    return list(zip(np.flatnonzero(changes==1).tolist(),np.flatnonzero(changes==-1).tolist()))


def close_gaps(mask, limit):
    result=np.asarray(mask,dtype=bool).copy()
    for a,b in runs(~result):
        if a>0 and b<len(result) and b-a<=limit:
            result[a:b]=True
    return result


def large_glyph_rows(crop, width, probability):
    """Find wide connected strokes using full-lane context, not a clipped box."""
    _,labels,stats,_=cv2.connectedComponentsWithStats(crop.astype(np.uint8))
    blocked=np.zeros(len(crop),bool)
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        if (w>=width*.62 and width*.14<=h<=width*1.15 and area>=width*width*.04
                and probability[y:y+h,x:x+w].mean()<.65):
            # Touching ink may connect a large character to the small note
            # below it. Only its genuinely wide rows define the veto extent.
            broad=[]
            for row in range(y,y+h):
                xs=np.flatnonzero(labels[row]==index)
                if len(xs) and xs[-1]-xs[0]+1>=width*.62:
                    broad.append(row)
            if broad:
                blocked[max(0,broad[0]-1):min(len(crop),broad[-1]+2)]=True
    return blocked


def complete_ink_edges(crop, a, b, width, blocked):
    """Snap a predicted interval to nearby whole ink components, not fixed padding.

    Closing joins disconnected strokes by at most two pixels. Only components
    already touched by the interval can extend it; neighbouring glyphs do not.
    Long rules and components spanning several characters cannot pull an edge.
    """
    joined=cv2.morphologyEx(crop.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(joined)
    start,stop=a,b
    for x,y,w,h,area in stats[1:]:
        if (area<3 or h>width*.95 or w>width*1.05 or
                min(b,y+h)-max(a,y)<2 or blocked[y:y+h].any()):
            continue
        if y<a and a-y<=width*.45:
            start=min(start,int(y))
        if y+h>b and y+h-b<=width*.45:
            stop=max(stop,int(y+h))
    return max(0,start-2),min(len(crop),stop+2)


def local_font_evidence(crop,width,p,wide):
    """Compare stroke weight within one lane; scale is not a page-wide constant."""
    dt=cv2.distanceTransform(crop.astype(np.uint8),cv2.DIST_L2,5)
    kernel=np.ones(max(5,round(width*.3)))
    mass=np.convolve(crop.sum(1),kernel,'same')
    stroke=np.convolve(dt.sum(1),kernel,'same')/np.maximum(1,mass)
    main=stroke[wide & (mass>width*2)]
    note=stroke[(p.mean(1)>.75) & (mass>width*2)]
    empty=np.zeros(len(crop),bool)
    if len(main)<4 or len(note)<4:return empty,empty.copy()
    main_size,note_size=float(np.median(main)),float(np.median(note))
    separated=main_size>=note_size*1.08
    cutoff=note_size+(main_size-note_size)*.50
    main_cutoff=max(note_size+(main_size-note_size)*.75,note_size*1.18) if separated else float('inf')
    blocked=empty.copy()
    for a,b in runs(close_gaps(crop.sum(1)>=2,2)):
        if width*.22<=b-a<=width*1.2:
            value=dt[a:b].sum()/max(1,crop[a:b].sum())
            if value>main_cutoff:blocked[a:b]=True
    joined=cv2.morphologyEx(crop.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(joined)
    glyphs=[]
    for x,y,w,h,area in stats[1:]:
        if (separated and width*.18<w<width*.48 and width*.24<h<width*.85 and .55<w/h<1.4
                and dt[y:y+h,x:x+w].sum()/max(1,crop[y:y+h,x:x+w].sum())<cutoff):
            glyphs.append(tuple(map(int,(x,y,w,h))))
    pairs=empty.copy()
    for i,(x,y,w,h) in enumerate(glyphs):
        for xx,yy,ww,hh in glyphs[i+1:]:
            gap=max(x,xx)-min(x+w,xx+ww)
            if (0<=gap<width*.2 and min(y+h,yy+hh)-max(y,yy)>max(h,hh)*.7
                    and max(x+w,xx+ww)-min(x,xx)>width*.55):
                pairs[min(y,yy):max(y+h,yy+hh)]=True
    # A short single-column note can occupy only half a physical lane. Require
    # the entire row group's ink to be narrow, not one radical of a wide glyph.
    supported=np.flatnonzero(p.mean(1)>.5)
    for a,b in runs(close_gaps(crop.sum(1)>=2,2)):
        xs=np.flatnonzero(crop[a:b].sum(0)>0)
        if not len(xs) or not width*.2<=b-a<=width*.72:continue
        if xs[-1]-xs[0]>width*.52:continue
        value=dt[a:b].sum()/max(1,crop[a:b].sum())
        nearby=len(supported) and np.min(np.abs(supported-(a+b)/2))<width*.7
        if value<(cutoff if separated else note_size*1.08) and (p[a:b].mean()>.15 or nearby):pairs[a:b]=True
    return blocked,pairs


def compound_glyph_rows(crop, width, probability, anchors):
    """Verify a whole left/right glyph against nearby main-text stroke weight.

    Both halves must carry main-weight strokes and form one character-height
    unit adjoining an established main stream. A central gap is not evidence.
    """
    blocked=np.zeros(len(crop),bool)
    dt=cv2.distanceTransform(crop.astype(np.uint8),cv2.DIST_L2,5)
    groups=runs(crop.sum(1)>=max(2,width*.08))
    note_pixels=(probability>.8) & crop & ~anchors[:,None]
    if note_pixels.sum()<width*2:return blocked
    note_weight=float(dt[note_pixels].mean())
    for a,b in groups:
        ys,xs=np.nonzero(crop[a:b])
        if not len(xs) or not width*.4<=b-a<=width*.95:continue
        left,right=int(xs.min()),int(xs.max())+1
        if right-left<width*.65:continue
        nearby=anchors[max(0,a-round(width)):min(len(crop),b+round(width))].copy()
        if not nearby.any():continue
        lo=max(0,a-round(width));hi=min(len(crop),b+round(width))
        reference=crop[lo:hi] & anchors[lo:hi,None]
        if reference.sum()<width:continue
        main_weight=float(dt[lo:hi][reference].mean())
        if main_weight<note_weight*1.08:continue
        threshold=max(note_weight*1.13,note_weight+.45*(main_weight-note_weight))
        middle=(left+right)//2
        halves=[(left,middle),(middle,right)]
        weights=[]
        for x1,x2 in halves:
            ink=crop[a:b,x1:x2]
            rows=np.flatnonzero(ink.any(1))
            if len(rows)<(b-a)*.65:break
            weights.append(float(dt[a:b,x1:x2][ink].mean()))
        if len(weights)==2 and min(weights)>note_weight*1.04 and np.mean(weights)>threshold:
            blocked[a:b]=True
    return blocked


def decode_notes(image, bands, probability, boundary=None, *, stroke_aware=False, primary_probability=None):
    """Generate lane-bounded notes; no primary prediction enters this decoder."""
    gray=np.asarray(image.convert('L'))
    ink=gray<180
    notes,primary=[],[]
    for band_index,band in enumerate(bands):
        if 'boundaries' not in band:
            continue
        _,top,_,bottom=band['band']
        boundaries=band['boundaries']
        for lane_index,(left,right) in enumerate(zip(boundaries,boundaries[1:])):
            x1,x2=max(0,round(left)),min(image.width,round(right))
            width=x2-x1
            if width<12:
                continue
            margin=max(2,round(width*.08))
            corners=lane_corners(band,lane_index)
            if 'slanted_tracks' in band:
                height=round(float((corners[2,1]+corners[3,1]-corners[0,1]-corners[1,1])/2))
                tile,prob=sample_lane(gray,probability.astype(np.float32),corners,width,height)
                crop=tile[:,margin:-margin]<180
                p=prob[:,margin:-margin]
                if primary_probability is not None:
                    _,main_prob=sample_lane(gray,primary_probability.astype(np.float32),corners,width,height)
                    main_prob=main_prob[:,margin:-margin]
            else:
                crop=ink[top:bottom,x1+margin:x2-margin]
                p=probability[top:bottom,x1+margin:x2-margin]
                if primary_probability is not None:
                    main_prob=primary_probability[top:bottom,x1+margin:x2-margin]
            def geometry(a,b):
                if 'slanted_tracks' not in band:
                    return {'boxes':[[x1,top+a,x2,top+b]]}
                points=interpolate(corners,np.array([0.,1.,1.,0.]),
                                   np.array([a,a,b,b])/max(1,len(crop)-1))
                points[:,0]=np.clip(points[:,0],0,image.width)
                points[:,1]=np.clip(points[:,1],0,image.height)
                box=[int(np.floor(points[:,0].min())),int(np.floor(points[:,1].min())),
                     int(np.ceil(points[:,0].max())),int(np.ceil(points[:,1].max()))]
                return {'boxes':[box],'polygon':np.round(points,1).tolist()}
            # Aggregate a 2-D note prediction only within this physical lane.
            support=p.mean(axis=1)
            # A boundary head can cover an entire short note. Use interior
            # evidence instead of subtracting its edge pixels. Keep the
            # original threshold; continuation is resolved between intervals.
            active=close_gaps(support>=.5,max(2,round(width*.12)))
            wide=large_glyph_rows(crop,width,p)
            pair_rows=np.zeros(len(crop),bool)
            if stroke_aware:
                font_main,pair_rows=local_font_evidence(crop,width,p,wide)
                compound=compound_glyph_rows(crop,width,p,wide)
                edges=np.zeros(len(crop),bool)
                for a,b in runs(active):
                    edges[a:min(b,a+round(width*.8))]=True
                    edges[max(a,b-round(width*.8)):b]=True
                wide|=font_main & edges
                for a,b in runs(compound):
                    if edges[a:b].any():wide[a:b]=True
                active|=pair_rows
            if primary_probability is not None:
                main_support=main_prob.mean(1)
                wide[(support>.8) & (main_support<support*.8)]=False
                for a,b in runs(crop.sum(1)>=max(2,width*.08)):
                    if b-a<=width*.95 and main_support[a:b].mean()>.65 and support[a:b].mean()<.35:
                        wide[a:b]=True
            active[wide]=False
            intervals=[]
            for a,b in runs(active):
                if (b-a<width*.15 or crop[a:b].sum()<width*1.2
                        or (np.count_nonzero(support[a:b]>=.5)<2 and not pair_rows[a:b].any())):
                    continue
                a,b=complete_ink_edges(crop,a,b,width,wide)
                occupied=np.flatnonzero(crop[a:b].sum(axis=1)>=2)
                if not len(occupied):
                    continue
                # Do not grow back across a detected large-character boundary.
                inner=np.flatnonzero(~wide[a:b])
                if not len(inner):
                    continue
                a,b=a+int(inner[0]),a+int(inner[-1])+1
                gap_start=intervals[-1][1] if intervals else a
                gap=slice(gap_start,a)
                ink_gap=crop[gap]
                weighted=float((p[gap]*ink_gap).sum()/max(1,ink_gap.sum()))
                bridge=(0<a-gap_start<=width*.65 and not wide[gap].any()
                        and (weighted>=.3 or ink_gap.sum()<width*.15))
                if intervals and (a<=intervals[-1][1] or bridge):
                    intervals[-1]=(intervals[-1][0],b)
                else:
                    intervals.append((a,b))
            for a,b in intervals:
                notes.append({'id':f'N{len(notes)+1:04d}','role':'annotation',
                              **geometry(a,b),'band':band_index,'lane':lane_index,
                              'score':round(float(support[a:b].mean()),4),
                              'source':'note_probability_in_ruled_lane'})
            # Outside notes, retain actual ink, not the blank complements.
            # These are local main-text candidates, not asserted record groups.
            remaining=crop.sum(axis=1)>=max(2,width*.06)
            if primary_probability is not None:
                remaining &= (main_support>=.2) | (support>=.2)
            excluded=np.zeros(len(crop),bool)
            for a,b in intervals:
                excluded[max(0,a-2):min(len(crop),b+2)]=True
            remaining=close_gaps(remaining,round(width*.45)) & ~excluded
            for a,b in runs(remaining):
                rows=np.flatnonzero(crop[a:b].sum(axis=1)>=max(2,width*.06))
                if not len(rows):
                    continue
                a,b=a+int(rows[0]),a+int(rows[-1])+1
                if b-a<width*.32 or crop[a:b].sum()<width*1.8:
                    continue
                primary.append({'id':f'M{len(primary)+1:04d}','role':'primary',
                                **geometry(max(0,a-2),min(len(crop),b+2)),
                                'band':band_index,'lane':lane_index,
                                'source':'ink_outside_notes','review':'candidate'})
    return notes,primary
