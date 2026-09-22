"""Image-based merged cells and text streams, independent of OCR semantics.

Physical rulings constrain geometry, not the main-text/annotation role. Page
types select different decoders; register count does not determine the type.
"""

import cv2
import numpy as np
from PIL import Image

from .ruled_notes import runs, close_gaps, decode_notes
from .ruling_geometry import lane_corners, sample_lane, interpolate, refine_tracks


def without_long_rules(ink, pitch):
    mask=ink.astype(np.uint8)
    length=max(25,round(pitch*2.8))
    lines=(cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((length,1),np.uint8)) |
           cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,length),np.uint8)))
    return (mask & (1-lines)).astype(bool)


def merged_headings(image, bands):
    """Detect oversize glyph streams that occupy several ordinary cells."""
    ink=np.asarray(image.convert('L'))<180
    headings=[]
    for bi,band in enumerate(bands):
        if 'pitch' not in band:
            continue
        x1,y1,x2,y2=band['band'];pitch=band['pitch']
        if bi!=0 or y2-y1>image.height*.6 or y1>image.height*.45:
            continue
        crop=without_long_rules(ink[y1:y2,x1:x2],pitch)
        joined=cv2.morphologyEx(crop.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(joined)
        seeds=[]
        for x,y,w,h,area in stats[1:]:
            if w>pitch*1.75 and pitch*.95<h<pitch*3.2 and area>pitch*pitch*.45:
                seeds.append([int(x),int(y),int(x+w),int(y+h)])
        groups=[]
        for box in sorted(seeds,key=lambda b:b[1]):
            for group in groups:
                overlap=min(box[2],group[2])-max(box[0],group[0])
                if overlap>min(box[2]-box[0],group[2]-group[0])*.65 and box[1]-group[3]<pitch:
                    group[:]=[min(group[0],box[0]),min(group[1],box[1]),max(group[2],box[2]),max(group[3],box[3])]
                    break
            else:
                groups.append(box.copy())
        for a,b,c,d in groups:
            if d-b<pitch*2 or b>(y2-y1)*.25:
                continue
            headings.append({'boxes':[[max(x1,x1+a-3),max(y1,y1+b-3),min(x2,x1+c+3),min(y2,y1+d+3)]],
                             'role':'primary','band':bi,'source':'oversize_glyph_stream',
                             'review':'candidate'})
    return headings


def mask_headings(image, probability, headings):
    pixels=np.asarray(image.convert('RGB')).copy()
    probability=probability.copy()
    for e in headings:
        x1,y1,x2,y2=e['boxes'][0]
        pixels[y1:y2,x1:x2]=255
        probability[y1:y2,x1:x2]=0
    return Image.fromarray(pixels),probability


def stream_core(ink, pitch, exclude_side_marks=True):
    """Find the text-bearing width; repeated small side marks stay outside it."""
    profile=np.convolve(ink.sum(axis=0),np.ones(3)/3,mode='same')
    if profile.max()<2:
        return None
    occupied=close_gaps(profile>max(2,profile.max()*.25),max(2,round(pitch*.07)))
    candidates=[(a,b) for a,b in runs(occupied) if b-a>=pitch*.18]
    if not candidates:
        return None
    if not exclude_side_marks:
        return max(0,candidates[0][0]-2),min(ink.shape[1],candidates[-1][1]+2)
    a,b=max(candidates,key=lambda pair:profile[pair[0]:pair[1]].sum())
    _,_,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    start,stop=a,b
    for x,y,w,h,area in stats[1:]:
        if (area>=8 and w<pitch*.85 and h<pitch*1.3 and
                min(b,x+w)-max(a,x)>=max(2,w*.25)):
            start=min(start,int(x));stop=max(stop,int(x+w))
    a=max(a-round(pitch*.18),start);b=min(b+round(pitch*.18),stop)
    return max(0,a-2),min(ink.shape[1],b+2)


def parallel_ranges(ink, main_width):
    """Find sustained narrow subcolumns; glyph height rejects split radicals.

    Two radical halves at the same full-size vertical cadence are still one
    main glyph. A subcolumn must contain compact glyphs, not just a gutter.
    """
    h,w=ink.shape
    occupied=close_gaps(ink.sum(axis=1)>=2,max(2,round(main_width*.22)))
    ranges=[]
    for a,b in runs(occupied):
        # Short split radicals are not enough evidence for parallel text. Do
        # not start windows in the middle of a full-size character.
        if b-a<main_width*1.3:
            continue
        for start,stop in [(a,b)]:
            tile=ink[start:stop]
            if stop-start<main_width*.15:
                continue
            projection=tile.sum(axis=0)
            columns=runs(close_gaps(projection>=max(2,(stop-start)*.08),1))
            columns=[(x,y) for x,y in columns if main_width*.15<=y-x<=main_width*.63]
            if len(columns)<2:
                continue
            heights=[]
            for x,y in columns:
                rr=runs(close_gaps(tile[:,x:y].sum(axis=1)>0,1))
                heights.extend(v-u for u,v in rr if v-u>=3)
            if len(heights)<6 or np.median(heights)>main_width*.62:
                continue
            # Both streams must actually be present at similar vertical spans.
            spans=[]
            for x,y in columns:
                ys=np.flatnonzero(tile[:,x:y].sum(axis=1)>0)
                spans.append((int(ys[0]),int(ys[-1])+1))
            overlap=min(v for u,v in spans)-max(u for u,v in spans)
            if overlap<min(v-u for u,v in spans)*.55:
                continue
            ranges.append((start,stop))
    result=np.zeros(h,bool)
    for a,b in ranges:result[a:b]=True
    result=close_gaps(result,round(main_width*.8))
    return runs(result)


def short_parallel_ranges(ink, main_width):
    """Paired compact glyphs in examiner titles, e.g. small side-by-side names."""
    joined=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(joined)
    row_runs=runs(close_gaps(ink.sum(axis=1)>=2,max(2,round(main_width*.1))))
    heights=[b-a for a,b in row_runs if main_width*.3<=b-a<=main_width*1.3]
    if len(heights)<3:return []
    main_height=float(np.percentile(heights,75))
    glyphs=[tuple(map(int,s)) for s in stats[1:]
            if main_width*.25<=s[3]<=main_height*.7 and .64<=s[2]/max(1,s[3])<=1.5
            and s[4]>=s[2]*s[3]*.18]
    active=np.zeros(len(ink),bool)
    for i,(x,y,w,h,_) in enumerate(glyphs):
        for xx,yy,ww,hh,_ in glyphs[i+1:]:
            gap=max(x,xx)-min(x+w,xx+ww)
            overlap=min(y+h,yy+hh)-max(y,yy)
            span=max(x+w,xx+ww)-min(x,xx)
            if 0<=gap<=main_width*.25 and overlap>max(h,hh)*.6 and span>main_width*.72:
                active[min(y,yy):max(y+h,yy+hh)]=True
    return runs(active)


def remove_repeated_rings(ink,pitch):
    """Remove only repeated hollow side/leading marks from working evidence."""
    contours,hierarchy=cv2.findContours(ink.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return ink
    candidates=[]
    for i,c in enumerate(contours):
        child=hierarchy[0,i,2]
        if hierarchy[0,i,3]>=0 or child<0:continue
        x,y,w,h=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if (pitch*.08<=w<=pitch*.45 and pitch*.08<=h<=pitch*.45 and .65<w/h<1.5
                and area>8 and cv2.contourArea(contours[child])/area>.24):
            candidates.append((i,x+w/2,y,w,h))
    result=ink.copy()
    # Hole contours remain available when a ring touches a neighbouring glyph.
    # Vote for repeated circular holes on one track, then remove a narrow
    # annulus only. Never erase the whole connected glyph/mark component.
    holes=[]
    for i,c in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c);perimeter=cv2.arcLength(c,True)
        if (pitch*.065<=w<=pitch*.30 and pitch*.065<=h<=pitch*.30 and .7<w/h<1.4
                and perimeter>0 and 4*np.pi*area/perimeter**2>.68
                and len(cv2.approxPolyDP(c,.025*perimeter,True))>=6):
            holes.append((i,x+w/2,y+h/2,w,h))
    for i,x,y,w,h in holes:
        peers=[c for c in holes if abs(c[1]-x)<pitch*.07]
        if len(peers)<4 or np.ptp([c[2] for c in peers])<len(ink)*.2:continue
        # Side marks occupy the outer portion of the printed lane. Leading
        # circles can share the title axis but precede the text at page top.
        leading=y<len(ink)*.13 and sum(c[2]<len(ink)*.13 for c in peers)>=3
        if x<ink.shape[1]*.58 and not leading:continue
        mask=np.zeros(ink.shape,np.uint8)
        cv2.ellipse(mask,(round(x),round(y)),(round(w/2+2),round(h/2+2)),0,0,360,1,-1)
        result[mask>0]=False
    profile=np.convolve(ink.sum(axis=0),np.ones(5)/5,mode='same')
    center=int(np.argmax(profile))
    for i,x,y,w,h in candidates:
        peers=[c for c in candidates if abs(c[1]-x)<pitch*.1]
        if len(peers)<3:continue
        near=sorted(peers,key=lambda c:c[2])
        leading=(y<ink.shape[0]*.15 and sum(c[2]<ink.shape[0]*.15 for c in near)>=3)
        if abs(x-center)<pitch*.19 and not leading:continue
        mask=np.zeros(ink.shape,np.uint8)
        cv2.drawContours(mask,contours,i,1,cv2.FILLED)
        result[mask>0]=False
    return result


def decode_streams(image, bands, *, exclude_side_marks=True, short_parallel=False, blank_split=None):
    """Shared rectification and ink geometry, configured by a page-type decoder."""
    gray=np.asarray(image.convert('L'))
    primary,notes=[],[]
    for bi,band in enumerate(bands):
        if 'boundaries' not in band:continue
        for li,(left,right) in enumerate(zip(band['boundaries'],band['boundaries'][1:])):
            width=round(right-left)
            corners=lane_corners(band,li)
            height=round((corners[2,1]+corners[3,1]-corners[0,1]-corners[1,1])/2)
            if width<12 or height<12:continue
            # The old frame inset can already cut the first glyph. Use the
            # actual inner edge, and remove ruling pixels from the evidence.
            corners[[0,1],1]-=2
            corners[[2,3],1]+=2
            height+=4
            tile,_=sample_lane(gray,np.zeros_like(gray,dtype=np.float32),corners,width,height)
            ink=without_long_rules(tile<180,width)
            ink[:3]=False;ink[-3:]=False
            margin=max(2,round(width*.035));ink[:,:margin]=False;ink[:,-margin:]=False
            marks=np.zeros_like(ink)
            if exclude_side_marks:
                cleaned=remove_repeated_rings(ink,width)
                marks=ink & ~cleaned
                ink=cleaned
            _,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
            genuine=[s for s in stats[1:] if s[2]>=width*.18 and width*.12<=s[3]<=width*1.4
                     and s[4]>width*width*.02]
            if not genuine:continue
            for index,(x,y,w,h,area) in enumerate(stats[1:],1):
                if w<width*.12 and h>width*1.5:ink[labels==index]=False
            core=stream_core(ink,width,exclude_side_marks)
            if core is None:continue
            a,b=core
            text=ink[:,a:b]
            rows=np.flatnonzero(text.sum(axis=1)>=2)
            if len(rows)<3 or text.sum()<width*1.5:continue
            # Full-size glyph width is local, rather than the printed pitch.
            # On marked papers, a row of circles beside one text stream is not
            # a second text column. Require compact multi-glyph cadence only
            # in the examiner path until touching ink is disentangled.
            small=parallel_ranges(text,b-a) if not exclude_side_marks else []
            if short_parallel:
                small+=short_parallel_ranges(text,b-a)
                small_mask=np.zeros(height,bool)
                for u,v in small:small_mask[u:v]=True
                small=runs(small_mask)
            def element(top,bottom,role):
                points=interpolate(corners,np.array([a,b,b,a])/max(1,width-1),
                                   np.array([top,top,bottom,bottom])/max(1,height-1))
                points[:,0]=np.clip(points[:,0],0,image.width)
                points[:,1]=np.clip(points[:,1],0,image.height)
                box=[int(np.floor(points[:,0].min())),int(np.floor(points[:,1].min())),
                     int(np.ceil(points[:,0].max())),int(np.ceil(points[:,1].max()))]
                record={'boxes':[box],'polygon':points.round(1).tolist(),'role':role,'band':bi,'lane':li,
                        'source':'parallel_subcolumns' if role=='annotation' else 'single_text_stream','review':'candidate'}
                contours,_=cv2.findContours(marks[top:bottom].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                exclusions=[]
                for contour in contours:
                    if cv2.contourArea(contour)<3:continue
                    xy=contour[:,0].astype(float);xy[:,1]+=top
                    if len(xy)<3:continue
                    mapped=interpolate(corners,xy[:,0]/max(1,width-1),xy[:,1]/max(1,height-1))
                    exclusions.append(mapped.round(1).tolist())
                if exclusions:
                    record['excluded_marks']=exclusions
                    record['geometry_semantics']='text_envelope_minus_excluded_marks'
                return record
            remaining=np.ones(height,bool)
            for top,bottom in small:
                remaining[top:bottom]=False
                notes.append(element(max(0,top-2),min(height,bottom+2),'annotation'))
            if blank_split is not None:
                # Poems separate the heading/author and distant verse groups;
                # examiner titles deliberately retain their large spacing.
                for u,v in runs(text.sum(axis=1)<2):
                    if v-u>width*blank_split:remaining[u:v]=False
            for top,bottom in runs(remaining):
                ys=np.flatnonzero(text[top:bottom].sum(axis=1)>=2)
                if not len(ys):continue
                top,bottom=top+int(ys[0]),top+int(ys[-1])+1
                if text[top:bottom].sum()<width:continue
                primary.append(element(max(0,top-2),min(height,bottom+2),'primary'))
    return notes,primary


def decode_exam(image,bands):
    return decode_streams(image,bands,exclude_side_marks=False,short_parallel=True)


def exam_rule_bands(image,bands):
    """Use repeated observed exam rulings before fitting sparse text corridors.

    A mostly empty exam register cannot determine column pitch from text. A
    well-supported ruling sequence in its other register can anchor both.
    """
    from .layout import _local_rule_segments
    ink=(np.asarray(image.convert('L'))<180).astype(np.uint8)
    valid=[b for b in bands if 'band' in b]
    if not valid:return bands
    left=min(b['band'][0] for b in valid);right=max(b['band'][2] for b in valid)
    positions=[]
    for s in sorted(_local_rule_segments(ink,'x'),key=lambda s:s.position):
        if left-4<=s.position<=right+4 and s.end-s.start>image.height*.12:
            if not positions or s.position-positions[-1]>8:positions.append(float(s.position))
    if len(positions)<6:return bands
    pitch=float(np.median(np.diff(positions)))
    if not 18<pitch<image.width*.15:return bands
    if positions[1]-positions[0]<pitch*.5:positions=positions[1:]
    if positions[-1]-positions[-2]<pitch*.5:positions=positions[:-1]
    gaps=np.diff(positions)/pitch
    if np.mean(np.abs(gaps-np.rint(gaps))<.16)<.85:return bands
    indices=np.rint((np.asarray(positions)-positions[0])/pitch).astype(int)
    if len(set(indices))<6:return bands
    pitch,origin=np.polyfit(indices,positions,1)
    boundaries=np.interp(np.arange(indices[-1]+1),indices,positions).tolist()
    output=[]
    for b in bands:
        if 'band' not in b:output.append(b);continue
        _,top,_,bottom=b['band']
        fresh={k:v for k,v in b.items() if k not in ('slanted_tracks','horizontal_edges','status')}
        fresh.update(band=[round(boundaries[0]),top,round(boundaries[-1]),bottom],
                     boundaries=boundaries,pitch=float(pitch),pitch_source='exam_observed_ruling_sequence')
        output.append(fresh)
    return refine_tracks(image,output)


def decode_essay(image,bands):
    return decode_streams(image,bands,exclude_side_marks=True,blank_split=2.5)


def decode_poem(image,bands):
    return decode_streams(image,bands,exclude_side_marks=True,blank_split=1.8)


PAGE_KINDS=('personal_biography','associate_biography','exam','essay','poem')


def decode_page(image,bands,probability,kind,primary_probability=None):
    """Explicit page-type dispatch. Unknown types are not silently guessed.

    Type labels are supplied by a classifier/reviewer upstream. In particular,
    a two-register exam page must not fall into the biography decoder.
    """
    headings=[]
    if kind=='personal_biography':
        headings=merged_headings(image,bands)
        masked,p=mask_headings(image,probability,headings)
        notes,primary=decode_notes(masked,bands,p,stroke_aware=True,primary_probability=primary_probability)
        primary+=headings
    elif kind=='associate_biography':
        notes,primary=decode_notes(image,bands,probability,stroke_aware=True,primary_probability=primary_probability)
    elif kind=='exam':
        notes,primary=decode_exam(image,bands)
    elif kind=='essay':
        notes,primary=decode_essay(image,bands)
        notes+=marginal_blocks(image,bands)
    elif kind=='poem':
        notes,primary=decode_poem(image,bands)
        notes+=marginal_blocks(image,bands)
    else:
        raise ValueError(f'Unresolved page type: {kind!r}; expected one of {PAGE_KINDS}')
    for e in notes+primary:e['page_kind']=kind
    return notes,primary,headings


def marginal_blocks(image,bands):
    """Group handwriting outside the frame, excluding the printed running head."""
    valid=[b for b in bands if 'band' in b and 'pitch' in b]
    if not valid:return []
    gray=np.asarray(image.convert('L'));height,width=gray.shape
    left=min(b['band'][0] for b in valid);right=max(b['band'][2] for b in valid)
    top=min(b['band'][1] for b in valid);bottom=max(b['band'][3] for b in valid)
    pitch=float(np.median([b['pitch'] for b in valid]))
    blocks=[]
    # Running heads adjoin the top right/left frame. This first pass retains
    # mid-page side comments; header-area comments remain unresolved.
    for x1,x2 in [(max(0,round(left-pitch*.9)),left-4),(right+4,min(width,round(right+pitch*.9)))]:
        y1=round(top+pitch*3);y2=bottom
        if x2-x1<8 or y2<=y1:continue
        ink=without_long_rules(gray[y1:y2,x1:x2]<180,pitch)
        rows=close_gaps(ink.sum(axis=1)>=3,round(pitch*.17))
        for a,b in runs(rows):
            if b-a<pitch*1.15 or ink[a:b].sum()<pitch*1.2:continue
            xs=np.flatnonzero(ink[a:b].sum(axis=0)>=2)
            if not len(xs) or xs[-1]-xs[0]<pitch*.15:continue
            # A fish-tail marker above an aligned running-head strip identifies
            # the printed book-fold column, rather than a marginal comment.
            strip=(gray[max(0,top):y1+a,x1:x2]<180).astype(np.uint8)
            _,_,marker_stats,_=cv2.connectedComponentsWithStats(strip)
            if any(pitch*.28<s[2]<pitch*.9 and pitch*.22<s[3]<pitch*.9
                   and s[4]>s[2]*s[3]*.45 for s in marker_stats[1:]):continue
            joined=cv2.morphologyEx(ink[a:b].astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            _,_,stats,_=cv2.connectedComponentsWithStats(joined)
            glyphs=[s for s in stats[1:] if pitch*.12<s[2]<pitch and pitch*.12<s[3]<pitch and s[4]>20]
            if len(glyphs)<2:continue
            blocks.append({'boxes':[[x1+int(xs[0])-2,y1+a-2,x1+int(xs[-1])+3,y1+b+2]],
                           'role':'annotation','source':'outside_frame_comment','review':'candidate'})
    return blocks
