"""Local slanted ruling tracks and lane rectification for note decoding."""

import cv2
import numpy as np


def weighted_median(values, weights):
    order=np.argsort(values)
    return float(np.asarray(values)[order][np.searchsorted(np.cumsum(np.asarray(weights)[order]),sum(weights)/2)])


def refine_tracks(image,bands):
    if not any('pitch' in b for b in bands):
        return bands
    ink=(np.asarray(image.convert('L'))<180).astype(np.uint8)
    pitch=float(np.median([b['pitch'] for b in bands if 'pitch' in b]))
    lines=cv2.HoughLinesP(ink,1,np.pi/1800,threshold=35,
                         minLineLength=round(pitch*1.8),maxLineGap=5)
    segments=[] if lines is None else lines[:,0].tolist()
    output=[]
    for band in bands:
        if 'boundaries' not in band:
            output.append(band);continue
        left,top,right,bottom=band['band'];middle=(top+bottom)/2
        priors=list(band['boundaries'])
        priors[0],priors[-1]=float(left),float(right)
        vertical=[]
        for x1,y1,x2,y2 in segments:
            if abs(y2-y1)<pitch*1.8 or abs(x2-x1)>abs(y2-y1)*.065:
                continue
            if min(max(y1,y2),bottom)-max(min(y1,y2),top)<pitch:
                continue
            slope=(x2-x1)/(y2-y1);center=x1+slope*(middle-y1)
            if min(abs(center-x) for x in priors)<pitch*.12:
                vertical.append((center,slope,abs(y2-y1)))
        long=[v for v in vertical if v[2]>max(pitch*3,(bottom-top)*.5)]
        common=weighted_median([v[1] for v in long],[v[2] for v in long]) if long else 0.
        tracks=[]
        for x in priors:
            local=[v for v in vertical if abs(v[0]-x)<pitch*.12]
            if local:
                # The longest observed fragment anchors position; use the
                # common slope when that fragment is too short for direction.
                center,slope,length=max(local,key=lambda v:v[2])
                if length<pitch*3:slope=common
            else:
                center,slope=x,common
            tracks.append([center+slope*(top-middle),center+slope*(bottom-middle)])
        edges=[]
        for target,inset in [(top-5,3),(bottom+5,-3)]:
            candidates=[]
            for x1,y1,x2,y2 in segments:
                if abs(x2-x1)<(right-left)*.6 or abs(y2-y1)>abs(x2-x1)*.065:
                    continue
                slope=(y2-y1)/(x2-x1);center=y1+slope*((left+right)/2-x1)
                if abs(center-target)<pitch*.25:
                    candidates.append((abs(center-target),center,slope))
            if candidates:
                _,center,slope=min(candidates)
                edges.append([center+slope*(left-(left+right)/2)+inset,
                              center+slope*(right-(left+right)/2)+inset])
            else:
                edges.append([target+inset,target+inset])
        output.append({**band,'slanted_tracks':tracks,'horizontal_edges':edges})
    return output


def lane_corners(band,index):
    left,top,right,bottom=band['band']
    if 'slanted_tracks' not in band:
        a,b=band['boundaries'][index:index+2]
        return np.array([[a,top],[b,top],[b,bottom],[a,bottom]],np.float32)
    a,b=band['slanted_tracks'][index:index+2]
    upper,lower=band['horizontal_edges']
    def edge_y(x,edge):
        return edge[0]+(edge[1]-edge[0])*(x-left)/(right-left)
    return np.array([[a[0],edge_y(a[0],upper)],[b[0],edge_y(b[0],upper)],
                     [b[1],edge_y(b[1],lower)],[a[1],edge_y(a[1],lower)]],np.float32)


def interpolate(corners,u,v):
    return ((1-u)*(1-v))[...,None]*corners[0]+(u*(1-v))[...,None]*corners[1]+(u*v)[...,None]*corners[2]+((1-u)*v)[...,None]*corners[3]


def sample_lane(gray,probability,corners,width,height):
    v,u=np.mgrid[:height,:width].astype(np.float32)
    points=interpolate(corners,u/max(1,width-1),v/max(1,height-1))
    remap=lambda a,border:cv2.remap(a,points[:,:,0],points[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    return remap(gray,255),remap(probability,0)
