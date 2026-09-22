"""Optional torch model for overlapping record and annotation box interiors."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def text_regions(image):
    """Locate substantial text masses; printed frame lines are removed first."""
    gray = np.asarray(image.convert('L'))
    ink = (gray < 180).astype(np.uint8)
    h, w = ink.shape
    lines = np.maximum(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(60, w//8)), np.uint8)),
                       cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(60, h//12), 1), np.uint8)))
    text = ink & (1-cv2.dilate(lines, np.ones((3,3),np.uint8)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(text)
    clean = np.zeros_like(text)
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA] >= 5:
            clean[labels == index] = 1
    # Join short inter-character spaces, not the larger gap to running furniture.
    joined = cv2.dilate(clean, np.ones((17,23),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    regions=[]
    for index in range(1,count):
        x,y,width,height,area = stats[index]
        if width<w*.22 or height<h*.08 or area<w*h*.008:
            continue
        ys,xs = np.nonzero((labels == index) & (clean > 0))
        if len(xs):
            regions.append([max(0,int(xs.min())-6),max(0,int(ys.min())-6),
                            min(w,int(xs.max())+7),min(h,int(ys.max())+7)])
    return regions


def regularize_boxes(boxes, image, regions):
    """Clip to text masses and align repeated records within each local lane."""
    ink = np.asarray(image.convert('L')) < 180
    retained=[]
    for element in boxes:
        b=element['boxes'][0]
        options=[]
        for index,r in enumerate(regions):
            clipped=[max(b[0],r[0]),max(b[1],r[1]),min(b[2],r[2]),min(b[3],r[3])]
            area=max(0,clipped[2]-clipped[0])*max(0,clipped[3]-clipped[1])
            options.append((area,index,clipped))
        if not options:
            continue
        area,index,clipped=max(options)
        if area<.6*(b[2]-b[0])*(b[3]-b[1]) or area<100:
            continue
        if ink[clipped[1]:clipped[3],clipped[0]:clipped[2]].sum()<max(12,area*.035):
            continue
        retained.append({**element,'boxes':[clipped],'region':index})
    for index in range(len(regions)):
        for role in ('primary','annotation'):
            group=[e for e in retained if e['region']==index and e['role']==role]
            if len(group)<3:
                continue
            width=float(np.median([e['boxes'][0][2]-e['boxes'][0][0] for e in group]))
            lanes=[]
            for e in sorted(group,key=lambda e:sum(e['boxes'][0][::2])/2):
                cx=sum(e['boxes'][0][::2])/2
                if not lanes or abs(cx-np.median([sum(p['boxes'][0][::2])/2 for p in lanes[-1]]))>width*.4:
                    lanes.append([])
                lanes[-1].append(e)
            for lane in lanes:
                if len(lane)<3:
                    continue
                left=int(round(float(np.median([e['boxes'][0][0] for e in lane]))))
                right=int(round(float(np.median([e['boxes'][0][2] for e in lane]))))
                for e in lane:
                    b=e['boxes'][0]
                    if max(abs(b[0]-left),abs(b[2]-right))<=width*.3:
                        e['boxes']=[[left,b[1],right,b[3]]]
    return retained


class BoxSegmenter(nn.Module):
    """Small fully convolutional model with independent overlapping outputs."""

    def __init__(self, channels=3):
        super().__init__()

        def block(source, target):
            return nn.Sequential(nn.Conv2d(source, target, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(target, target, 3, padding=1), nn.ReLU())

        self.enc1 = block(1, 8)
        self.enc2 = block(8, 16)
        self.enc3 = block(16, 32)
        self.bottom = block(32, 48)
        self.dec3 = block(80, 32)
        self.dec2 = block(48, 16)
        self.dec1 = block(24, 8)
        self.out = nn.Conv2d(8, channels, 1)

    def forward(self, image):
        a = self.enc1(image)
        b = self.enc2(nn.functional.max_pool2d(a, 2))
        c = self.enc3(nn.functional.max_pool2d(b, 2))
        d = self.bottom(nn.functional.max_pool2d(c, 2))

        def up(x, skip):
            return torch.cat((nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1)

        return self.out(self.dec1(up(self.dec2(up(self.dec3(up(d, c)), b)), a)))


def separate_instances(support, threshold=.5):
    """Split weakly connected foreground using two-dimensional interior seeds.

    A connected semantic mask is not necessarily one rectangular instance.
    Thin bridges must not make its bounding rectangle engulf adjacent records.
    """
    mask=(support>=threshold).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    # Calibrate interiors within each connected component. A fixed erosion
    # leaves broad accidental bridges intact on larger type sizes.
    count,components,component_stats,_=cv2.connectedComponentsWithStats(mask)
    cores=np.zeros_like(mask)
    for index in range(1,count):
        pixels=components==index
        depth=max(4.,float(np.percentile(distance[pixels],90))*.65)
        cores[pixels & (distance>=depth) & (support>=.65)]=1
    count,seeds,stats,_=cv2.connectedComponentsWithStats(cores)
    markers=np.zeros(mask.shape,np.int32)
    markers[mask==0]=1
    next_id=2
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]>=25:
            markers[seeds==index]=next_id
            next_id+=1
    # Retain isolated small objects without a broad enough interior seed.
    stats=component_stats
    count=len(stats)
    for index in range(1,count):
        pixels=components==index
        if stats[index,cv2.CC_STAT_AREA]>=40 and not np.any(markers[pixels]>1):
            y,x=np.unravel_index(np.argmax(np.where(pixels,distance,0)),mask.shape)
            markers[y,x]=next_id
            next_id+=1
    surface=(255*(1-support)).clip(0,255).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(surface,cv2.COLOR_GRAY2BGR),markers)
    markers[mask==0]=1
    instances=[]
    for index in range(2,next_id):
        ys,xs=np.nonzero(markers==index)
        if len(xs)<40:
            continue
        x,y=int(xs.min()),int(ys.min())
        width,height=int(xs.max())-x+1,int(ys.max())-y+1
        if width<7 or height<7:
            continue
        instances.append((x,y,width,height,len(xs)))
    return instances


@torch.inference_mode()
def predict_boxes(image, model, scale=.75, threshold=.5, constrain=True, instance_split=True):
    """Predict from image pixels only. No manual coordinates enter this path."""
    gray = np.asarray(image.convert('L'))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy((1 - small.astype(np.float32) / 255)[None, None])
    model.eval()
    probabilities = model(input_tensor).sigmoid()[0].numpy()
    boxes = []
    for channel, role in enumerate(('primary', 'annotation')):
        support = probabilities[channel]
        if role == 'primary':
            support = support * (1 - probabilities[2])
        elif len(probabilities)>3:
            support = support * (1 - probabilities[3])
        if instance_split:
            stats=separate_instances(support,threshold)
        else:
            mask = (support >= threshold).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            _, _, raw_stats, _ = cv2.connectedComponentsWithStats(mask)
            stats=raw_stats[1:]
        for x, y, width, height, area in stats:
            if area < 40 or width < 7 or height < 7:
                continue
            box = [max(0, round(x / scale - 3)), max(0, round(y / scale - 3)),
                   min(image.width, round((x + width) / scale + 3)),
                   min(image.height, round((y + height) / scale + 3))]
            boxes.append({'id': f'box_{len(boxes):04d}', 'role': role, 'boxes': [box],
                          'score': round(float(probabilities[channel, y:y+height, x:x+width].mean()), 4)})
    if constrain:
        boxes=regularize_boxes(boxes,image,text_regions(image))
    return boxes, probabilities
