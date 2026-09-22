"""Local pilots for dense examiner records and touching paper marks."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from digitalization.ruled_notes import runs,close_gaps
from digitalization.page_structure import remove_repeated_rings,without_long_rules
from digitalization.ruling_geometry import lane_corners,sample_lane


def dense_block(gray):
    ink=gray<180;h,w=ink.shape
    # Local two-dimensional texture: count separate ink runs across each row.
    crossings=((~np.pad(ink,((0,0),(1,0)))[:,:-1]) & ink).sum(1)
    window=max(5,round(w*.4))
    texture=np.convolve(crossings,np.ones(window)/window,'same')
    dense=close_gaps(texture>=4.,round(w*.3))
    spans=[(a,b) for a,b in runs(dense) if b-a>w]
    if not spans:return []
    a,b=max(spans,key=lambda pair:pair[1]-pair[0])
    # Keep the densely composed record as a single region.
    return [[0,max(0,a-window//2),w,min(h,b+window//2)]]


def separate_contact(ink,pitch):
    cleaned=remove_repeated_rings(ink,pitch)
    mark_seeds=ink & ~cleaned
    # Propagate competing text and mark seeds within their connected ink.
    distance=cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,5)
    profile=cleaned.sum(0).astype(float)
    profile= np.convolve(profile,np.ones(max(3,round(pitch*.25))),'same')
    axis=int(np.argmax(profile));xx=np.arange(ink.shape[1])[None,:]
    text_seeds=cleaned & (abs(xx-axis)<pitch*.22) & (distance>1)
    markers=np.zeros(ink.shape,np.int32);markers[~ink]=1
    markers[text_seeds]=2;markers[mark_seeds]=3
    relief=np.uint8(255-distance/max(1,distance.max())*255)
    cv2.watershed(cv2.cvtColor(relief,cv2.COLOR_GRAY2BGR),markers)
    removed=(markers==3)&ink
    return removed,mark_seeds


def main():
    cv2.setNumThreads(1)
    out=ROOT/'training/layout_review/special_pilots';out.mkdir(exist_ok=True)
    reports=[]
    for number in [23,27]:
        image=Image.open(ROOT/'runs/instruction_review'/f'image{number}.png').convert('RGB')
        boxes=dense_block(np.asarray(image.convert('L')))
        marked=image.copy();draw=ImageDraw.Draw(marked)
        for b in boxes:draw.rectangle(b,outline='#e87900',width=1)
        panel=Image.new('RGB',(image.width*2,image.height),'white')
        panel.paste(image,(0,0));panel.paste(marked,(image.width,0))
        panel.resize((panel.width*3,panel.height*3)).save(out/f'exam{number}.png')
        reports.append({'figure':number,'dense_boxes':boxes})
    stem='archive_420_p0387'
    image=Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('L')
    gray=np.asarray(image)
    data=json.loads((ROOT/'training/layout_review/batches/typed_feedback_20260918'/f'{stem}.json').read_text(encoding='utf-8'))
    candidates=[]
    for bi,band in enumerate(data['bands']):
        if 'boundaries' not in band:continue
        for li,(left,right) in enumerate(zip(band['boundaries'],band['boundaries'][1:])):
            corners=lane_corners(band,li);w=round(right-left);h=round((corners[2,1]+corners[3,1]-corners[0,1]-corners[1,1])/2)
            tile,_=sample_lane(gray,np.zeros_like(gray,dtype=np.float32),corners,w,h)
            ink=without_long_rules(tile<180,w)
            removed,seeds=separate_contact(ink,w)
            count,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
            for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                pixels=labels==index
                if (removed[pixels].sum()>8 and (pixels&~removed).sum()>25 and cw>w*.35 and ch<w*2):
                    candidates.append((int(removed[pixels].sum()),tile,removed,(max(0,x-5),max(0,y-8),min(w,x+cw+5),min(h,y+ch+8)),bi,li))
    for i,(_,tile,mask,bounds,bi,li) in enumerate(sorted(candidates,key=lambda row:row[0],reverse=True)[:3],1):
        original=Image.fromarray(tile).convert('RGB');colored=np.asarray(original).copy();colored[mask]=[220,55,65]
        a=original.crop(bounds);b=Image.fromarray(colored).crop(bounds)
        panel=Image.new('RGB',(a.width*2,a.height),'white');panel.paste(a,(0,0));panel.paste(b,(a.width,0))
        panel.resize((panel.width*5,panel.height*5)).save(out/f'paper_contact{i}.png')
        reports.append({'contact':i,'band':bi,'lane':li,'local_bounds':list(map(int,bounds))})
    (out/'results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    html='<!doctype html><meta charset="utf-8"><title>密集小字与粘连墨迹试验</title><style>body{font:18px system-ui;max-width:1100px;margin:30px auto}img{max-width:95%;display:block;margin:15px 0}</style><h1>局部试验</h1><p>左右分别为原图与试验结果。橙框表示整段密集小字；红色表示候选墨迹排除区域。</p>'
    for p in sorted(out.glob('*.png')):html+=f'<h2>{p.stem}</h2><img src="{p.name}">'
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(reports))


if __name__=='__main__':main()
