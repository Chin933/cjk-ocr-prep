"""Fine-tune contextual text roles with reviewed pages and partial corrections."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from digitalization.box_detector import BoxSegmenter


def examples():
    batch=ROOT/'training/layout_review/batches/typed_feedback_20260918'
    result=[]
    for stem in ['archive_113_p0054','archive_127_p0056','archive_155_p0396','archive_127_p0058','archive_113_p0219']:
        page=json.loads((batch/f'{stem}.json').read_text(encoding='utf-8'))
        gray=np.asarray(Image.open(ROOT/'training/images/archive_diverse'/f'{stem}.jpg').convert('L'))
        elements={e['id']:e for e in page['notes']+page['primary_candidates']}
        regions=[]
        if stem=='archive_113_p0054':
            gold=json.loads((ROOT/'training/layout_review/annotations'/f'{stem}.user-linked.json').read_text(encoding='utf-8-sig'))
            for role in ['primary','annotation']:
                regions.extend((role,b) for e in gold['elements'] if e['role']==role for b in e['boxes'])
        elif stem=='archive_127_p0056':
            regions=[(e['role'],e['boxes'][0]) for e in page['primary_candidates']+page['notes']]
        elif stem=='archive_155_p0396':
            regions=[('primary',[231,579,289,614]),('annotation',[231,614,289,671]),
                     ('primary',[287,675,345,706]),('annotation',[287,602,345,675]),
                     ('background',[231,555,289,579])]
        elif stem=='archive_127_p0058':
            boxes=[elements[k]['boxes'][0] for k in ['N0008','N0009','M0006']]
            regions=[('annotation',[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])]
            regions.extend(('primary',elements[k]['boxes'][0]) for k in ['N0024','N0026'])
        else:
            boxes=[elements[k]['boxes'][0] for k in ['N0015','M0015']]
            regions=[('annotation',[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])]
        result.append((stem,gray,regions,.75*51/np.median([b['pitch'] for b in page['bands'] if 'pitch' in b])))
    gray=np.asarray(Image.open(ROOT/'runs/instruction_review/image27.png').convert('L'))
    # The supplied figure is one complete examiner record.
    h,w=gray.shape
    result.append(('intro_image27',gray,[('primary',[0,0,w,round(h*.21)]),
                  ('annotation',[0,round(h*.21),w,round(h*.88)]),
                  ('primary',[0,round(h*.88),w,h])],.75*51/w))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/context_roles');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(27);rng=np.random.default_rng(27)
    saved=torch.load(ROOT/'runs/box_detector_v2_final/box_segmenter.pt',weights_only=True)
    model=BoxSegmenter(channels=4);model.load_state_dict(saved['state_dict'])
    teacher=BoxSegmenter(channels=4).eval();teacher.load_state_dict(saved['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00015)
    samples=[];manifest=[]
    for name,gray,regions,scale in examples():
        label=np.zeros((2,*gray.shape),np.float32);known=np.zeros(gray.shape,np.float32)
        for role,(x1,y1,x2,y2) in regions:
            label[:,y1:y2,x1:x2]=0
            if role!='background':label[['primary','annotation'].index(role),y1:y2,x1:x2]=1
            known[y1:y2,x1:x2]=1
        gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        label=np.stack([cv2.resize(m,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST) for m in label])
        known=cv2.resize(known,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
        centers=[(round((b[0]+b[2])/2*scale),round((b[1]+b[3])/2*scale)) for _,b in regions]
        pad=128
        samples.append((np.pad(gray,pad,constant_values=255),np.pad(label,((0,0),(pad,pad),(pad,pad))),np.pad(known,pad),centers))
        manifest.append({'image':name,'regions':[{'role':r,'box':b} for r,b in regions]})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'training_regions.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.monotonic()
    for step in range(args.steps):
        xs=[];ys=[];masks=[]
        for _ in range(4):
            pixels,labels,known,centers=samples[int(rng.integers(len(samples)))]
            cx,cy=centers[int(rng.integers(len(centers)))];cx+=int(rng.integers(-16,17));cy+=int(rng.integers(-16,17))
            x=max(0,min(pixels.shape[1]-192,cx+32));y=max(0,min(pixels.shape[0]-192,cy+32))
            tile=pixels[y:y+192,x:x+192].copy()
            choice=rng.random()
            if choice<.25:tile=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(4,4)).apply(tile)
            elif choice<.5:tile=np.uint8(255*(tile.astype(np.float32)/255)**rng.uniform(.75,1.5))
            elif choice<.65:
                operation=cv2.erode if rng.random()<.5 else cv2.dilate
                tile=operation(tile,np.ones((1,2),np.uint8))
            xs.append((1-tile.astype(np.float32)/255)[None]);ys.append(labels[:,y:y+192,x:x+192]);masks.append(known[y:y+192,x:x+192])
        x=torch.from_numpy(np.stack(xs));y=torch.from_numpy(np.stack(ys));mask=torch.from_numpy(np.stack(masks))[:,None]
        logits=model(x)
        with torch.no_grad():prior=teacher(x).sigmoid()
        supervised=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:2],y,reduction='none')
        preserve=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:2],prior[:,:2],reduction='none')
        loss=(supervised*mask).sum()/(2*mask.sum().clamp_min(1))+.25*(preserve*(1-mask)).mean()
        # Keep the existing edge head while the role representation adapts.
        loss+=.1*torch.nn.functional.binary_cross_entropy_with_logits(logits[:,2:],prior[:,2:])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%40==0 or step+1==args.steps:print(json.dumps({'step':step+1,'loss':round(loss.item(),4),'seconds':round(time.monotonic()-start,1)}),flush=True)
    torch.save({**saved,'state_dict':model.state_dict(),'steps':saved.get('steps',0)+args.steps,'context_role_finetuning':True},args.output/'box_segmenter.pt')


if __name__=='__main__':main()
