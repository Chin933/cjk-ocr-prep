"""Spatially held-out, single-page pilot for direct image-to-box learning.

The left part is never sampled into training crops. This is not a held-out-page
benchmark. Human relations and parent ids are not used in either training or
inference. Original labels stay unchanged, including overlapping envelopes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from digitalization.box_detector import BoxSegmenter, predict_boxes


def targets(data, width, height):
    masks = np.zeros((4, height, width), np.float32)
    for e in data['elements']:
        if e['role'] not in ('primary', 'annotation'):
            continue
        channel = ('primary', 'annotation').index(e['role'])
        for x1, y1, x2, y2 in e['boxes']:
            if x2-x1 > 6 and y2-y1 > 6:
                masks[channel, y1+3:y2-3, x1+3:x2-3] = 1
            cv2.rectangle(masks[channel+2], (x1, y1), (x2-1, y2-1), 1, 5)
    return masks


def overlap(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def score_boxes(gold, predicted, x1, x2):
    result = {}
    for role in ('primary', 'annotation'):
        truth = [e for e in gold if e['role'] == role and all(b[0] >= x1 and b[2] <= x2 for b in e['boxes'])]
        proposal = [e for e in predicted if e['role'] == role and x1 <= (e['boxes'][0][0]+e['boxes'][0][2])/2 <= x2]
        candidates = sorted([(overlap(a['boxes'][0], b['boxes'][0]), i, j) for i,a in enumerate(truth) for j,b in enumerate(proposal)], reverse=True)
        used_gold, used_prediction, matches = set(), set(), []
        for iou, i, j in candidates:
            if iou < .5 or i in used_gold or j in used_prediction:
                continue
            used_gold.add(i); used_prediction.add(j)
            matches.append({'gold': truth[i]['id'], 'predicted': proposal[j]['id'], 'iou': round(iou, 3)})
        result[role] = {'gold': len(truth), 'predicted': len(proposal), 'matched_iou50': len(matches),
                        'recall_iou50': round(len(matches)/max(1,len(truth)), 4),
                        'prediction_match_rate': round(len(matches)/max(1,len(proposal)),4),
                        'missed': [e['id'] for i,e in enumerate(truth) if i not in used_gold], 'matches': matches}
    return result


def compose_records(snippets, rng, size):
    """Recombine complete training-side records with their nested note masks."""
    canvas = np.zeros((size+180,size+100),np.float32)
    target = np.zeros((4,*canvas.shape),np.float32)
    x = 0
    while x < canvas.shape[1]:
        width = int(rng.integers(30,49))
        y = -int(rng.integers(0,60))
        while y < canvas.shape[0]:
            pixels,notes = snippets[int(rng.integers(len(snippets)))]
            height = max(15,round(pixels.shape[0]*width/pixels.shape[1]*rng.uniform(.85,1.15)))
            resized = cv2.resize(pixels,(width,height))
            label = cv2.resize(notes,(width,height),interpolation=cv2.INTER_NEAREST)
            xend,yend = min(x+width,canvas.shape[1]),min(y+height,canvas.shape[0])
            top = max(y,0)
            if yend>top:
                canvas[top:yend,x:xend] = resized[top-y:yend-y,:xend-x]
                target[1,top:yend,x:xend] = label[top-y:yend-y,:xend-x]
            cv2.rectangle(target[0],(x+2,y+2),(x+width-3,y+height-3),1,-1)
            cv2.rectangle(target[2],(x,y),(x+width-1,y+height-1),1,3)
            y += height+int(rng.integers(1,5))
        x += width+int(rng.integers(2,6))
    y = int(rng.integers(0,canvas.shape[0]-size+1))
    x = int(rng.integers(0,canvas.shape[1]-size+1))
    target[3]=cv2.morphologyEx(target[1],cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    return canvas[y:y+size,x:x+size].copy(),target[:,y:y+size,x:x+size].copy()


def weaken_rules(ink, strength=1.):
    """Fade long straight rules in the input, retaining the same box labels."""
    binary=(ink>.4).astype(np.uint8)
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((55,1),np.uint8))
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,85),np.uint8))
    rules=cv2.dilate(np.maximum(vertical,horizontal),np.ones((3,3),np.uint8))
    return ink*(1-strength*rules)


def overlay(image, elements, split, train_start, path):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for e in elements:
        if e['role'] not in ('primary', 'annotation'):
            continue
        for box in e['boxes']:
            draw.rectangle(box, outline='#8745c4' if e['role']=='primary' else '#e27a10', width=1)
    if split is not None:
        draw.line((split, 190, split, 1185), fill='#047b58', width=2)
        draw.line((train_start, 190, train_start, 1185), fill='#2476c5', width=2)
    result.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--compose-records', action='store_true')
    parser.add_argument('--weaken-rules', action='store_true')
    parser.add_argument('--full-page', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/box_detector_current')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    label_path = ROOT/'training/layout_review/annotations/archive_113_p0054.user-linked.json'
    data = json.loads(label_path.read_text(encoding='utf-8-sig'))
    image = Image.open(ROOT/'training/images/archive_diverse'/data['image']['name']).convert('RGB')
    region_boxes = [b for e in data['elements'] if e['role']=='region' for b in e['boxes']]
    left, right = min(b[0] for b in region_boxes), max(b[2] for b in region_boxes)
    top, bottom = min(b[1] for b in region_boxes), max(b[3] for b in region_boxes)
    split = round(left+(right-left)/3)
    train_start = 0 if args.full_page else split+40
    scale = .75
    gray = 1-np.asarray(image.convert('L'),dtype=np.float32)/255
    masks = targets(data,image.width,image.height)
    snippets = []
    if args.compose_records:
        for e in data['elements']:
            if e['role'] != 'primary':
                continue
            for x1,y1,x2,y2 in e['boxes']:
                if x1>=train_start and 20<x2-x1<80 and y2-y1>20:
                    snippets.append((gray[y1:y2,x1:x2],masks[1,y1:y2,x1:x2]))
    # Slice before augmentation: even interpolation cannot read held-out pixels.
    train_image = gray[:, train_start:]
    train_masks = masks[:,:,train_start:]
    train_image = cv2.resize(train_image,None,fx=scale,fy=scale)
    train_masks = np.stack([cv2.resize(m,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST) for m in train_masks])
    model = BoxSegmenter(channels=4)
    previous_steps = 0
    if args.checkpoint:
        saved = torch.load(args.checkpoint,weights_only=True)
        weights=saved['state_dict']
        if weights['out.weight'].shape[0]==3:
            expanded=model.state_dict()
            for key,value in weights.items():
                if key.startswith('out.'):
                    expanded[key][:3]=value
                    expanded[key][3]=value[2]
                else:
                    expanded[key]=value
            weights=expanded
        model.load_state_dict(weights)
        previous_steps = saved['steps']
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0007 if args.finetune else .002,weight_decay=.0001)
    size, batch = 160, 4
    start = time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    for step in range(0 if args.checkpoint and not args.finetune else args.steps):
        xs, ys = [], []
        for _ in range(batch):
            if snippets and rng.random()<.5:
                tile,target = compose_records(snippets,rng,size)
            else:
                crop = int(rng.integers(140,201))
                y = int(rng.integers(0,train_image.shape[0]-crop+1))
                x = int(rng.integers(0,train_image.shape[1]-crop+1))
                tile = cv2.resize(train_image[y:y+crop,x:x+crop],(size,size))
                target = np.stack([cv2.resize(m[y:y+crop,x:x+crop],(size,size),interpolation=cv2.INTER_NEAREST) for m in train_masks])
            if args.weaken_rules and rng.random()<.3:
                tile=weaken_rules(tile,float(rng.uniform(.3,.9)))
            tile = np.clip(tile*rng.uniform(.75,1.15)+rng.normal(0,.02,tile.shape),0,1).astype(np.float32)
            xs.append(tile[None]);ys.append(target)
        inputs, labels = torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))
        model.train()
        logits = model(inputs)
        probs = logits.sigmoid()
        dice = 1-((2*(probs*labels).sum((0,2,3))+1)/(probs.sum((0,2,3))+labels.sum((0,2,3))+1)).mean()
        loss = nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=torch.tensor([1.,1.,3.,3.])[:,None,None])+dice
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%50==0 or step==args.steps-1:
            print(json.dumps({'step':step+1,'loss':round(loss.item(),4),'seconds':round(time.monotonic()-start,1)}),flush=True)
        if (step+1)%100==0:
            torch.save({'state_dict':model.state_dict(),'scale':scale,'train_start_x':train_start,'validation_end_x':split,
                        'steps':previous_steps+step+1,'composed_record_augmentation':args.compose_records,
                        'rule_weakening_augmentation':args.weaken_rules,'full_page_training':args.full_page},args.output/'box_segmenter.pt')
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.checkpoint or args.finetune:
        torch.save({'state_dict':model.state_dict(),'scale':scale,'train_start_x':train_start,'validation_end_x':split,
                    'steps':previous_steps+args.steps,'composed_record_augmentation':args.compose_records,
                    'rule_weakening_augmentation':args.weaken_rules,'full_page_training':args.full_page},args.output/'box_segmenter.pt')
    predicted, probabilities = predict_boxes(image,model,scale)
    # Validation inference also excludes training-side context at the crop border.
    if not args.full_page:
        heldout_prediction,_ = predict_boxes(image.crop((0,0,split,image.height)),model,scale)
    report = {'experiment':'full_page_fit_separate_page_visual_review' if args.full_page else 'single_page_spatial_holdout', 'validation_x':None if args.full_page else [left,split], 'training_x':[train_start,image.width],
              'validation':None if args.full_page else score_boxes(data['elements'],heldout_prediction,left,split),
              'training_region':score_boxes(data['elements'],predicted,train_start,image.width),
              'full_page':score_boxes(data['elements'],predicted,0,image.width),
              'notes':['原始人工框不修改；重复框仍计入原始分母','预测不读取人工框、归属或跨列关系',
                       '整张标注页参与训练；本页数字仅为拟合检查，泛化由另页人工验收' if args.full_page else '左右空间留出，尚未跨页验证']}
    (args.output/'predictions.json').write_text(json.dumps(predicted,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    overlay(image,data['elements'],None if args.full_page else split,train_start,args.output/'human.png')
    overlay(image,predicted,None if args.full_page else split,train_start,args.output/'predicted.png')
    if not args.full_page:
        overlay(image.crop((0,0,split,image.height)),heldout_prediction,split,train_start,args.output/'heldout.predicted.png')
    for channel,name in enumerate(('primary','annotation','record_boundary','annotation_boundary')):
        Image.fromarray((probabilities[channel]*255).astype(np.uint8)).save(args.output/f'{name}_probability.png')
    brief={part:{role:{k:v for k,v in scores.items() if k not in ('matches','missed')} for role,scores in report[part].items()} for part in ('validation','training_region','full_page') if report[part] is not None}
    print(json.dumps(brief,indent=2),flush=True)


if __name__=='__main__':
    main()
