"""Evaluate geometry-only relationships against separately held human labels."""

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from digitalization.record_flow import infer_record_flow


def evaluate(data, result):
    gold_parent = {e['id']: e['parent_id'] for e in data['elements'] if e['role'] == 'annotation' and e.get('parent_id')}
    rows = []
    for prediction in result['ownership']:
        if prediction['source'] not in gold_parent:
            continue
        top = prediction['candidates'][0]['parent_id'] if prediction['candidates'] else None
        rows.append({'source': prediction['source'], 'expected': gold_parent[prediction['source']],
                     'predicted': prediction['parent_id'], 'top_candidate': top,
                     'status': prediction['status'], 'correct': prediction['parent_id'] == gold_parent[prediction['source']]})
    roles = {e['id']: e['role'] for e in data['elements']}
    known = {(r['source'], r['target']) for r in data['relations']
             if r['type'] == 'continues' and roles[r['source']] == roles[r['target']] == 'annotation'}
    predicted = {(r['source'], r['target']) for r in result['annotation_continuations']}
    accepted = [r for r in rows if r['status'] == 'inferred']
    return {'input_elements': len(data['elements']),
            'ownership': {'labeled': len(rows), 'accepted': len(accepted),
                          'accepted_correct': sum(r['correct'] for r in accepted),
                          'top_candidate_correct': sum(r['top_candidate'] == r['expected'] for r in rows),
                          'unresolved_or_wrong': [r for r in rows if not r['correct']]},
            'annotation_continuations': {'known_positive_count': len(known), 'recovered': len(known & predicted),
                                         'missed': sorted(known - predicted), 'additional_unreviewed': sorted(predicted - known)},
            'limitations': ['人工框输入，未评价原图检测', '未标关系不是负例', '未评价整页逐字阅读顺序', '单页开发集，不代表其他页面准确率']}


def make_report(data, result, metrics, output):
    cards = []
    for row in metrics['ownership']['unresolved_or_wrong']:
        cards.append({'title': '归属待核对 · ' + row['source'][-4:], 'source': row['source'],
                      'gold': row['expected'], 'prediction': row['predicted'] or row['top_candidate'], 'kind': 'owner'})
    for edge in result['annotation_continuations']:
        cards.append({'title': '附注接续 · ' + edge['source'][-4:] + ' → ' + edge['target'][-4:],
                      'source': edge['source'], 'gold': edge['target'], 'prediction': edge['target'], 'kind': 'wrap'})
    for edge in result['primary_continuation_candidates']:
        cards.append({'title': '主文接续候选 · ' + edge['source'][-4:] + ' → ' + edge['target'][-4:],
                      'source': edge['source'], 'gold': None, 'prediction': edge['target'], 'kind': 'candidate'})
    payload = json.dumps({'elements': data['elements'], 'cards': cards}, ensure_ascii=False).replace('<', '\\u003c')
    owner, wraps = metrics['ownership'], metrics['annotation_continuations']
    output.write_text('''<!doctype html><meta charset="utf-8"><title>记录关系回归</title>
<style>body{margin:0;background:#eeeae2;color:#25392d;font:15px system-ui}header{padding:18px 25px;background:#204b3b;color:white}h1{font-size:21px;margin:0 0 8px}.layout{display:grid;grid-template-columns:minmax(700px,1fr) 330px}main{overflow:auto;height:85vh}.page{position:relative;width:872px;margin:18px auto}img{width:872px;display:block}svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}aside{padding:16px;max-height:81vh;overflow:auto}button{display:block;text-align:left;width:100%;padding:10px;margin:8px 0;background:white;border:1px solid #c5d0c1;border-radius:5px;cursor:pointer}p{line-height:1.6}.legend{font-size:13px}</style>
<header><h1>人工框上的关系推断</h1>''' + html.escape(f"归属：明确输出 {owner['accepted_correct']}/{owner['accepted']} 正确，人工已标 {owner['labeled']} 条；附注接续：找回 {wraps['recovered']}/{wraps['known_positive_count']} 条。") + '''</header>
<div class="layout"><main><div class="page"><img src="../../training/images/archive_diverse/archive_113_p0054.jpg"><svg viewBox="0 0 872 1280" id="drawing"></svg></div></main><aside><p>选择一条关系，单独看它的框。推断仅使用框的几何和类型，人工关系只用于对照。</p><p class="legend">紫色：起点<br>绿色：人工归属 / 已标接续终点<br>橙色虚线：算法候选<br>细灰框：上下区边界</p><p>主文接续候选未当成自动确认结果。这里的成绩不包含原图检测，也不代表整页阅读顺序准确率。</p><div id="cards"></div></aside></div><script>
const data=''' + payload + ''';
const svg=document.querySelector('#drawing'),ns='http://www.w3.org/2000/svg';
function box(id,color,dash){const e=data.elements.find(e=>e.id===id);if(!e)return;for(const b of e.boxes){const r=document.createElementNS(ns,'rect');for(const[k,v]of Object.entries({x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash?'7 4':'none'}))r.setAttribute(k,v);svg.append(r);}}
function show(c){svg.replaceChildren();for(const e of data.elements.filter(e=>e.role==='region'))box(e.id,'#bbb');box(c.source,'#7947bc');box(c.gold,'#16805f');if(c.prediction!==c.gold)box(c.prediction,'#da7716',true);const source=data.elements.find(e=>e.id===c.source);document.querySelector('main').scrollTop=Math.max(0,source.boxes[0][1]-180);}
for(const card of data.cards){const b=document.createElement('button');b.textContent=card.title;b.onclick=()=>show(card);document.querySelector('#cards').append(b);}if(data.cards.length)show(data.cards[0]);
</script>''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, default=ROOT / 'training/layout_review/annotations/archive_113_p0054.user-linked.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/record_flow_current')
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text(encoding='utf-8-sig'))
    geometry = [{k: e[k] for k in ('id', 'role', 'boxes')} for e in data['elements']]
    result = infer_record_flow(geometry)
    metrics = evaluate(data, result)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'predictions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    make_report(data, result, metrics, args.output / 'report.html')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
