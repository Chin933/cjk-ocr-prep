"""Build the multi-page, note-first human review alongside the frozen views."""

import base64
import json
from pathlib import Path
import re
import argparse

ROOT=Path(__file__).resolve().parent.parent
folder=ROOT/'training/layout_review'
parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,default=ROOT/'runs/note_first')
parser.add_argument('--baseline',type=Path)
parser.add_argument('--output',default='review_20260916_notes.html')
parser.add_argument('--external-rules',action='store_true')
parser.add_argument('--review-progress',action='store_true')
parser.add_argument('--typed',action='store_true')
args=parser.parse_args()
artifacts=args.artifacts
template=(folder/'review_20260916_v1.html').read_text(encoding='utf-8')
match=re.search(r'const pages=([\s\S]*?);\nconst \$',template)
pages=[]
for summary in json.loads((artifacts/'summary.json').read_text()):
    stem=summary['image']
    result=json.loads((artifacts/f'{stem}.json').read_text(encoding='utf-8'))
    gold=folder/'annotations'/f'{stem}.user-linked.json'
    human=json.loads(gold.read_text(encoding='utf-8-sig'))['elements'] if gold.exists() else []
    note=(f"训练页拟合检查：附注 {result['fit_check']['matched_iou50']}/{result['fit_check']['gold']} 匹配（IoU ≥ 0.5），不是跨页准确率。"
          if human else '未参与训练的页面，尚无人工框作定量评估。')
    if result.get('checkpoint'):
        regions_path=Path(result['checkpoint']).parent/'training_regions.json'
        if regions_path.exists():
            trained={r['image'] for r in json.loads(regions_path.read_text(encoding='utf-8'))}
            if stem in trained and not human:note='本页的确认框或局部修正参与了本轮训练。'
    missing_lanes=not any('boundaries' in b for b in result['bands'])
    diagnostic='栏位恢复失败，未生成附注框' if missing_lanes else ('未检出附注，待核查' if not result['notes'] and not (args.typed and result['primary_candidates']) else '')
    if args.typed:
        names={'personal_biography':'个人履历','associate_biography':'亲属履历','exam':'考试／考官','essay':'文章','poem':'诗文'}
        note=names[result['page_kind']]+' · 样本类型已预先核对，非自动分类结果。'+note
    if diagnostic:note+=' 【'+diagnostic+'】'
    pages.append({'name':stem+'.jpg','width':result['size'][0],'height':result['size'][1],
                  'human':human,'predicted':result['notes']+result['primary_candidates'],
                  'rules':((artifacts.resolve().relative_to(folder.resolve())/f'{stem}.rules.png').as_posix() if args.external_rules
                           else 'data:image/png;base64,'+base64.b64encode((artifacts/f'{stem}.rules.png').read_bytes()).decode()),
                  'bands':result['bands'],'note':note,'diagnostic':diagnostic,
                  'fingerprint':result.get('pipeline_fingerprint'),'pageKind':result.get('page_kind')})
    if args.baseline:
        old_path=Path(result['baseline_file']) if args.typed else args.baseline/f'{stem}.json'
        old=json.loads(old_path.read_text(encoding='utf-8'))
        pages[-1]['baseline']=old['notes']+old['primary_candidates']
output=template[:match.start(1)]+json.dumps(pages,ensure_ascii=False).replace('<','\\u003c')+template[match.end(1):]


def change(old,new):
    global output
    if old not in output:
        raise ValueError(f'Missing template fragment: {old[:80]}')
    output=output.replace(old,new)


change('框识别 V1 · 人工验收',f'附注优先 · {len(pages)} 页验收')
change('let current=0,scale=1',"let current=Math.max(0,Math.min(pages.length-1,Number(new URLSearchParams(location.search).get('page'))||0)),scale=1")
change('2026-09-16 固定快照 · 紫色主文，橙色附注','橙色附注 · 紫色主文候选')
change('<button id="page1" class="active">图 1：书页 52</button><button id="page2">图 2：书页 219</button>',
       '<select id="pageSelect">'+''.join(f'<option value="{i}">{i+1} · {Path(p["name"]).stem}</option>' for i,p in enumerate(pages))+'</select>')
change('id="primary" checked>主文框','id="primary">主文候选')
change('id="annotation">附注框','id="annotation" checked>附注框')
change('<button id="out">','<label><input type="checkbox" id="rules">右侧显示补线</label><button id="out">')
change('本版含正文范围和同列宽度整理；减少黑线依赖的训练尚未完成。',
       '默认只看附注。主文候选由附注以外的实际文字区间推得，尚不代表完整记录或跨栏归属。补线图：青色为检测线段，红色为推断部分。')
change('算法预测 · 待验收','附注优先结果 · 待验收')
change("img.src='../images/archive_diverse/'+page.name;paper.style.width=872*scale+'px';paper.style.height=1280*scale+'px';layer.replaceChildren();",
       "img.src=side==='right'&&$('rules').checked?page.rules:'../images/archive_diverse/'+page.name;paper.style.width=page.width*scale+'px';paper.style.height=page.height*scale+'px';layer.setAttribute('viewBox',`0 0 ${page.width} ${page.height}`);layer.replaceChildren();")
change("$('page1').classList.toggle('active',current===0);$('page2').classList.toggle('active',current===1);","$('pageSelect').value=current;")
change("['primary','annotation','human','labels']","['primary','annotation','human','labels','rules']")
change("$('fit').onclick=()=>setZoom(($('left').clientWidth-40)/872);","$('fit').onclick=()=>setZoom(($('left').clientWidth-40)/pages[current].width);")
change("$('left').scrollTop=560*scale;","$('left').scrollTop=(pages[current].bands[1]?.band?.[1]||pages[current].height*.45)*scale;")
change("for(let i=0;i<2;i++)$('page'+(i+1)).onclick=()=>{current=i;selected=null;$('selection').textContent='点击任一框，可查看编号。';render();$('left').scrollTop=0;$('right').scrollTop=0;};",
       "$('pageSelect').onchange=()=>{current=Number($('pageSelect').value);selected=null;scale=Math.min(1,($('left').clientWidth-40)/pages[current].width);$('selection').textContent='点击任一框，可查看编号。';render();$('left').scrollTop=0;$('right').scrollTop=0;};")
change("(e.role==='primary'?'主文':'附注')","(e.role==='primary'?'主文候选':'附注')")
change("const box=document.createElementNS(ns,'rect');", "const box=document.createElementNS(ns,e.polygon?'polygon':'rect');if(e.polygon)box.setAttribute('points',e.polygon.map(p=>p.join(',')).join(' '));")
if args.baseline:
    change(f'附注优先 · {len(pages)} 页验收','附注优先 V2 · 旧版对照')
    change('<label><input type="checkbox" id="human"', '<label><input type="checkbox" id="baseline" checked>左侧显示上一版</label><label><input type="checkbox" id="human"')
    change("side==='right'?page.predicted:$('human').checked?page.human:[]", "side==='right'?page.predicted:$('baseline').checked?page.baseline:$('human').checked?page.human:[]")
    change("$('leftTitle').textContent=page.human.length&&$('human').checked?'原图 ＋ 你的标注':'原图';", "$('leftTitle').textContent=$('baseline').checked?'上一版 · 旧编号':page.human.length&&$('human').checked?'原图 ＋ 你的标注':'原图';")
    change("['primary','annotation','human','labels','rules']","['primary','annotation','human','labels','rules','baseline']")
    change("(side==='left'?'人工':'预测')","(side==='left'?$('leftTitle').textContent:'新版')")
    change('补线图：青色为检测线段，红色为推断部分。','蓝色线为局部栏线方向估计；新版用四边形显示倾斜边界。左侧保留旧编号，右侧编号已重新生成。')
change("scale=Math.min(1,($('left').clientWidth-40)/872);render();","scale=Math.min(1,($('left').clientWidth-40)/pages[current].width);render();")
if args.review_progress:
    if not args.baseline:
        change('补线图：青色为检测线段，红色为推断部分。','蓝色线为局部栏线方向估计；四边形边界跟随估计斜率。')
    change('<div class="columns">','''<div class="toolbar" id="reviewControls">
<button id="previousPage">上一页</button><button id="nextPage">下一页</button>
<button id="nextUnreviewed">下个未检查</button><button id="nextEmpty">下个空结果</button>
<label>本页 <select id="reviewState"><option value="unreviewed">未检查</option><option value="ok">基本正确</option><option value="issue">有问题</option></select></label>
<span id="reviewCount"></span><span id="saveState" role="status"></span>
<button id="exportReview">导出反馈</button><button id="importReview">导入反馈</button><input id="importFile" type="file" accept=".json" hidden>
<textarea id="reviewText" rows="2" aria-label="本页问题" placeholder="例如：N0014–N0016 应合并；右下方漏了“监生”。直接输入，自动保存。"></textarea>
</div><div class="columns">''')
    change('</style>','''#reviewText{width:100%;font:inherit;padding:7px;resize:vertical;min-height:45px}
#reviewControls{gap:8px}#saveState{font-size:12px}.viewport{height:calc(100vh - 350px);min-height:260px}
</style>''')
    change('</script></html>','</script><script src="batch_review.js"></script></html>')
if args.typed:
    change('附注优先 V2 · 旧版对照' if args.baseline else f'附注优先 · {len(pages)} 页验收',f'分类型识别 · {len(pages)} 页对照')
    change('id="primary">主文候选','id="primary" checked>主文候选')
    change('默认只看附注。主文候选由附注以外的实际文字区间推得，尚不代表完整记录或跨栏归属。',
           '紫色为主文候选，橙色为附注候选。个人履历、亲属履历、考试、文章和诗文分别处理；本轮不评估跨栏归属。')
    change('附注优先结果 · 待验收','分类型结果 · 待验收')
    if any(e.get('excluded_marks') for p in pages for e in p['predicted']):
        change('<button id="out">','<label><input type="checkbox" id="markMask" checked>显示圈点排除层</label><button id="out">')
        change("for(const e of elements){", """for(const e of elements){
   if(side==='right'&&$('markMask').checked)for(const points of e.excluded_marks||[]){
    const mask=document.createElementNS(ns,'polygon');mask.setAttribute('points',points.map(p=>p.join(',')).join(' '));
    mask.setAttribute('fill','#ffffff');mask.setAttribute('stroke','#68b7b0');mask.setAttribute('stroke-width','.4');mask.setAttribute('pointer-events','none');layer.appendChild(mask);
   }""")
        change("for(const id of ['primary'", "$('markMask').onchange=render;\nfor(const id of ['primary'")
        change('本轮不评估跨栏归属。','本轮不评估跨栏归属。青色轮廓标出已排除圈点，可关闭排除层查看原墨迹；原始图片未改动。')
(folder/args.output).write_text(output,encoding='utf-8')
print(folder/args.output)
