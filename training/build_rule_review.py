"""Publish the frozen original/restored input comparison for manual review."""

import base64
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent.parent
folder=ROOT/'training/layout_review'
artifacts=ROOT/'runs/rule_restoration'
template=(folder/'review_20260916_v1.html').read_text(encoding='utf-8')
match=re.search(r'const pages=([\s\S]*?);\nconst \$',template)
pages=json.loads(match.group(1))
for page in pages:
    stem=Path(page['name']).stem
    for key,variant in [('baseline','original'),('predicted','restored')]:
        page[key]=json.loads((artifacts/f'{stem}.{variant}.predicted.json').read_text(encoding='utf-8'))
    for variant in ['rules','restored']:
        page[variant]='data:image/png;base64,'+base64.b64encode((artifacts/f'{stem}.{variant}.png').read_bytes()).decode()
    page['note']='本页参与了模型训练，仅作拟合检查。' if page['human'] else '本页没有参与模型训练，请人工检查效果。'
output=template[:match.start(1)]+json.dumps(pages,ensure_ascii=False).replace('<','\\u003c')+template[match.end(1):]
output=output.replace('框识别 V1 · 人工验收','栏线恢复 · 同模型前后对照')
output=output.replace('2026-09-16 固定快照','固定模型 · 仅输入图像不同')
output=output.replace('<label><input type="checkbox" id="human" checked>左侧显示你的标注</label>',
    '<label><input type="checkbox" id="showBoxes">显示识别框</label><label><input type="checkbox" id="ruleOverlay" checked>显示补线来源</label><label><input type="checkbox" id="human">左侧改看人工框</label>')
output=output.replace('本版含正文范围和同列宽度整理；减少黑线依赖的训练尚未完成。',
    '右侧青色是检测到的线段，红色是推断补出的部分，并非原始真值。勾选「显示识别框」比较结果；可关闭补线颜色。')
output=output.replace('算法预测 · 待验收','补线图／补线后识别 · 待验收')
output=output.replace("?'原图 ＋ 你的标注':'原图';","?'原图 ＋ 你的标注':'原图／补线前识别';")
output=output.replace("img.src='../images/archive_diverse/'+page.name;",
    "img.src=side==='right'?page[$('ruleOverlay').checked?'rules':'restored']:'../images/archive_diverse/'+page.name;")
output=output.replace("side==='right'?page.predicted:$('human').checked?page.human:[]",
    "side==='right'?page.predicted:$('human').checked?page.human:page.baseline")
output=output.replace("for(const e of elements){","for(const e of elements){if(!$('showBoxes').checked)continue;")
output=output.replace("['primary','annotation','human','labels']","['primary','annotation','human','labels','showBoxes','ruleOverlay']")
output=output.replace("e.id.replace('box_','P')","e.id.replace('box_',side==='right'?'R':'O')")
(folder/'review_20260916_rules.html').write_text(output,encoding='utf-8')
print(folder/'review_20260916_rules.html')
