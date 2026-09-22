"""Freeze a V1/V2 image comparison with shortcuts to inspected failures."""

import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent.parent


def main():
    folder=ROOT/'training/layout_review'
    template=(folder/'review_20260916_v1.html').read_text(encoding='utf-8')
    match=re.search(r'const pages=([\s\S]*?);\nconst \$',template)
    pages=json.loads(match.group(1))
    for page in pages:
        page['baseline']=page['predicted']
        name=Path(page['name']).stem
        page['predicted']=json.loads((ROOT/'runs/box_detector_v2_final'/f'{name}.predicted.json').read_text(encoding='utf-8'))
        page['note']='图 1 的整页标注已用于 V2 训练，本页用于拟合验收。' if page['human'] else '图 2 没有用于训练，用于检查跨页效果。'
    payload=json.dumps(pages,ensure_ascii=False).replace('<','\\u003c')
    output=template[:match.start(1)]+payload+template[match.end(1):]
    output=output.replace('框识别 V1 · 人工验收','框识别 V2 · 新旧对照')
    output=output.replace('2026-09-16 固定快照','V1 / V2 固定快照')
    output=output.replace('<label><input type="checkbox" id="human" checked>左侧显示你的标注</label>',
                          '<select id="leftMode"><option value="baseline">左侧：V1 预测</option><option value="human">左侧：你的标注</option><option value="original">左侧：原图</option></select>')
    output=output.replace('本版含正文范围和同列宽度整理；减少黑线依赖的训练尚未完成。','V2：二维实例分离、主文和附注分别学习边界、混合清晰线与弱线训练。')
    output=output.replace('算法预测 · 待验收','V2 预测 · 编号 N · 待验收')
    output=output.replace("$('human').disabled=!page.human.length;$('leftTitle').textContent=page.human.length&&$('human').checked?'原图 ＋ 你的标注':'原图';",
                          "$('leftMode').querySelector('[value=human]').disabled=!page.human.length;if(!page.human.length&&$('leftMode').value==='human')$('leftMode').value='original';$('leftTitle').textContent=({baseline:'V1 预测 · 编号 P',human:'你的标注',original:'原图'})[$('leftMode').value];")
    output=output.replace("side==='right'?page.predicted:$('human').checked?page.human:[]", "side==='right'?page.predicted:$('leftMode').value==='original'?[]:page[$('leftMode').value]")
    output=output.replace("['primary','annotation','human','labels']","['primary','annotation','leftMode','labels']")
    output=output.replace("(side==='left'?'人工':'预测')","(side==='left'?$('leftTitle').textContent:'V2')")
    output=output.replace("e.id.replace('box_','P')","e.id.replace('box_',side==='right'?'N':'P')")
    output=output.replace('<div class="columns">','<div class="toolbar" id="issues">定位旧问题：</div><div class="columns">')
    output=output.replace('height:calc(100vh - 218px)','height:calc(100vh - 268px)')
    shortcuts="""
for(const id of ['0014','0007','0029','0033','0055','0063','0064','0065','0086']){
 const b=document.createElement('button');b.textContent='P'+id;
 b.onclick=()=>{current=0;const e=pages[0].baseline.find(e=>e.id==='box_'+id);if(!e)return;
 $('leftMode').value='baseline';$('primary').checked=e.role==='primary';$('annotation').checked=e.role==='annotation';
 selected='left'+e.id;setZoom(1);const r=e.boxes[0];$('left').scrollTop=Math.max(0,r[1]-70);$('left').scrollLeft=Math.max(0,(r[0]+r[2])/2-$('left').clientWidth/2);
 $('selection').textContent='旧问题 P'+id+'：左侧 V1，右侧 V2 同位置。';};$('issues').appendChild(b);
}
"""
    output=output.replace("scale=Math.min(1,($('left').clientWidth-40)/872);render();",shortcuts+"\nscale=Math.min(1,($('left').clientWidth-40)/872);render();")
    (folder/'review_20260916_v2.html').write_text(output,encoding='utf-8')


if __name__=='__main__':
    main()
