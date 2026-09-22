"""Assemble the diverse full-page corpus and its frozen development split."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
GROUPS={
 'biography':{127:[56,58],113:[54,219,53,55,220,221,222],29:[237,236,238],155:[396,395,397],
              71:[55,56,57,230,231,232],99:[220,221,222],169:[381],183:[54],197:[53],239:[385]},
 'exam':{85:[221,220],99:[54,55],127:[230],155:[55,56],197:[380],253:[359],295:[365]},
 'paper':{15:[55,57,56],420:[387,386,388],85:[223],141:[52],211:[221],309:[204],169:[54],113:[389]}}
EVAL={'biography':{71,169,183,239},'exam':{197,253,295},'paper':{141,211,309}}


def main():
    original=json.loads((ROOT/'training/layout_review/page_types_20260918.json').read_text(encoding='utf-8'))
    known={p['image']:p for p in original['pages']}
    pages=[]
    for group,volumes in GROUPS.items():
        for volume,numbers in volumes.items():
            for number in numbers:
                stem=f'archive_{volume:03d}_p{number:04d}'
                kind={'biography':'associate_biography','exam':'exam','paper':'essay'}[group]
                if (volume,number)==(155,395):kind='personal_biography'
                row=known.get(stem,{'image':stem,'kind':kind,'case':'diverse_archive_page'}).copy()
                row.update(family=group,volume=volume,pdf_page=number,
                           split='evaluation' if volume in EVAL[group] else 'development')
                pages.append(row)
    assert len(pages)==50 and len({p['image'] for p in pages})==50
    output=ROOT/'training/layout_review/corpus_50';output.mkdir(exist_ok=True)
    for group in GROUPS:
        selected=[p for p in pages if p['family']==group]
        for start in range(0,len(selected),12):
            sheet=Image.new('RGB',(1200,((len(selected[start:start+12])+3)//4)*410),'#eee')
            draw=ImageDraw.Draw(sheet)
            for index,row in enumerate(selected[start:start+12]):
                image=Image.open(ROOT/'training/images/archive_diverse'/f"{row['image']}.jpg")
                image.thumbnail((292,380));x=index%4*300;y=index//4*410
                sheet.paste(image,(x,y+25));draw.text((x+3,y+4),row['image'],fill='black')
            sheet.save(output/f'{group}_{start//12+1}.jpg',quality=90)
    result={'pages':pages,'split_semantics':'frozen_for_this_training_round; earlier review history retained',
            'source':'existing full-page archive renders; instruction crops tracked separately'}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print({group:sum(p['family']==group for p in pages) for group in GROUPS})
    print({split:sum(p['split']==split for p in pages) for split in ['development','evaluation']})


if __name__=='__main__':main()
