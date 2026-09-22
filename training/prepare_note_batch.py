"""Extend the existing archive image sample with nearby unseen source pages."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--output',type=Path,default=ROOT/'training/layout_review/batches/notes_100')
    args=parser.parse_args()
    images=ROOT/'training/images/archive_diverse'
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'manifest.json'
    renderer=shutil.which('pdftoppm')
    if renderer is None:raise RuntimeError('pdftoppm is required')
    if manifest.exists():
        records=json.loads(manifest.read_text(encoding='utf-8'))['pages']
    else:
        records=[]
        for path in sorted(images.glob('archive_*.jpg')):
            volume,page=map(int,re.fullmatch(r'archive_(\d+)_p(\d+)',path.stem).groups())
            records.append({'image':path.stem,'volume':volume,'page':page,'existing':True})
        anchors=list(records);seen={(r['volume'],r['page']) for r in records}
        counts={}
        for offset in (1,-1,2,-2,3,-3):
            for anchor in anchors:
                if len(records)>=args.count:break
                volume,page=anchor['volume'],anchor['page']+offset
                if volume not in counts:
                    with fitz.open(args.archive/f'{volume}.pdf') as doc:counts[volume]=len(doc)
                if (volume,page) in seen or not 20<page<=counts[volume]:continue
                records.append({'image':f'archive_{volume:03d}_p{page:04d}',
                                'volume':volume,'page':page,'existing':False})
                seen.add((volume,page))
        records=records[:args.count]
        manifest.write_text(json.dumps({'archive':str(args.archive),'pages':records},indent=2),encoding='utf-8')
    new=[r for r in records if not r['existing']]
    for index,record in enumerate(new):
        target=images/record['image']
        if not target.with_suffix('.jpg').exists():
            subprocess.run([renderer,'-f',str(record['page']),'-l',str(record['page']),
                            '-singlefile','-scale-to','1280','-jpeg','-jpegopt','quality=90',
                            str(args.archive/f"{record['volume']}.pdf"),str(target)],check=True,timeout=60)
        print(f"Rendered {index+1}/{len(new)}: {record['image']}",flush=True)
    for start in range(0,len(new),20):
        sheet=Image.new('RGB',(1200,1400),'#eeeeea');draw=ImageDraw.Draw(sheet)
        for i,record in enumerate(new[start:start+20]):
            image=Image.open(images/(record['image']+'.jpg'));image.thumbnail((232,320))
            x,y=(i%5)*240,(i//5)*350
            sheet.paste(image,(x,y+22));draw.text((x+2,y+3),record['image'],fill='black')
        sheet.save(args.output/f'sample_contact_{start//20+1}.jpg',quality=85)
    print(json.dumps({'pages':len(records),'new':len(new),'volumes':len({r['volume'] for r in records})}))


if __name__=='__main__':main()
