"""Extract contextual correction cases from a reviewed prediction batch."""
import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feedback', type=Path)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    feedback = json.loads(args.feedback.read_text(encoding='utf-8-sig'))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'feedback.json').write_bytes(args.feedback.read_bytes())
    cases = []
    for name, entry in feedback['pages'].items():
        if entry['status'] != 'issue':
            continue
        stem = Path(name).stem
        result = json.loads((args.batch/f'{stem}.json').read_text(encoding='utf-8'))
        image = Image.open(ROOT/'training/images/archive_diverse'/name).convert('RGB')
        ids = list(dict.fromkeys(re.findall(r'[NM]\d{4}', entry['note'])))
        elements = {e['id']: e for e in result['notes']+result['primary_candidates']}
        for ident in ids:
            element = elements[ident]
            x1,y1,x2,y2 = element['boxes'][0]
            width = x2-x1
            bounds = [max(0,x1-width),max(0,y1-60),min(image.width,x2+width),min(image.height,y2+60)]
            original = image.crop(bounds)
            marked = original.copy()
            draw = ImageDraw.Draw(marked)
            for nearby in elements.values():
                a,b,c,d = nearby['boxes'][0]
                if a>=bounds[2] or c<=bounds[0] or b>=bounds[3] or d<=bounds[1]:
                    continue
                color = '#ec7900' if nearby['role']=='annotation' else '#8745c4'
                draw.rectangle((a-bounds[0],b-bounds[1],c-bounds[0],d-bounds[1]),outline=color,width=1)
                draw.text((a-bounds[0],b-bounds[1]),nearby['id'],fill=color)
            filename = f'{stem}.{ident}.png'
            panel = Image.new('RGB',(original.width*2,original.height),'white')
            panel.paste(original,(0,0));panel.paste(marked,(original.width,0))
            panel.resize((panel.width*3,panel.height*3)).save(args.output/filename)
            cases.append({'page':stem,'id':ident,'crop':filename,'bounds':bounds,
                          'predicted_box':element['boxes'][0],'feedback':entry['note'],
                          'label_status':'needs_corrected_geometry'})
    (args.output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(cases)} contextual correction crops: {args.output}')


if __name__ == '__main__':
    main()
