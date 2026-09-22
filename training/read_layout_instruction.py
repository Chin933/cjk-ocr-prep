"""Read the instruction's paragraphs and embedded figures without changing it."""
import argparse
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('docx',type=Path)
parser.add_argument('--output',type=Path,default=Path('runs/instruction_review'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
embed='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
with zipfile.ZipFile(args.docx) as z:
    relations={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
    paragraphs=[]
    for index,p in enumerate(ET.fromstring(z.read('word/document.xml')).findall('.//w:p',ns)):
        text=''.join(t.text or '' for t in p.findall('.//w:t',ns))
        figures=[relations[b.attrib[embed]] for b in p.findall('.//a:blip',ns)]
        if text or figures:paragraphs.append({'paragraph':index,'text':text,'figures':figures})
    (args.output/'paragraphs.json').write_text(json.dumps(paragraphs,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=[]
    for name in z.namelist():
        if name.startswith('word/media/'):
            image=Image.open(io.BytesIO(z.read(name))).convert('RGB')
            image.save(args.output/Path(name).name)
            figures.append((Path(name).name,image))
    for start in range(0,len(figures),12):
        sheet=Image.new('RGB',(1200,1200),'#eeeeea');draw=ImageDraw.Draw(sheet)
        for i,(name,figure) in enumerate(figures[start:start+12]):
            figure.thumbnail((390,265));x,y=(i%3)*400,(i//3)*300
            sheet.paste(figure,(x,y+25));draw.text((x+4,y+4),name,fill='black')
        sheet.save(args.output/f'figures_{start//12+1}.jpg')
print(json.dumps({'paragraphs':len(paragraphs),'figures':len(figures),'output':str(args.output)}))
