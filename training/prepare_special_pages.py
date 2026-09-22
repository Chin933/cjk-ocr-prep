"""Render a small held-out neighbourhood of known archive structures."""
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('D:/Programs/Keju/Zhujuanjicheng/1_420')
PAGES=[(85,223),(127,58),(15,57),(420,386),(113,219)]
images=ROOT/'training/images/archive_diverse'
output=ROOT/'runs/special_page_sample';output.mkdir(parents=True,exist_ok=True)
renderer=shutil.which('pdftoppm')
if renderer is None:raise RuntimeError('pdftoppm missing')
records=[]
sheet=Image.new('RGB',(1200,840),'#eeeae3');draw=ImageDraw.Draw(sheet)
for i,(volume,page) in enumerate(PAGES):
    stem=f'archive_{volume:03d}_p{page:04d}'
    target=images/stem
    existing=target.with_suffix('.jpg').exists()
    if not existing:
        subprocess.run([renderer,'-f',str(page),'-l',str(page),'-singlefile','-scale-to','1280',
                        '-jpeg','-jpegopt','quality=90',str(ARCHIVE/f'{volume}.pdf'),str(target)],check=True,timeout=60)
    records.append({'image':stem,'volume':volume,'pdf_page':page,'previously_rendered':existing})
    im=Image.open(target.with_suffix('.jpg'));im.thumbnail((290,390))
    x,y=i%4*300,i//4*420;sheet.paste(im,(x,y+24));draw.text((x+4,y+4),stem,fill='black')
sheet.save(output/'contact.jpg')
(output/'manifest.json').write_text(json.dumps({'archive':str(ARCHIVE),'pages':records},indent=2),encoding='utf-8')
print(json.dumps(records))
