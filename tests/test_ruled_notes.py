import json

import numpy as np
from PIL import Image,ImageDraw

from digitalization.ruled_notes import decode_notes,compound_glyph_rows


def test_compound_glyph_uses_whole_character_strokes_not_central_gap():
    image=Image.new('L',(50,240),255)
    draw=ImageDraw.Draw(image)
    for y,weight in [(8,4),(48,4),(90,1),(125,1),(160,1),(195,1)]:
        for x in (5,28):
            draw.line((x,y,x,y+27),fill=0,width=weight)
            for dy in (0,12,26):draw.line((x,y+dy,x+15,y+dy),fill=0,width=weight)
    ink=np.asarray(image)<180
    p=np.zeros(ink.shape,np.float32);p[45:225]=.95
    anchors=np.zeros(240,bool);anchors[6:38]=True
    blocked=compound_glyph_rows(ink,50,p,anchors)
    assert blocked[50:73].all()
    assert not blocked[90:118].any()


def test_compound_glyph_requires_main_text_context():
    ink=np.zeros((100,50),bool)
    ink[20:48,5:18]=True;ink[20:48,28:43]=True
    assert not compound_glyph_rows(ink,50,np.ones(ink.shape),np.zeros(100,bool)).any()


def test_notes_stay_in_lanes_and_main_text_comes_from_remaining_ink():
    image=Image.new('L',(120,160),255)
    draw=ImageDraw.Draw(image)
    for x in (20,60):
        draw.rectangle((x+6,20,x+30,48),fill=0)
        draw.rectangle((x+6,65,x+16,105),fill=0)
        draw.rectangle((x+22,65,x+32,105),fill=0)
    band={'band':[20,10,100,150],'boundaries':[20,60,100]}
    probability=np.zeros((160,120),np.float32)
    probability[62:108,20:100]=.95
    notes,primary=decode_notes(image,[band],probability)
    assert len(notes)==2
    assert [n['boxes'][0][::2] for n in notes]==[[20,60],[60,100]]
    assert len(primary)==2
    assert all(p['boxes'][0][3]<60 for p in primary)
    assert all(p['review']=='candidate' for p in primary)
    json.dumps(notes+primary)


def test_blank_lanes_do_not_generate_main_text_or_notes():
    image=Image.new('L',(120,160),255)
    band={'band':[20,10,100,150],'boundaries':[20,60,100]}
    probability=np.ones((160,120),np.float32)
    assert decode_notes(image,[band],probability)==([],[])


def test_short_note_is_not_erased_by_its_boundary_prediction():
    image=Image.new('L',(80,100),255)
    draw=ImageDraw.Draw(image)
    draw.rectangle((25,40,35,51),fill=0)
    draw.rectangle((43,40,53,51),fill=0)
    p=np.zeros((100,80),np.float32);p[39:53,20:60]=.8
    edge=np.ones_like(p)
    notes,_=decode_notes(image,[{'band':[20,10,60,90],'boundaries':[20,60]}],p,edge)
    assert len(notes)==1
    assert notes[0]['boxes'][0][1]<=40 and notes[0]['boxes'][0][3]>=52


def test_continuous_note_bridges_short_low_probability_space():
    image=Image.new('L',(80,160),255)
    draw=ImageDraw.Draw(image)
    for y in (30,70):
        for x in (25,43):draw.rectangle((x,y,x+9,y+19),fill=0)
    p=np.zeros((160,80),np.float32);p[28:52,20:60]=.9;p[68:92,20:60]=.9
    notes,_=decode_notes(image,[{'band':[20,10,60,150],'boundaries':[20,60]}],p)
    assert len(notes)==1


def test_slanted_lane_returns_polygon_without_clipping_note():
    image=Image.new('L',(100,160),255)
    draw=ImageDraw.Draw(image)
    for x in (30,47):draw.rectangle((x,60,x+9,84),fill=0)
    p=np.zeros((160,100),np.float32);p[55:90,20:70]=.95
    band={'band':[20,10,60,150],'boundaries':[20,60],
          'slanted_tracks':[[20,28],[60,68]],'horizontal_edges':[[10,12],[150,152]]}
    notes,_=decode_notes(image,[band],p)
    assert len(notes)==1
    points=notes[0]['polygon']
    assert points[3][0]>points[0][0]
    assert points[1][1]>points[0][1]
    assert notes[0]['boxes'][0][1]<60 and notes[0]['boxes'][0][3]>84
