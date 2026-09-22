import numpy as np
from PIL import Image,ImageDraw

from digitalization.page_structure import merged_headings,decode_streams,parallel_ranges,decode_page
from digitalization.ruled_notes import complete_ink_edges


def test_partial_probability_edge_recovers_whole_glyph_not_next_glyph():
    ink=np.zeros((100,40),bool)
    ink[20:38,5:14]=True;ink[20:38,23:33]=True
    ink[50:68,5:33]=True
    a,b=complete_ink_edges(ink,28,36,40,np.zeros(100,bool))
    assert a<=20 and 38<=b<50


def test_parallel_small_columns_are_one_region():
    ink=np.zeros((160,60),bool)
    for x in (3,17,31,45):
        for y in range(10,140,12):ink[y:y+8,x:x+9]=True
    spans=parallel_ranges(ink,60)
    assert len(spans)==1
    assert spans[0][0]<=10 and spans[0][1]>=138


def test_full_height_radical_pairs_are_not_small_parallel_text():
    ink=np.zeros((180,50),bool)
    for y in (10,65,120):
        ink[y:y+38,3:19]=True;ink[y:y+38,28:45]=True
    assert parallel_ranges(ink,50)==[]


def test_large_name_is_one_merged_region_without_changing_lower_register():
    im=Image.new('L',(400,500),255);d=ImageDraw.Draw(im)
    for y in (30,105,180):d.rectangle((250,y,350,y+62),fill=0)
    bands=[{'band':[20,20,380,250],'pitch':45,'boundaries':list(range(20,381,45))},
           {'band':[20,270,380,480],'pitch':45,'boundaries':list(range(20,381,45))}]
    headings=merged_headings(im,bands)
    assert len(headings)==1 and headings[0]['band']==0
    assert headings[0]['boxes'][0][0]<250 and headings[0]['boxes'][0][2]>350


def test_sparse_main_stream_keeps_large_vertical_spaces():
    im=Image.new('L',(100,300),255);d=ImageDraw.Draw(im)
    for y in (25,115,225):d.rectangle((32,y,60,y+25),fill=0)
    band={'band':[20,10,80,280],'boundaries':[20,80],'pitch':60}
    notes,primary=decode_streams(im,[band])
    assert notes==[] and len(primary)==1
    assert primary[0]['boxes'][0][1]<=25 and primary[0]['boxes'][0][3]>=250


def test_exam_type_does_not_use_biography_probability_even_with_two_registers():
    im=Image.new('L',(100,300),255);d=ImageDraw.Draw(im)
    for y in (25,80,175,230):d.rectangle((32,y,60,y+25),fill=0)
    bands=[{'band':[20,a,80,b],'boundaries':[20,80],'pitch':60} for a,b in [(10,140),(160,290)]]
    notes,primary,headings=decode_page(im,bands,np.ones((300,100),np.float32),'exam')
    assert notes==[] and len(primary)==2 and headings==[]
    assert all(e['page_kind']=='exam' for e in primary)


def test_poem_separates_author_gap_that_examiner_path_keeps():
    im=Image.new('L',(100,400),255);d=ImageDraw.Draw(im)
    for y in (25,60,320):d.rectangle((32,y,60,y+25),fill=0)
    bands=[{'band':[20,10,80,380],'boundaries':[20,80],'pitch':60}]
    p=np.zeros((400,100),np.float32)
    _,poem,_=decode_page(im,bands,p,'poem')
    _,exam,_=decode_page(im,bands,p,'exam')
    assert len(poem)==2 and len(exam)==1


def test_unresolved_page_type_requires_an_explicit_choice():
    import pytest
    with pytest.raises(ValueError,match='Unresolved page type'):
        decode_page(Image.new('L',(10,10)),[],np.zeros((10,10)),'unknown')
