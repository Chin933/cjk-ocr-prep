"""Small checks of the frame-anchored reconstruction experiment."""

import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('restore_rules',Path(__file__).parents[1]/'training/restore_rules.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_regular_corridors_recover_pitch_without_internal_black_lines():
    ink=np.zeros((200,750),np.uint8)
    for x in range(50,700,50):
        ink[:,x+7:x+44]=1
    result=module.fit_lattice(ink,50,700)
    assert result is not None
    _,pitch,count,_=result
    assert abs(pitch-50)<1
    assert count==13


def test_blank_or_uniform_content_does_not_support_a_lattice():
    assert module.fit_lattice(np.zeros((200,750)),50,700) is None
    assert module.fit_lattice(np.ones((200,750)),50,700) is None


def test_printed_rulings_distinguish_lanes_from_half_width_text_tracks():
    ink=np.zeros((200,750),np.uint8)
    for x in range(50,650,30):
        ink[:,x+6:x+25]=1
    positions=np.arange(110,650,60)
    result=module.fit_lattice(ink,50,650,positions=positions)
    assert result is not None
    assert result[2]==10
    assert abs(result[1]-60)<1


def test_other_band_pitch_requires_support_from_local_corridors():
    ink=np.zeros((200,750),np.uint8)
    for x in range(50,650,30):
        ink[:,x+6:x+25]=1
    result=module.fit_lattice(ink,50,650,pitch_hint=60)
    assert result is not None and result[2]==10
    assert module.fit_lattice(np.ones((200,750)),50,650,pitch_hint=60) is None
