import pytest
from drone_safety.stability import calculate_stability, stability_status, motion_entropy_proxy

def test_formula_maximum():
    assert calculate_stability(smoothness=1, gesture_consistency=1, velocity_fluctuation=0, tremor_proxy=0).score == pytest.approx(1)

def test_threshold_edges():
    assert stability_status(.5999) == "LOW MOVEMENT STABILITY"
    assert stability_status(.60) == "UNCERTAIN"
    assert stability_status(.7199) == "UNCERTAIN"
    assert stability_status(.72) == "HIGHER STABILITY"

def test_bad_inputs():
    with pytest.raises(ValueError): calculate_stability(smoothness=2, gesture_consistency=1, velocity_fluctuation=0, tremor_proxy=0)
    with pytest.raises(ValueError): calculate_stability(smoothness=1, gesture_consistency=1, velocity_fluctuation=-1, tremor_proxy=0)

def test_entropy_separate_from_formula():
    assert motion_entropy_proxy([1,0]) == pytest.approx(0)
    assert motion_entropy_proxy([.5,.5]) > 0
