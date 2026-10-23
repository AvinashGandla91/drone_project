import pytest
from drone_safety.labels import GESTURE_CLASSES, validate_label, display_label

def test_exact_class_order():
    assert GESTURE_CLASSES == ("LEFT", "RIGHT", "FORWARD", "BACKWARD", "LAND", "UPWARD", "STOP_HOLD")

def test_stop_display_and_validation():
    assert validate_label("STOP / HOLD") == "STOP_HOLD"
    assert display_label("STOP_HOLD") == "STOP / HOLD"
    with pytest.raises(ValueError): validate_label("SWIPE_LEFT")
