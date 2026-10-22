import numpy as np
import pytest

from drone_safety.movement_features import StabilityWindow


def test_window_warms_up_then_emits_finite_metrics():
    window = StabilityWindow(window_size=5)
    features = np.zeros(63, dtype=np.float32)
    results = [window.update(features, "LEFT", i / 30) for i in range(5)]
    assert all(x is None for x in results[:-1])
    metric = results[-1]
    assert metric is not None
    assert 0 <= metric.score <= 1
    assert metric.status == "HIGHER STABILITY"
    assert metric.gesture_consistency == 1
    assert metric.motion_entropy == pytest.approx(0)


def test_label_changes_reduce_consistency_and_add_entropy():
    window = StabilityWindow(window_size=5)
    features = np.zeros(63, dtype=np.float32)
    labels = ["LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"]
    metric = None
    for i, label in enumerate(labels):
        metric = window.update(features, label, i / 30)
    assert metric is not None
    assert metric.gesture_consistency == pytest.approx(3 / 5)
    assert metric.motion_entropy > 0


def test_bad_feature_vector_resets_window():
    window = StabilityWindow(window_size=5)
    assert window.update(np.zeros(62), "LEFT", 0.0) is None
    for i in range(4):
        assert window.update(np.zeros(63), "LEFT", i / 30) is None


def test_nonmonotonic_timestamp_resets_temporal_history():
    window = StabilityWindow(window_size=5)
    for i in range(4):
        window.update(np.zeros(63), "LEFT", i / 30)
    assert window.update(np.zeros(63), "LEFT", 0.01) is None
    assert len(window._samples) == 1
