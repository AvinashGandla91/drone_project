import numpy as np
import pytest
from drone_safety.model import train_model, predict
from drone_safety.labels import GESTURE_CLASSES
from drone_safety.landmarks import FEATURE_DIM

def test_training_refuses_missing_classes():
    X = np.zeros((14, FEATURE_DIM), dtype=np.float32)
    y = np.array(["LEFT"]*14)
    sessions = np.array(["s1"]*7+["s2"]*7)
    with pytest.raises(ValueError, match="Missing required classes"):
        train_model(X, y, sessions)
