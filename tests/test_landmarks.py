import numpy as np
import pytest
from drone_safety.landmarks import normalize_landmarks, FEATURE_DIM

def points():
    p = np.zeros((21,3)); p[:,0] = np.linspace(0,1,21); p[:,1] = np.linspace(0,.5,21); p[:,2] = np.linspace(0,.1,21)
    return p

def test_normalization_invariant_to_translation_and_scale():
    p = points()
    assert normalize_landmarks(p).shape == (FEATURE_DIM,)
    assert np.allclose(normalize_landmarks(p), normalize_landmarks(p*3 + [5,7,2]), atol=1e-6)

def test_invalid_landmarks_rejected():
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((20,3)))
    p = points(); p[2,0] = np.nan
    with pytest.raises(ValueError): normalize_landmarks(p)
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((21,3)))
