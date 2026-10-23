"""Position/scale-normalized 21-point landmark features."""
import numpy as np
LANDMARK_COUNT = 21
FEATURE_DIM = 63

def normalize_landmarks(landmarks) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected (21, 3) landmarks, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks must be finite")
    points = points - points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Degenerate landmarks")
    return (points / scale).astype(np.float32).reshape(FEATURE_DIM)

def validate_feature_vector(features) -> np.ndarray:
    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {FEATURE_DIM} finite features")
    return vector
