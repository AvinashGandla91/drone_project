"""Historical score formula; feature semantics are proxies, not medical measures."""
from dataclasses import dataclass
import math
LOW_THRESHOLD = 0.60
HIGH_THRESHOLD = 0.72

@dataclass(frozen=True)
class StabilityResult:
    score: float
    status: str

def stability_status(score: float) -> str:
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Score must be finite and in [0, 1]")
    if score < LOW_THRESHOLD:
        return "LOW MOVEMENT STABILITY"
    if score < HIGH_THRESHOLD:
        return "UNCERTAIN"
    return "HIGHER STABILITY"

def calculate_stability(*, smoothness: float, gesture_consistency: float,
                         velocity_fluctuation: float, tremor_proxy: float) -> StabilityResult:
    q, g, v, t = map(float, (smoothness, gesture_consistency, velocity_fluctuation, tremor_proxy))
    if not all(map(math.isfinite, (q, g, v, t))):
        raise ValueError("Inputs must be finite")
    if not (0 <= q <= 1 and 0 <= g <= 1) or v < 0 or t < 0:
        raise ValueError("Q and G must be in [0,1]; V and T must be non-negative")
    score = 0.35*q + 0.30*g + 0.20/(1+v) + 0.15/(1+10*t)
    return StabilityResult(min(1.0, max(0.0, score)), stability_status(score))

def motion_entropy_proxy(probabilities) -> float:
    """Shannon entropy in nats, retained as a separate diagnostic."""
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Expected finite non-negative 1D probabilities with positive sum")
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
