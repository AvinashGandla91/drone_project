"""Windowed temporal movement proxies for the research stability score.

These are engineering proxies, not physiological measurements or flight-safety probabilities.
A score is emitted only after a sufficiently long, finite, time-ordered window exists.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from .stability import calculate_stability, motion_entropy_proxy


@dataclass(frozen=True)
class WindowMetrics:
    score: float
    status: str
    smoothness: float
    gesture_consistency: float
    velocity_fluctuation: float
    tremor_proxy: float
    motion_entropy: float


class StabilityWindow:
    """Maintain recent normalized landmark vectors and labels for temporal proxies."""

    def __init__(self, window_size: int = 15, min_dt: float = 1e-4, max_dt: float = 0.5):
        if window_size < 5:
            raise ValueError("window_size must be at least 5 frames")
        self.window_size = int(window_size)
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self._samples = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._samples.clear()

    def update(self, features, label: str, timestamp: float, position=None) -> WindowMetrics | None:
        vector = np.asarray(features, dtype=np.float64)
        stamp = float(timestamp)
        # Use the wrist's image-space location to retain whole-hand translation,
        # which the pose-normalized classifier features intentionally remove.
        anchor = np.asarray(position if position is not None else vector[:3], dtype=np.float64)
        if (vector.shape != (63,) or not np.isfinite(vector).all()
                or anchor.shape != (3,) or not np.isfinite(anchor).all()
                or not math.isfinite(stamp)):
            self.reset()
            return None
        if self._samples:
            dt = stamp - self._samples[-1][0]
            if dt <= self.min_dt or dt > self.max_dt:
                self.reset()
        self._samples.append((stamp, vector.copy(), str(label), anchor.copy()))
        if len(self._samples) < self.window_size:
            return None

        times = np.asarray([s[0] for s in self._samples], dtype=np.float64)
        points = np.stack([s[3] for s in self._samples])
        labels = [s[2] for s in self._samples]
        dts = np.diff(times)
        if not np.isfinite(dts).all() or (dts <= self.min_dt).any() or (dts > self.max_dt).any():
            self.reset()
            return None

        # Mean per-frame landmark displacement, corrected for irregular frame times.
        velocities = np.diff(points, axis=0) / dts[:, None]
        speed = np.linalg.norm(velocities, axis=1)
        # Normalize speed by a robust scale so V is dimensionless and non-negative.
        speed_scale = float(np.median(speed)) + 1e-6
        velocity_fluctuation = float(np.std(speed) / speed_scale)
        velocity_fluctuation = min(velocity_fluctuation, 100.0)

        # Smoothness proxy: lower second-difference energy maps to higher Q.
        accel = np.diff(velocities, axis=0)
        accel_energy = float(np.mean(np.linalg.norm(accel, axis=1)))
        # Normalize acceleration by a characteristic speed-per-time scale so Q
        # is less sensitive to the camera's frame rate and hand scale.
        characteristic_accel = (float(np.median(speed)) + 1e-6) / (float(np.median(dts)) + 1e-6)
        normalized_accel = accel_energy / (characteristic_accel + 1e-6)
        smoothness = 1.0 / (1.0 + min(normalized_accel, 100.0))

        # Tremor proxy: high-frequency residual around a local 3-frame moving average.
        if len(points) >= 5:
            smooth_track = np.stack([
                (points[i - 1] + points[i] + points[i + 1]) / 3.0
                for i in range(1, len(points) - 1)
            ])
            residual = points[1:-1] - smooth_track
            tremor_proxy = float(np.mean(np.linalg.norm(residual, axis=1)))
        else:
            tremor_proxy = 0.0
        tremor_proxy = min(tremor_proxy, 100.0)

        gesture_consistency = labels.count(labels[-1]) / len(labels)
        # Label-transition entropy is diagnostic only, not part of the historical formula.
        unique = sorted(set(labels))
        counts = np.asarray([labels.count(k) for k in unique], dtype=np.float64)
        entropy = motion_entropy_proxy(counts)

        result = calculate_stability(
            smoothness=smoothness,
            gesture_consistency=gesture_consistency,
            velocity_fluctuation=velocity_fluctuation,
            tremor_proxy=tremor_proxy,
        )
        return WindowMetrics(
            score=result.score,
            status=result.status,
            smoothness=smoothness,
            gesture_consistency=gesture_consistency,
            velocity_fluctuation=velocity_fluctuation,
            tremor_proxy=tremor_proxy,
            motion_entropy=entropy,
        )
