"""Safety-gated command decisions; output is simulation-only."""
from dataclasses import dataclass
import math
from .labels import DIRECTIONAL_COMMANDS, UNKNOWN
CONFIDENCE_THRESHOLD = 0.75

@dataclass(frozen=True)
class Decision:
    prediction: str
    confidence: float
    stability_score: float | None
    stability_status: str
    decision: str
    accepted_command: str | None
    simulation_only: bool = True

class SafetyGate:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in (0,1]")
        self.confidence_threshold = confidence_threshold
        self._pending_label = None
        self._pending_count = 0
        self.active_command = None

    def reset(self):
        self._pending_label = None
        self._pending_count = 0
        self.active_command = None

    def process(self, prediction: str, confidence: float, stability_score: float | None,
                stability_status: str) -> Decision:
        label = str(prediction).upper()
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = float("nan")
        try:
            score = float(stability_score) if stability_score is not None else float("nan")
        except (TypeError, ValueError):
            score = float("nan")
        valid_score = math.isfinite(score) and 0 <= score <= 1

        # Stop is an immediate override even if confidence/stability is low.
        if label in {"STOP", "STOP_HOLD"}:
            self._pending_label, self._pending_count = None, 0
            self.active_command = "STOP_HOLD"
            return Decision("STOP_HOLD", conf, score if valid_score else None, stability_status,
                            "ACCEPTED_IMMEDIATE_OVERRIDE", "STOP_HOLD")
        if label == "LAND":
            self._pending_label, self._pending_count = None, 0
            if not math.isfinite(conf) or conf < self.confidence_threshold:
                return Decision(label, conf, score if valid_score else None, stability_status, "REJECTED_LOW_CONFIDENCE", None)
            if not valid_score or score < 0.72:
                return Decision(label, conf, score if valid_score else None, stability_status, "HELD_LAND_SAFETY_GATE", None)
            self.active_command = "LAND"
            return Decision(label, conf, score, stability_status, "ACCEPTED_LAND_REQUEST", "LAND")
        if label not in DIRECTIONAL_COMMANDS or not math.isfinite(conf) or conf < self.confidence_threshold:
            self._pending_label, self._pending_count = None, 0
            return Decision(label or UNKNOWN, conf, score if valid_score else None, stability_status,
                            "REJECTED_UNKNOWN_OR_LOW_CONFIDENCE", None)
        if not valid_score or score < 0.60:
            self._pending_label, self._pending_count = None, 0
            return Decision(label, conf, score if valid_score else None, stability_status, "HELD_LOW_STABILITY", None)
        if score < 0.72:
            self._pending_label, self._pending_count = None, 0
            return Decision(label, conf, score, stability_status, "HELD_UNCERTAIN_STABILITY", None)
        self._pending_count = self._pending_count + 1 if self._pending_label == label else 1
        self._pending_label = label
        if self._pending_count < 2:
            return Decision(label, conf, score, stability_status, "PENDING_CONFIRMATION", None)
        self.active_command = label
        self._pending_label, self._pending_count = None, 0
        return Decision(label, conf, score, stability_status, "ACCEPTED", label)
