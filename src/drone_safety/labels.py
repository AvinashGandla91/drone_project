"""Canonical seven-class mapping."""
GESTURE_CLASSES = ("LEFT", "RIGHT", "FORWARD", "BACKWARD", "LAND", "UPWARD", "STOP_HOLD")
DIRECTIONAL_COMMANDS = frozenset({"LEFT", "RIGHT", "FORWARD", "BACKWARD", "UPWARD"})
UNKNOWN = "UNKNOWN"

def validate_label(label: str) -> str:
    value = label.strip().upper().replace("/", "_").replace(" ", "_")
    if value == "STOP":
        value = "STOP_HOLD"
    if value not in GESTURE_CLASSES:
        raise ValueError(f"Unsupported label {label!r}; expected one of {GESTURE_CLASSES}")
    return value

def display_label(label: str) -> str:
    return "STOP / HOLD" if label == "STOP_HOLD" else label
