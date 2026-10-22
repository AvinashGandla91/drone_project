"""Live webcam prediction and conservative simulation-only safety decisions."""
import argparse
from pathlib import Path
import time
import numpy as np
from .model import load_model, predict
from .safety import SafetyGate
from .labels import UNKNOWN
from .movement_features import StabilityWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gesture_model.joblib")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        raise SystemExit(f"No trained model at {args.model}. Collect real samples and train first.")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install webcam dependencies: pip install -e '.[webcam]'") from exc
    from .landmarks import normalize_landmarks
    payload = load_model(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")
    gate = SafetyGate()
    stability_window = StabilityWindow(window_size=15)
    try:
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=.6, min_tracking_confidence=.6) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    gate.reset()
                    stability_window.reset()
                    print("Camera lost; pending command state cleared.")
                    break
                now = time.monotonic()
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label, confidence = UNKNOWN, 0.0
                stability = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                    try:
                        features = normalize_landmarks(points)
                        label, confidence = predict(payload, features)
                        stability = stability_window.update(features, label, now, position=points[0])
                    except ValueError:
                        label, confidence = UNKNOWN, 0.0
                        stability_window.reset()
                else:
                    # Missing hand breaks continuity; do not reuse stale stability.
                    stability_window.reset()
                    gate.reset()

                if stability is None:
                    decision = gate.process(label, confidence, None, "UNAVAILABLE")
                    stability_line = "STABILITY: WARMING UP / UNAVAILABLE"
                else:
                    decision = gate.process(label, confidence, stability.score, stability.status)
                    stability_line = f"STABILITY: {stability.score:.2f} | {stability.status}"
                lines = [
                    f"POSE: {label} | confidence: {confidence:.2f}",
                    stability_line,
                    f"DECISION: {decision.decision}",
                    f"SIMULATION COMMAND: {decision.accepted_command or 'NONE'}",
                    "SIMULATION ONLY - NO PHYSICAL DRONE CONTROL",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 26+i*28), cv2.FONT_HERSHEY_SIMPLEX, .53, (255,255,255), 2)
                cv2.imshow("Drone gesture safety prototype - simulation only", frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    finally:
        gate.reset()
        stability_window.reset()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
