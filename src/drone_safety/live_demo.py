"""Live webcam prediction and conservative simulation-only safety decisions."""
import argparse
from pathlib import Path
import numpy as np
from .model import load_model, predict
from .safety import SafetyGate
from .labels import UNKNOWN

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
    try:
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=.6, min_tracking_confidence=.6) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    gate.reset()
                    print("Camera lost; pending command state cleared.")
                    break
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label, confidence = UNKNOWN, 0.0
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                    try:
                        label, confidence = predict(payload, normalize_landmarks(points))
                    except ValueError:
                        label, confidence = UNKNOWN, 0.0
                # Temporal stability features are not yet implemented as a validated
                # window. Do not fabricate a score: directional and LAND requests
                # remain held; STOP/HOLD can still override immediately.
                decision = gate.process(label, confidence, None, "UNAVAILABLE")
                lines = [f"POSE: {label} | confidence: {confidence:.2f}",
                         "STABILITY: UNAVAILABLE (not measured)",
                         f"DECISION: {decision.decision}",
                         f"SIMULATION COMMAND: {decision.accepted_command or 'NONE'}"]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 28+i*30), cv2.FONT_HERSHEY_SIMPLEX, .58, (255,255,255), 2)
                cv2.imshow("Drone gesture safety prototype - simulation only", frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    finally:
        gate.reset()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
