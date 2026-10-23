"""Collect real hand landmarks; selected label is not the model prediction."""
import argparse, csv, time, uuid
from pathlib import Path
import numpy as np
from .labels import GESTURE_CLASSES, validate_label
from .landmarks import normalize_landmarks


def append_sample(path, label, session_id, landmarks, timestamp=None):
    label = validate_label(label)
    features = normalize_landmarks(landmarks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "session_id", "label"] + [f"f{i:02d}" for i in range(63)])
        writer.writerow([timestamp if timestamp is not None else time.time(), session_id, label] + features.tolist())
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=GESTURE_CLASSES)
    parser.add_argument("--output", default="data/gesture_landmarks.csv")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Reuse one ID across all seven labels recorded in the same session. "
             "Use a different ID for a separate recording session.",
    )
    args = parser.parse_args()
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install optional webcam dependencies: pip install -e '.[webcam]'") from exc
    session = args.session_id or uuid.uuid4().hex
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")
    count = 0
    try:
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=.6, min_tracking_confidence=.6) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame unavailable; collection stopped.")
                    break

                # Mirror before detection so preview, drawn landmarks, and saved
                # normalized coordinates all use the same left-right orientation.
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                cv2.putText(frame, f"COLLECTING: {args.label} (not prediction) | saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2)
                cv2.putText(frame, "MIRRORED PREVIEW | SPACE=save valid hand | Q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                cv2.imshow("Dataset collector - mirrored preview", frame)
                key = cv2.waitKey(1) & 0xff
                if key == ord("q"):
                    break
                if key == ord(" ") and points is not None:
                    append_sample(args.output, args.label, session, points)
                    count += 1
                    print(f"Saved {count} sample(s) for {args.label}; session={session}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
