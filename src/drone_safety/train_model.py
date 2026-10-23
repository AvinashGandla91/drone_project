"""Train and evaluate on a session-grouped holdout."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from .labels import GESTURE_CLASSES, validate_label
from .landmarks import FEATURE_DIM
from .model import train_model, save_model

def load_csv(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"timestamp", "session_id", "label"} | {f"f{i:02d}" for i in range(FEATURE_DIM)}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Dataset missing columns: {sorted(expected - set(reader.fieldnames or []))}")
        for row in reader:
            label = validate_label(row["label"])
            values = [float(row[f"f{i:02d}"]) for i in range(FEATURE_DIM)]
            if not np.isfinite(values).all(): raise ValueError("Dataset has non-finite features")
            rows.append((values, label, row["session_id"]))
    if not rows: raise ValueError("Dataset is empty")
    return np.asarray([r[0] for r in rows], dtype=np.float32), np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gesture_landmarks.csv")
    parser.add_argument("--model-out", default="models/gesture_model.joblib")
    parser.add_argument("--metrics-out", default="results/gesture_metrics.json")
    args = parser.parse_args()
    X, y, sessions = load_csv(args.data)
    clf, metrics = train_model(X, y, sessions)
    save_model(args.model_out, clf)
    path = Path(args.metrics_out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {args.model_out}")
    print(f"Saved holdout metrics: {args.metrics_out}")
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "test_samples", "train_sessions", "test_sessions")}, indent=2))

if __name__ == "__main__": main()
