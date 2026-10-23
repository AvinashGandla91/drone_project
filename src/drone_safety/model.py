"""Grouped session holdout, Random Forest training, and confidence rejection."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from .labels import GESTURE_CLASSES, UNKNOWN
from .landmarks import FEATURE_DIM, validate_feature_vector
SCHEMA_VERSION = 1
CONFIDENCE_THRESHOLD = 0.75

def train_model(X, y, sessions, random_state=42):
    X, y, sessions = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=str), np.asarray(sessions, dtype=str)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM or len(X) != len(y) or len(y) != len(sessions):
        raise ValueError("Inconsistent feature, label, and session dimensions")
    if not np.isfinite(X).all():
        raise ValueError("Features contain non-finite values")
    absent = sorted(set(GESTURE_CLASSES) - set(y.tolist()))
    if absent:
        raise ValueError(f"Missing required classes: {absent}")
    if len(np.unique(sessions)) < 2:
        raise ValueError("At least two recording sessions are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    tr, te = next(splitter.split(X, y, groups=sessions))
    missing_tr = sorted(set(GESTURE_CLASSES) - set(y[tr].tolist()))
    missing_te = sorted(set(GESTURE_CLASSES) - set(y[te].tolist()))
    if missing_tr or missing_te:
        raise ValueError(f"Grouped split lacks classes: train={missing_tr}; test={missing_te}. Collect more balanced sessions.")
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    metrics = {
        "accuracy": float(accuracy_score(y[te], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "classification_report": classification_report(y[te], pred, labels=list(GESTURE_CLASSES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y[te], pred, labels=list(GESTURE_CLASSES)).tolist(),
        "class_order": list(GESTURE_CLASSES), "train_sessions": sorted(set(sessions[tr])),
        "test_sessions": sorted(set(sessions[te])), "test_samples": int(len(te)),
    }
    clf.fit(X, y)
    return clf, metrics

def save_model(path, classifier, confidence_threshold=CONFIDENCE_THRESHOLD):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": SCHEMA_VERSION, "class_order": list(GESTURE_CLASSES),
                 "feature_dim": FEATURE_DIM, "confidence_threshold": confidence_threshold,
                 "classifier": classifier}, p)

def load_model(path):
    payload = joblib.load(path)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("class_order") != list(GESTURE_CLASSES) or payload.get("feature_dim") != FEATURE_DIM:
        raise ValueError("Model schema mismatch")
    return payload

def predict(payload, features):
    vector = validate_feature_vector(features)
    clf = payload["classifier"]
    probabilities = clf.predict_proba(vector.reshape(1, -1))[0]
    i = int(np.argmax(probabilities)); confidence = float(probabilities[i])
    label = str(clf.classes_[i]) if confidence >= float(payload.get("confidence_threshold", CONFIDENCE_THRESHOLD)) else UNKNOWN
    return label, confidence
