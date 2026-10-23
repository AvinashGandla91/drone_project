# Cognitive Stability-Aware Safety Framework for Adaptive Vision-Based Drone Navigation

Research prototype for static hand-pose command recognition, movement-stability analysis, and safety-gated simulated commands. Physical drone control is disabled.

## Status

This repository is being initialized. No trained gesture model or evaluation results are included until they are generated from real labeled recordings. Webcam evaluation requires an operator and camera.

## Gesture classes

Fixed class order:

1. `LEFT` — first supplied reference pose
2. `RIGHT` — second supplied reference pose
3. `FORWARD` — third supplied reference pose (palm visible)
4. `BACKWARD` — fourth supplied reference pose (back of hand visible)
5. `LAND` — fifth supplied reference pose, exactly as supplied
6. `UPWARD` — sixth supplied reference pose
7. `STOP_HOLD` — closed fist

Unknown, missing, invalid, or low-confidence input must not create a directional command. LEFT and RIGHT are static pose labels, not swipe directions. The reference composite image is not yet stored in this repository; users should retain it as the authoritative data-collection guide.

## Install

Python 3.10–3.12 is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

MediaPipe and OpenCV camera support can vary by operating system and Python version; use the declared dependencies and verify camera access locally.

## Workflow

1. Collect labeled samples with `python -m drone_safety.collect_dataset --label LEFT` (see CLI help for available labels).
2. Validate collected data before training.
3. Train using `python -m drone_safety.train_model` only after all seven classes have real samples across suitable recording sessions.
4. Run webcam inference using `python -m drone_safety.live_demo`.
5. Run deterministic tests using `pytest`.

Dataset and model outputs are local artifacts and should not be committed by default.

## Safety model

The historical score formula and thresholds must be verified against original project source before being claimed as experimentally validated. The reference implementation documents the previously recorded formula as:

`S = 0.35 Q + 0.30 G + 0.20/(1+V) + 0.15/(1+10T)`

Historical bands: `S < 0.60` LOW MOVEMENT STABILITY; `0.60 <= S < 0.72` UNCERTAIN; `S >= 0.72` HIGHER STABILITY. This is a research score, not a calibrated probability of flight safety. Motion entropy is recorded separately and is not silently added to this formula. The exact feature scales and normalization must be reviewed with collected data.

Directional commands require two consecutive consistent frames and sufficient confidence (initial threshold 0.75). STOP/HOLD immediately clears pending direction requests. LAND has separate semantics. Simulation output never moves or lands a physical drone.

## Live stability window

The live demo now computes provisional movement proxies over a 15-frame window using normalized landmark vectors: trajectory smoothness, label consistency, velocity fluctuation, and a high-frequency tremor proxy. The historical score is not shown until the window is full and valid. Missing hands or timing gaps reset the window. Motion entropy is displayed in research metrics only and is not added to the historical score. These formulas are engineering proxies and require empirical validation; they are not physiological measurements or a calibrated probability of flight safety.

## Evaluation and limitations

Unit tests, offline model evaluation, live webcam trials, simulation results, and physical-hardware tests are separate evidence categories. Do not report accuracy, latency, or safety claims until measured and saved with the dataset split, session IDs, configuration, and environment. 2D landmarks alone may not reliably distinguish palm-side from back-of-hand orientation; verify this experimentally and use an appropriate orientation cue or reject ambiguous input.

## Project layout

- `src/drone_safety/` — reusable feature, model, stability, safety, and camera modules
- `tests/` — deterministic unit tests
- `data/` — local datasets (ignored by Git)
- `results/` — local experiment outputs (ignored by Git)
- `docs/` — protocol and research notes

## Research integrity

No model, benchmark, webcam trial, physical flight, or IEEE result is claimed complete without recorded evidence. Preserve historical experiments separately from static-pose experiments.
