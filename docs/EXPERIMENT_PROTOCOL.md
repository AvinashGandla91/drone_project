# Evaluation protocol

## Evidence categories
Keep deterministic unit tests, offline model evaluation, live webcam trials, simulation results, and physical-hardware trials separate. This repository only emits simulation decisions; physical control is disabled.

## Data collection
Use the supplied reference image as the authoritative pose guide. Collect all seven labels across multiple sessions. The operator must deliberately hold the matching pose; the collector's selected label is metadata, not a prediction. Store normalized numeric landmarks, label, timestamp, and random session ID. Do not store video by default. Vary camera-frame position, distance, lighting, and hand orientation.

## Offline evaluation
Use a grouped holdout by recording session. Report accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix, sample count, and exact train/test session IDs. The current trainer refuses to proceed if any class is absent from the train or test group. This conservative constraint may require more balanced recording sessions. Never use test data for tuning.

## Live trials
Record for each class: attempts, correct, incorrect, UNKNOWN, confidence values, conditions, and software/camera environment. Include transitions, position/distance changes, lighting, partial occlusion, ambiguous poses, missing hand, and STOP while a direction is pending. Live trials remain pending until run and logged by an operator.

## Latency
Measure capture-to-prediction and capture-to-decision with a monotonic clock, after warm-up; report sample count, median and 95th percentile. Record machine, OS, Python, camera, and dependency versions.

## Stability features
- Tremor proxy: mean high-frequency landmark residual from a 3-frame moving average over the current normalized-landmark window.
- Trajectory smoothness proxy Q: inverse transform of mean second-difference energy of per-frame landmark velocities, `Q = 1 / (1 + min(mean_acceleration_norm, 100))`.
- Gesture consistency G: fraction of labels in the window equal to the current frame's label, range [0,1].
- Velocity fluctuation V: standard deviation of per-frame mean landmark speed divided by median speed plus a small epsilon; capped at 100.
- Motion entropy: Shannon entropy of label frequencies in nats; diagnostic only and deliberately not added to the historical score.
The live demo now computes these proxies from a 15-frame window. It waits for a complete valid window before showing a score; missing hands or discontinuous timestamps clear the window. These are engineering proxies, not physiological measurements, and need empirical validation. Their scales and thresholds are provisional.

## Formula and thresholds
Historical formula as recorded: S=0.35Q+0.30G+0.20/(1+V)+0.15/(1+10T). Q/G in [0,1], V/T non-negative. Historical status thresholds are 0.60 and 0.72. The original source could not be cross-checked because the repository was empty at initialization. This score is not a calibrated flight-safety probability.

## Known recognition limitation
Position/scale-normalized 2D landmarks can discard useful absolute orientation cues and may not reliably distinguish palm-side from back-of-hand views. The reference pose distinctions, especially FORWARD/BACKWARD and the fifth LAND pose, must be validated using real samples. If classes overlap, improve features using orientation/handedness cues or return UNKNOWN rather than inventing a label.
