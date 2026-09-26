# Subsidence Risk — Random Forest Prototype

A working prototype on **synthetic** data, built so the pipeline (feature
engineering → model → risk % → alert) is ready to point at real sensor
data as soon as you have it.

## Files

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | Creates `synthetic_subsidence_data.csv` — 15 nodes, hourly readings over 6 months, 5 of which develop a simulated subsidence event. **Delete this step once you have real data.** |
| `train_model.py` | Feature engineering, train/test split, Random Forest training, evaluation, risk % scoring, and a demo alert rule. |
| `synthetic_subsidence_data.csv` | The generated dataset (~65k rows). |
| `subsidence_rf_model.joblib` | Trained model + feature list, loadable with `joblib.load(...)`. |
| `metrics.json` | Evaluation metrics from the last training run. |

## Results on the synthetic set

Accuracy 96.2%, macro F1 0.77. Low/High/Critical are all strong
(F1 0.85–0.99). **Medium is the weak point** (recall 17.5%) because it's
a narrow transition band that overlaps heavily with Low in feature space
— a realistic failure mode, not a bug. Two ways to address it on real
data: merge Medium into a wider "Elevated" band if your use case doesn't
need 4-way granularity, or add features specifically aimed at *early*
onset detection (e.g. shorter rolling windows) since Medium mostly
represents the very start of a developing event.

## How the pieces fit together

- **Feature engineering**: raw instantaneous readings are weak signals
  for a physical process like subsidence, so each raw feature gets a
  24-reading (1-day) rolling mean, rolling std, and rolling slope
  (trend), plus cumulative displacement from each node's baseline. This
  is what let the model tell a real trend apart from single-reading
  noise.
- **Split**: time-respecting — first 80% of each node's timeline for
  training, last 20% for testing. Never shuffle time-series data
  randomly; it leaks. Once you have more than 15 nodes, also try
  holding out entire nodes to check generalization to new locations.
- **risk_pct**: `predict_proba` gives P(Low), P(Medium), P(High),
  P(Critical) per reading. These are weighted against class midpoints
  (12.5 / 37.5 / 62.5 / 87.5) into a single 0–100 score — see
  `risk_percentage()` in `train_model.py`.
- **Alerting**: `debounced_alerts()` only fires when risk_pct has
  stayed ≥70 for 3 consecutive readings, so a single noisy spike
  doesn't trigger a false alarm. Tune the threshold and window to your
  false-alarm/missed-detection tolerance.

## Swapping in real data

1. Replace `severity_to_label()`'s role entirely — your real
   `risk_level` should come from actual incident records, inspection
   logs, or a validated engineering threshold rule, not synthetic
   severity.
2. Keep the column names matching `RAW_FEATURES` in `train_model.py`,
   or update that list.
3. Re-run `train_model.py` directly on your real CSV (skip the
   generator).
4. Recheck class balance — real subsidence events will likely be rarer
   than in this synthetic set, so you may need `class_weight` tuning,
   SMOTE, or an anomaly-detection framing (e.g. Isolation Forest as a
   first-pass filter before the classifier) if High/Critical examples
   are very scarce.
5. Validate the debounce threshold against real false-alarm tolerance
   before wiring it to actual alerts.
