"""
train_model.py

Trains a RandomForestClassifier to predict subsidence risk_level
(Low/Medium/High/Critical) from multi-node sensor time-series data, and
derives a continuous 0-100 "risk percentage" plus a debounced alert
signal from the model's class probabilities.

Pipeline:
1. Load data, sort by node + time.
2. Feature engineering: rolling-window stats per node (raw readings alone
   are weak signals for a physical process like subsidence).
3. Time-respecting train/test split (never randomly shuffle time series).
4. Train RandomForestClassifier with class balancing.
5. Evaluate: per-class precision/recall/F1, confusion matrix, feature
   importances.
6. Convert predict_proba -> risk_pct, and demonstrate a debounced alert
   rule so single noisy readings don't fire false alarms.
7. Save the trained model + feature list with joblib.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = "/home/claude/synthetic_subsidence_data.csv"
MODEL_PATH = "/home/claude/subsidence_rf_model.joblib"

RAW_FEATURES = [
    "displacement_mm", "velocity_mm_day", "acceleration_mm_day2",
    "vibration_rms", "vibration_peak", "dominant_frequency_hz",
    "tilt_deg", "angular_velocity_dps",
]
ROLL_WINDOW = 24        # 24 hourly readings = 1 day of context
CLASS_ORDER = ["Low", "Medium", "High", "Critical"]
CLASS_MIDPOINTS = {"Low": 12.5, "Medium": 37.5, "High": 62.5, "Critical": 87.5}


# ---------------------------------------------------------------- #
# 1. Load
# ---------------------------------------------------------------- #
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
df["risk_level"] = pd.Categorical(df["risk_level"], categories=CLASS_ORDER, ordered=True)


# ---------------------------------------------------------------- #
# 2. Feature engineering: rolling-window stats per node
#    (mean / std / slope over the trailing window, plus cumulative
#    displacement from each node's own baseline)
# ---------------------------------------------------------------- #
def rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Linear trend (units per reading) over the trailing window."""
    idx = np.arange(window)
    idx_mean = idx.mean()
    idx_var = ((idx - idx_mean) ** 2).sum()

    def _slope(vals):
        return ((vals - vals.mean()) * (idx - idx_mean)).sum() / idx_var

    return x.rolling(window).apply(_slope, raw=True)


engineered_cols = []
for col in RAW_FEATURES:
    grp = df.groupby("node_id")[col]
    df[f"{col}_roll_mean"] = grp.transform(lambda s: s.rolling(ROLL_WINDOW).mean())
    df[f"{col}_roll_std"] = grp.transform(lambda s: s.rolling(ROLL_WINDOW).std())
    df[f"{col}_roll_slope"] = grp.transform(lambda s: rolling_slope(s, ROLL_WINDOW))
    engineered_cols += [f"{col}_roll_mean", f"{col}_roll_std", f"{col}_roll_slope"]

df["cum_displacement"] = df.groupby("node_id")["displacement_mm"].transform(
    lambda s: s - s.iloc[0]
)
engineered_cols.append("cum_displacement")

FEATURE_COLS = RAW_FEATURES + engineered_cols

# drop the first (ROLL_WINDOW - 1) rows per node -> NaNs from rolling calc
df_model = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"Rows after feature engineering (dropped rolling warm-up): {len(df_model)}")


# ---------------------------------------------------------------- #
# 3. Time-respecting split: per node, first 80% of timestamps = train,
#    last 20% = test. (Alternative: hold out whole nodes to test
#    generalization to new locations -- worth doing once you have more
#    nodes than 15.)
# ---------------------------------------------------------------- #
train_parts, test_parts = [], []
for nid, g in df_model.groupby("node_id"):
    g = g.sort_values("timestamp")
    cut = int(len(g) * 0.8)
    train_parts.append(g.iloc[:cut])
    test_parts.append(g.iloc[cut:])

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

X_train, y_train = train_df[FEATURE_COLS], train_df["risk_level"]
X_test, y_test = test_df[FEATURE_COLS], test_df["risk_level"]

print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")


# ---------------------------------------------------------------- #
# 4. Train
# ---------------------------------------------------------------- #
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)


# ---------------------------------------------------------------- #
# 5. Evaluate
# ---------------------------------------------------------------- #
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=3)
print("\nClassification report:\n", report)

cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

macro_f1 = f1_score(y_test, y_pred, labels=CLASS_ORDER, average="macro")
print(f"\nMacro F1: {macro_f1:.3f}")

importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
top_importances = importances.sort_values(ascending=False).head(12)
print("\nTop 12 feature importances:")
print(top_importances.round(4))


# ---------------------------------------------------------------- #
# 6. risk_pct from predict_proba + a debounced alert rule
# ---------------------------------------------------------------- #
def risk_percentage(proba: np.ndarray, class_order: list) -> np.ndarray:
    """Weighted sum of class probabilities against class midpoints ->
    a single interpretable 0-100 risk score."""
    midpoints = np.array([CLASS_MIDPOINTS[c] for c in class_order])
    return proba @ midpoints


proba_test = model.predict_proba(X_test)
test_df = test_df.copy()
test_df["risk_pct"] = risk_percentage(proba_test, list(model.classes_))


def debounced_alerts(risk_pct: pd.Series, threshold: float = 70.0, consecutive: int = 3) -> pd.Series:
    """Fire True only when risk_pct has stayed >= threshold for
    `consecutive` readings in a row -- avoids single noisy-reading
    false alarms."""
    above = risk_pct >= threshold
    return above.rolling(consecutive).sum() >= consecutive


test_df["alert"] = test_df.groupby("node_id")["risk_pct"].transform(
    lambda s: debounced_alerts(s)
)

n_alerts = int(test_df["alert"].sum())
print(f"\nDebounced alerts fired on test set (risk_pct>=70 for 3+ readings): {n_alerts}")
print(test_df.loc[test_df["alert"], ["timestamp", "node_id", "risk_level", "risk_pct"]].head(10))


# ---------------------------------------------------------------- #
# 7. Save model + metadata
# ---------------------------------------------------------------- #
joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "class_order": list(model.classes_)}, MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")

metrics_out = {
    "macro_f1": macro_f1,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "confusion_matrix": cm.tolist(),
    "class_order": CLASS_ORDER,
    "top_features": top_importances.round(4).to_dict(),
    "per_class": classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True),
}
with open("/home/claude/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)
print("Saved metrics.json")
