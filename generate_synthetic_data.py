"""
generate_synthetic_data.py

Creates a synthetic multi-node sensor time-series dataset for subsidence
risk modeling, since no real labeled data exists yet.

Design intent (read this before you swap in real data):
- Most nodes are "stable": small noise around a baseline, no trend, with
  occasional brief noise "blips" (e.g. passing traffic, minor transient
  vibration) that are NOT real subsidence -- this is intentional, so the
  model has to learn to tell a sustained trend apart from a one-off spike,
  which is exactly why rolling-window/trend features matter later.
- A subset of nodes develop a simulated subsidence event partway through
  the timeline, following an S-curve severity ramp (slow onset ->
  accelerating -> plateau), each with its own severity cap so some
  events top out at Medium/High and others reach Critical.
- velocity and acceleration are derived from displacement (not generated
  independently), and angular_velocity from tilt, so the dataset is
  internally physically consistent, same as real sensors would be.
- The 'risk_level' label is generated from the same underlying severity
  curve that drives the features (plus its own noise), which is a
  STAND-IN for real incident labels or a validated engineering rule.
  When real data arrives, replace `severity_to_label()` with actual
  historical outcomes.

Output: synthetic_subsidence_data.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_NODES = 15
N_AT_RISK = 5                 # nodes that develop a subsidence event
DAYS = 180                    # 6 months
HOURS = DAYS * 24
FREQ = "h"

START = pd.Timestamp("2026-01-01")
timestamps = pd.date_range(START, periods=HOURS, freq=FREQ)

node_ids = [f"NODE_{i:03d}" for i in range(1, N_NODES + 1)]
at_risk_nodes = set(RNG.choice(node_ids, size=N_AT_RISK, replace=False))

severity_profiles = {}
for nid in at_risk_nodes:
    onset_frac = RNG.uniform(0.15, 0.5)
    cap = RNG.uniform(0.4, 1.05)              # some events never reach full severity
    ramp_hours = RNG.uniform(700, 2600)        # gradual, weeks-to-months ramp
    severity_profiles[nid] = {
        "onset_hour": onset_frac * HOURS,
        "cap": cap,
        "ramp_hours": ramp_hours,
    }

rows = []

for nid in node_ids:
    is_at_risk = nid in at_risk_nodes
    baseline_freq = RNG.uniform(8.0, 15.0)
    baseline_tilt = RNG.uniform(-0.3, 0.3)
    t = np.arange(HOURS)

    # --- severity curve s(t) in [0, ~1.1], drives BOTH features and label ---
    if is_at_risk:
        onset = severity_profiles[nid]["onset_hour"]
        cap = severity_profiles[nid]["cap"]
        ramp = severity_profiles[nid]["ramp_hours"]
        s = cap / (1 + np.exp(-(t - onset - ramp / 2) / (ramp / 7)))
        s = np.where(t < onset, 0.0, s)
    else:
        s = np.zeros(HOURS)

    s = s + np.abs(RNG.normal(0, 0.015, HOURS))          # sensor-level noise
    blip_mask = RNG.random(HOURS) < 0.004                 # rare transient blips
    s = s + blip_mask * RNG.uniform(0.15, 0.35, HOURS)
    s = np.clip(s, 0, 1.15)

    # --- features derived from severity + independent physical noise ---
    disp_noise = RNG.normal(0, 0.02, HOURS).cumsum() * 0.03
    displacement = disp_noise + s * 55 + RNG.normal(0, 0.06, HOURS)

    velocity = np.gradient(displacement) * 24.0 + RNG.normal(0, 0.04, HOURS)
    acceleration = np.gradient(velocity) * 24.0 + RNG.normal(0, 0.02, HOURS)

    tilt_noise = RNG.normal(0, 0.001, HOURS).cumsum() * 0.03
    tilt = baseline_tilt + tilt_noise + s * 1.1 + RNG.normal(0, 0.004, HOURS)
    angular_velocity = np.gradient(tilt) * 24.0 + RNG.normal(0, 0.01, HOURS)

    vib_rms = 0.15 + 0.02 * np.abs(RNG.normal(0, 1, HOURS)) + s * 0.55
    vib_peak = vib_rms * RNG.uniform(2.5, 4.0, HOURS)
    freq_hz = np.clip(baseline_freq - s * 4.2 + RNG.normal(0, 0.15, HOURS), 1.0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "node_id": nid,
        "displacement_mm": displacement,
        "velocity_mm_day": velocity,
        "acceleration_mm_day2": acceleration,
        "vibration_rms": vib_rms,
        "vibration_peak": vib_peak,
        "dominant_frequency_hz": freq_hz,
        "tilt_deg": tilt,
        "angular_velocity_dps": angular_velocity,
        "_severity": s,  # internal only, dropped before saving
    })
    rows.append(df)

data = pd.concat(rows, ignore_index=True)


def severity_to_label(s: np.ndarray) -> pd.Categorical:
    """STAND-IN labeling rule: buckets the severity curve (+ its own
    label-level noise) into Low/Medium/High/Critical. Replace with real
    incident-derived labels when available."""
    s_label = s + RNG.normal(0, 0.05, len(s))
    bins = [-np.inf, 0.12, 0.35, 0.65, np.inf]
    labels = ["Low", "Medium", "High", "Critical"]
    return pd.cut(s_label, bins=bins, labels=labels)


data["risk_level"] = severity_to_label(data["_severity"].to_numpy())
data = data.drop(columns=["_severity"])
data = data.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

out_path = "/home/claude/synthetic_subsidence_data.csv"
data.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(data), "| Nodes:", data['node_id'].nunique())
print("\nRisk level distribution:")
print((data["risk_level"].value_counts(normalize=True) * 100).round(2))
print("\nAt-risk nodes:", sorted(at_risk_nodes))
