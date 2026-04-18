#!/usr/bin/env python3
"""
Train an XGBoost ranker for rapid-threat priority and export ONNX.
Fallback architecture for Feature 3 in ADITI package.

Outputs:
- ai/rapid/data/rapid_ranker.onnx
- ai/rapid/data/ranker_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from skl2onnx import to_onnx


def synthesize(n: int = 20000, seed: int = 13):
    rng = np.random.default_rng(seed)

    mach = rng.uniform(0.5, 12.0, n)
    altitude = rng.uniform(5.0, 250.0, n)
    burnout = rng.uniform(0.0, 6.0, n)
    flight_angle = rng.uniform(0.05, 1.1, n)
    distance = rng.uniform(50.0, 2500.0, n)
    rcs = rng.uniform(0.05, 6.0, n)

    class_boost = np.where((mach >= 5) & (altitude >= 20) & (altitude <= 100), 100, np.where((burnout >= 1) & (altitude < 300), 80, np.where(altitude >= 120, 40, 20)))
    dist_boost = np.where(distance <= 500, 50, np.where(distance <= 1500, 25, 0))
    mach_boost = np.where(mach >= 8, 25, np.where(mach >= 5, 15, 0))
    stealth_boost = np.where(rcs < 0.5, 10, 0)

    baseline = class_boost + dist_boost + mach_boost + stealth_boost

    # Synthetic surrogate objective: dynamic score with noise and interaction terms.
    dynamic = baseline + 0.9 * mach + 0.02 * (2500 - distance) + 3.0 * burnout + rng.normal(0, 4.5, n)
    dynamic = np.clip(dynamic, 0, 300)

    x = np.stack([mach, altitude, burnout, flight_angle, distance, rcs, baseline], axis=1).astype(np.float32)
    y = dynamic.astype(np.float32)
    return x, y


def main():
    x, y = synthesize()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    pred = model.predict(x_val)
    mae = float(mean_absolute_error(y_val, pred))

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    onx = to_onnx(model, x_train[:1], target_opset=17)
    onnx_path = out_dir / "rapid_ranker.onnx"
    with onnx_path.open("wb") as f:
        f.write(onx.SerializeToString())

    metrics = {
        "model": "XGBRegressor priority ranker",
        "feature_order": [
            "mach",
            "altitude_km",
            "burnout_velocity_km_s",
            "flight_path_angle_rad",
            "distance_to_asset_km",
            "radar_cross_section",
            "baseline_priority",
        ],
        "val_mae": mae,
        "onnx": str(onnx_path),
    }
    with (out_dir / "ranker_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"exported={onnx_path}")
    print(f"val_mae={mae:.4f}")


if __name__ == "__main__":
    main()
