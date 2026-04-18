#!/usr/bin/env python3
"""
Fit Prophet baseline for blind-spot ratio forecasting (24-72h).
If Prophet is unavailable, script falls back to a deterministic seasonal-trend baseline.

Outputs:
- ai/coverage/data/prophet_forecast_72h.csv
- ai/coverage/data/prophet_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_series(n_hours=24 * 21, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    y = 0.36 + 0.05 * np.sin(2 * np.pi * (t % 24) / 24.0) + 0.0002 * t + rng.normal(0, 0.01, size=t.shape)
    y = np.clip(y, 0.02, 0.98)
    ds = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"ds": ds, "y": y})


def fallback_forecast(df: pd.DataFrame, horizon=72):
    y = df["y"].to_numpy()
    t = np.arange(len(y))
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    out = []
    base = len(y)
    for h in range(1, horizon + 1):
        seasonal = 0.05 * np.sin(2 * np.pi * ((base + h) % 24) / 24.0)
        pred = np.clip(y[-1] + slope * h + seasonal, 0.02, 0.98)
        out.append(pred)
    idx = pd.date_range(df["ds"].iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.DataFrame({"ds": idx, "yhat": out, "yhat_lower": np.clip(np.array(out) - 0.04, 0.0, 1.0), "yhat_upper": np.clip(np.array(out) + 0.04, 0.0, 1.0)}), "fallback"


def main():
    df = build_series()
    horizon = 72

    mode = "prophet"
    try:
        from prophet import Prophet  # type: ignore

        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon, freq="h")
        fc = model.predict(future).tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        fc["yhat"] = fc["yhat"].clip(0.0, 1.0)
        fc["yhat_lower"] = fc["yhat_lower"].clip(0.0, 1.0)
        fc["yhat_upper"] = fc["yhat_upper"].clip(0.0, 1.0)
    except Exception:
        fc, mode = fallback_forecast(df, horizon=horizon)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "prophet_forecast_72h.csv"
    fc.to_csv(csv_path, index=False)

    with (out_dir / "prophet_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": mode, "horizon": horizon, "artifact": str(csv_path)}, f, indent=2)

    print(f"saved={csv_path}")
    print(f"mode={mode}")


if __name__ == "__main__":
    main()
