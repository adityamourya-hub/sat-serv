#!/usr/bin/env python3
"""
Simple SHAP-like attribution proxy for maneuver windows.
This is a lightweight fallback when full SHAP integration is unavailable in deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def explain_window(window: np.ndarray) -> dict:
    # window shape: [T, 2], columns: residual, delta_v
    residual_abs_mean = float(np.mean(np.abs(window[:, 0])))
    delta_v_mean = float(np.mean(window[:, 1]))
    denom = residual_abs_mean + abs(delta_v_mean)
    if denom < 1e-9:
        denom = 1.0

    return {
        "residual_mean_abs": {
            "value": residual_abs_mean,
            "contribution": residual_abs_mean / denom,
        },
        "delta_v_mean": {
            "value": delta_v_mean,
            "contribution": abs(delta_v_mean) / denom,
        },
    }


if __name__ == "__main__":
    sample = np.array(
        [
            [0.002, 0.003],
            [0.004, 0.004],
            [0.052, 0.040],
            [0.067, 0.060],
        ],
        dtype=np.float32,
    )
    out = explain_window(sample)

    out_path = Path(__file__).resolve().parent / "data" / "sample_shap_proxy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out_path)
