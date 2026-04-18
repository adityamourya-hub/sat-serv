#!/usr/bin/env python3
"""
Prepare English instruction-style dataset for SSA report generation fine-tuning.
Outputs:
- ai/report/data/report_sft_dataset.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report_sft_dataset.jsonl"

    samples = [
        {
            "instruction": "Write a concise English SSA alert summary with action recommendation.",
            "input": {
                "event_id": "EV-001",
                "event_type": "CONJUNCTION_ALERT",
                "severity": "high",
                "asset_id": 43286,
                "asset_name": "IRNSS-1I",
                "threat_id": 99999,
                "threat_name": "DEBRIS_UNKNOWN",
                "miss_distance_km": 0.028,
                "collision_probability": 2.4e-5,
                "tca_minutes": 11.2,
                "priority_score": 112,
            },
            "output": "SSA alert EV-001: asset IRNSS-1I (43286) is in high-severity conjunction with DEBRIS_UNKNOWN (99999). Miss distance is 0.028 km, collision probability is 2.40e-05, and TCA is 11.2 minutes. Recommended action: prepare evasive maneuver options and increase tracking cadence.",
        },
        {
            "instruction": "Write a concise English SSA alert summary with action recommendation.",
            "input": {
                "event_id": "EV-002",
                "event_type": "RAPID_THREAT_ALERT",
                "severity": "critical",
                "asset_id": 45034,
                "asset_name": "GSAT-30",
                "threat_id": 77777,
                "threat_name": "HYPERSONIC_TRACK",
                "miss_distance_km": 0.012,
                "collision_probability": 8.1e-4,
                "tca_minutes": 3.8,
                "priority_score": 170,
            },
            "output": "SSA alert EV-002: asset GSAT-30 (45034) faces a critical rapid threat from HYPERSONIC_TRACK (77777). Miss distance is 0.012 km, collision probability is 8.10e-04, and TCA is 3.8 minutes. Recommended action: initiate immediate collision-avoidance protocol and retask sensors continuously.",
        },
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
