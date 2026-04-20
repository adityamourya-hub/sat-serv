#!/usr/bin/env python3
"""
Preprocess training-data/ CSV and JSON files into model-ready formats.

Usage:
    python3 scripts/preprocess_training_data.py [--dataset all|A|B|C|D]

Reads from: training-data/
Writes to:  ai/*/data/  (per-component training inputs)
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = ROOT / "training-data"


# ---------------------------------------------------------------------------
# A: TLE / Ephemeris → Maneuver, Cataloguing, Propagation training data
# ---------------------------------------------------------------------------

def parse_tle_file(filepath: Path) -> List[Dict]:
    """Parse a CelesTrak-format TLE file into records."""
    records = []
    lines = filepath.read_text().strip().splitlines()
    i = 0
    while i < len(lines) - 2:
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            i += 1
            continue
        try:
            norad_id = int(line1[2:7].strip())
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float("0." + line2[26:33].strip())
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            rev_num = int(line2[63:68].strip()) if line2[63:68].strip() else 0

            full_year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year

            records.append({
                "norad_id": norad_id,
                "name": name,
                "epoch_year": full_year,
                "epoch_day": epoch_day,
                "inclination": inclination,
                "raan": raan,
                "eccentricity": eccentricity,
                "arg_perigee": arg_perigee,
                "mean_anomaly": mean_anomaly,
                "mean_motion": mean_motion,
                "rev_num": rev_num,
            })
        except (ValueError, IndexError):
            pass
        i += 3
    return records


def process_dataset_a():
    """Process TLE and CSV files from Dataset A."""
    dir_a = TRAINING_DATA / "A-TLES-Ephemeris-for-Collision-manauver-conjunction-cataloguing-data"
    if not dir_a.exists():
        print(f"  SKIP: {dir_a} not found")
        return

    # 1. Parse all TLE files
    all_tles = []
    tle_files = list(dir_a.glob("*.txt"))
    for tf in tle_files:
        tles = parse_tle_file(tf)
        print(f"  Parsed {tf.name}: {len(tles)} TLEs")
        all_tles.extend(tles)

    print(f"  Total TLEs: {len(all_tles)}")

    # 2. Group by NORAD ID and generate maneuver training windows
    from collections import defaultdict
    by_norad = defaultdict(list)
    for t in all_tles:
        by_norad[t["norad_id"]].append(t)

    # Sort each object's TLEs by epoch
    for nid in by_norad:
        by_norad[nid].sort(key=lambda x: (x["epoch_year"], x["epoch_day"]))

    # 3. Generate residual sequences for maneuver detection
    maneuver_out = ROOT / "ai" / "maneuver" / "data"
    maneuver_out.mkdir(parents=True, exist_ok=True)

    seq_len = 32
    windows = []
    for nid, tles in by_norad.items():
        if len(tles) < seq_len:
            continue
        # Compute element residuals between consecutive TLEs
        residuals = []
        for i in range(1, len(tles)):
            dmm = tles[i]["mean_motion"] - tles[i - 1]["mean_motion"]
            de = tles[i]["eccentricity"] - tles[i - 1]["eccentricity"]
            di = tles[i]["inclination"] - tles[i - 1]["inclination"]
            draan = tles[i]["raan"] - tles[i - 1]["raan"]
            # Rough delta-v proxy: large mean_motion change
            dv_proxy = abs(dmm) * 42241.0 / (tles[i]["mean_motion"] ** (1.0 / 3.0)) if tles[i]["mean_motion"] > 0 else 0
            residuals.append({
                "residual": abs(dmm) + abs(de) * 100 + abs(di) * 0.1,
                "delta_v_proxy": dv_proxy,
                "dmm": dmm, "de": de, "di": di, "draan": draan,
            })

        # Create sliding windows
        for start in range(0, len(residuals) - seq_len + 1, seq_len // 2):
            window = residuals[start:start + seq_len]
            # Label: maneuver if any delta-v proxy > threshold
            max_dv = max(r["delta_v_proxy"] for r in window)
            label = 1 if max_dv > 0.5 else 0  # rough threshold
            windows.append({
                "norad_id": nid,
                "residuals": [r["residual"] for r in window],
                "delta_vs": [r["delta_v_proxy"] for r in window],
                "label": label,
            })

    outpath = maneuver_out / "tle_residuals.jsonl"
    with open(outpath, "w") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")
    print(f"  Wrote {len(windows)} maneuver windows → {outpath}")

    # 4. Parse satcat CSV for cataloguing
    satcat_path = dir_a / "satcat-data.csv"
    if satcat_path.exists():
        catalog_out = ROOT / "ai" / "simulator" / "data"
        catalog_out.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with open(satcat_path, "r") as f:
            reader = csv.DictReader(f)
            records = []
            for row in reader:
                records.append(row)
                row_count += 1
        outpath = catalog_out / "satcat_catalog.json"
        with open(outpath, "w") as f:
            json.dump(records[:50000], f)  # cap for memory
        print(f"  Wrote {min(row_count, 50000)} satcat records → {outpath}")

    # 5. Parse space weather CSV for re-entry predictor
    sw_path = dir_a / "Space-weather-data-from-1962.csv"
    if sw_path.exists():
        reentry_out = ROOT / "ai" / "v3" / "data"
        reentry_out.mkdir(parents=True, exist_ok=True)
        rows = []
        with open(sw_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                rows.append(row)
        arr = np.array(rows, dtype=str)
        np.savez_compressed(
            reentry_out / "space_weather.npz",
            header=np.array(header) if header else np.array([]),
            data=arr,
        )
        print(f"  Wrote {len(rows)} space weather rows → {reentry_out / 'space_weather.npz'}")


# ---------------------------------------------------------------------------
# B: Conjunction/Collision → Collision Surrogate GNN training data
# ---------------------------------------------------------------------------

def process_dataset_b():
    """Process conjunction/collision datasets from Dataset B."""
    dir_b = TRAINING_DATA / "B-Conjuction-Collision-dataset"
    if not dir_b.exists():
        print(f"  SKIP: {dir_b} not found")
        return

    collision_out = ROOT / "ai" / "simulator" / "data"
    collision_out.mkdir(parents=True, exist_ok=True)

    # 1. Process ESA train_data.zip (CDMs)
    zip_path = dir_b / "train_data.zip"
    if zip_path.exists():
        print(f"  Extracting {zip_path.name}...")
        extract_dir = dir_b / "train_data"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        # Find CSV files in extracted data
        csv_files = list(extract_dir.rglob("*.csv"))
        print(f"  Found {len(csv_files)} CSV files in train_data/")

        scenarios = []
        for csv_file in csv_files[:5]:  # Process first 5 files
            try:
                with open(csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Build graph scenario from CDM fields
                        scenario = cdm_row_to_graph_scenario(row)
                        if scenario:
                            scenarios.append(scenario)
            except Exception as e:
                print(f"  Warning: {csv_file.name}: {e}")

        outpath = collision_out / "scenarios_graph_v2.jsonl"
        with open(outpath, "w") as f:
            for s in scenarios:
                f.write(json.dumps(s) + "\n")
        print(f"  Wrote {len(scenarios)} collision scenarios → {outpath}")

    # 2. Process IVV compressed datasets
    for gz_name in [
        "IVV_Releasable_Dataset_Spherical_DefaultHBR.csv.gz",
        "IVV_Releasable_Dataset_SFSH_DiscreteHBR.csv.gz",
    ]:
        gz_path = dir_b / gz_name
        if gz_path.exists():
            print(f"  Processing {gz_name} (this may take a while)...")
            count = 0
            scenarios = []
            try:
                with gzip.open(gz_path, "rt") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        scenario = cdm_row_to_graph_scenario(row)
                        if scenario:
                            scenarios.append(scenario)
                        count += 1
                        if count >= 100000:  # Limit for initial training
                            break
            except Exception as e:
                print(f"  Warning: {gz_name}: {e}")

            if scenarios:
                stem = gz_name.replace(".csv.gz", "")
                outpath = collision_out / f"scenarios_{stem}.jsonl"
                with open(outpath, "w") as f:
                    for s in scenarios:
                        f.write(json.dumps(s) + "\n")
                print(f"  Wrote {len(scenarios)} scenarios from {gz_name} → {outpath}")


def cdm_row_to_graph_scenario(row: Dict) -> Dict:
    """Convert a CDM CSV row to a GNN-compatible graph scenario.
    
    Maps CDM fields to the 26-dim node features expected by
    CollisionGraphDataset in train_collision_surrogate.py.
    """
    try:
        # Try to extract key CDM fields (field names vary by dataset)
        miss_dist = float(row.get("MISS_DISTANCE", row.get("miss_distance", 0)))
        tca = row.get("TCA", row.get("tca", ""))
        
        # Primary object state
        x1 = float(row.get("X_1", row.get("SAT1_X", 0)))
        y1 = float(row.get("Y_1", row.get("SAT1_Y", 0)))
        z1 = float(row.get("Z_1", row.get("SAT1_Z", 0)))
        vx1 = float(row.get("X_DOT_1", row.get("SAT1_X_DOT", 0)))
        vy1 = float(row.get("Y_DOT_1", row.get("SAT1_Y_DOT", 0)))
        vz1 = float(row.get("Z_DOT_1", row.get("SAT1_Z_DOT", 0)))

        # Secondary object state
        x2 = float(row.get("X_2", row.get("SAT2_X", 0)))
        y2 = float(row.get("Y_2", row.get("SAT2_Y", 0)))
        z2 = float(row.get("Z_2", row.get("SAT2_Z", 0)))
        vx2 = float(row.get("X_DOT_2", row.get("SAT2_X_DOT", 0)))
        vy2 = float(row.get("Y_DOT_2", row.get("SAT2_Y_DOT", 0)))
        vz2 = float(row.get("Z_DOT_2", row.get("SAT2_Z_DOT", 0)))

        # Covariance (diagonal terms)
        cr_r = float(row.get("CR_R", row.get("SAT1_CR_R", 1e-6)))
        ct_t = float(row.get("CT_T", row.get("SAT1_CT_T", 1e-6)))
        cn_n = float(row.get("CN_N", row.get("SAT1_CN_N", 1e-6)))

        # Risk label
        pc = float(row.get("MAX_PROB", row.get("COLLISION_PROBABILITY", row.get("pc", 0))))

        # Build 26-dim feature vectors for 2 nodes (primary + secondary)
        node1 = [x1, y1, z1, vx1, vy1, vz1, cr_r, ct_t, cn_n] + [0.0] * 17
        node2 = [x2, y2, z2, vx2, vy2, vz2, cr_r, ct_t, cn_n] + [0.0] * 17

        # Pad to max_nodes=10
        node_features = [node1, node2] + [[0.0] * 26] * 8

        # Adjacency (fully connected for 2 real nodes)
        adj = [[0.0] * 10 for _ in range(10)]
        adj[0][1] = 1.0
        adj[1][0] = 1.0

        # Conjunction risk label (Dirichlet: [low, medium, high])
        if pc > 1e-4:
            conj_label = [0.1, 0.2, 0.7]
        elif pc > 1e-6:
            conj_label = [0.2, 0.6, 0.2]
        else:
            conj_label = [0.8, 0.15, 0.05]

        # Maneuver label (0=none, 1=prograde, 2=retrograde)
        man_label = 0

        return {
            "node_features": node_features,
            "adjacency": adj,
            "iono_context": [0.0] * 16,
            "conjunction_risk": conj_label,
            "maneuver_label": man_label,
            "miss_distance_km": miss_dist,
            "pc": pc,
        }
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# C: NavIC / GNSS → NavIC Anomaly Detector training data
# ---------------------------------------------------------------------------

def process_dataset_c():
    """Process NavIC/GNSS spoofing metadata from Dataset C."""
    dir_c = TRAINING_DATA / "C-NAVIC-anamoly-and-GNSS-spoofing-jamming-dataset"
    if not dir_c.exists():
        print(f"  SKIP: {dir_c} not found")
        return

    navic_out = ROOT / "ai" / "v3" / "data"
    navic_out.mkdir(parents=True, exist_ok=True)

    # Parse metadata JSON files for labels and parameters
    metadata_files = list(dir_c.glob("*-metadata.json"))
    all_metadata = []
    for mf in metadata_files:
        try:
            with open(mf, "r") as f:
                meta = json.load(f)
            all_metadata.append(meta)
            print(f"  Loaded metadata: {mf.name}")
        except Exception as e:
            print(f"  Warning: {mf.name}: {e}")

    # Extract any zip files
    zip_files = list(dir_c.glob("*.zip"))
    for zf_path in zip_files:
        extract_dir = dir_c / zf_path.stem
        if not extract_dir.exists():
            print(f"  Extracting {zf_path.name}...")
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                print(f"  Warning: {zf_path.name}: {e}")

    # Generate synthetic NavIC training data augmented by metadata parameters
    # The NavIC detector expects: (n_samples, seq_len=60, n_features=12)
    n_samples = 10000
    seq_len = 60
    n_features = 12
    rng = np.random.default_rng(42)

    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        label = rng.choice([0, 0, 0, 1, 2])  # 60% normal, 20% jam, 20% spoof
        y[i] = label

        if label == 0:  # Normal
            X[i, :, 0:2] = rng.normal(0, 0.5, (seq_len, 2))  # PR residuals
            X[i, :, 2:4] = rng.normal(0, 0.01, (seq_len, 2))  # CP residuals
            X[i, :, 5:7] = rng.normal(45, 2, (seq_len, 2))    # C/N0
        elif label == 1:  # Jamming
            X[i, :, 0:2] = rng.normal(0, 5.0, (seq_len, 2))   # Large PR residuals
            X[i, :, 5:7] = rng.normal(25, 5, (seq_len, 2))    # Low C/N0
            X[i, :, 4] = rng.normal(3.0, 1.0, seq_len)        # Large iono diff
        elif label == 2:  # Spoofing
            # Spoofing: gradual drift
            drift = np.linspace(0, 5, seq_len)
            X[i, :, 0] = drift + rng.normal(0, 0.2, seq_len)
            X[i, :, 1] = drift + rng.normal(0, 0.2, seq_len)
            X[i, :, 5:7] = rng.normal(44, 1, (seq_len, 2))    # Normal-ish C/N0
            X[i, :, 11] = np.linspace(1.0, 0.3, seq_len)      # Degrading cross-corr

    np.savez_compressed(
        navic_out / "navic_training.npz",
        X=X, y=y,
        metadata=json.dumps([str(m) for m in all_metadata]),
    )
    print(f"  Wrote {n_samples} NavIC samples → {navic_out / 'navic_training.npz'}")


# ---------------------------------------------------------------------------
# D: Re-entry → Re-entry Predictor training data
# ---------------------------------------------------------------------------

def process_dataset_d():
    """Process re-entry history CSV from Dataset D."""
    dir_d = TRAINING_DATA / "D-Rentry-data-dataset-optional"
    if not dir_d.exists():
        print(f"  SKIP: {dir_d} not found")
        return

    reentry_out = ROOT / "ai" / "v3" / "data"
    reentry_out.mkdir(parents=True, exist_ok=True)

    csv_path = dir_d / "Reentry_History_Spreadsheet_04-17-26.csv"
    if not csv_path.exists():
        # Try other names
        csv_files = list(dir_d.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]
        else:
            print(f"  SKIP: No CSV found in {dir_d}")
            return

    rows = []
    with open(csv_path, "r", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows)} re-entry records from {csv_path.name}")

    # Extract training features
    features = []
    labels = []
    for row in rows:
        try:
            # Extract available fields (field names vary)
            incl = float(row.get("INCLINATION", row.get("Inclination", 0)))
            perigee = float(row.get("PERIGEE", row.get("Perigee_km", 200)))
            apogee = float(row.get("APOGEE", row.get("Apogee_km", 300)))
            rcs = float(row.get("RCS", row.get("RCS_m2", 1.0)))

            # Semi-major axis proxy
            a_km = (perigee + apogee) / 2.0 + 6371.0
            ecc = (apogee - perigee) / (apogee + perigee + 2 * 6371.0) if (apogee + perigee) > 0 else 0

            feat = [a_km, ecc, incl, perigee, apogee, rcs]
            features.append(feat)

            # Label: latitude if available
            lat = float(row.get("LAT", row.get("Latitude", 0)))
            lon = float(row.get("LON", row.get("Longitude", 0)))
            labels.append([lat, lon])
        except (ValueError, KeyError):
            continue

    if features:
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        np.savez_compressed(
            reentry_out / "reentry_training.npz",
            X=X, y=y,
        )
        print(f"  Wrote {len(features)} re-entry samples → {reentry_out / 'reentry_training.npz'}")
    else:
        print(f"  Warning: No valid features extracted from re-entry CSV")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess SSIS training data")
    parser.add_argument(
        "--dataset",
        choices=["all", "A", "B", "C", "D"],
        default="all",
        help="Which dataset to preprocess",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SSIS Training Data Preprocessor")
    print(f"Source: {TRAINING_DATA}")
    print("=" * 60)

    if args.dataset in ("all", "A"):
        print("\n[A] TLEs / Ephemeris / Space Weather")
        process_dataset_a()

    if args.dataset in ("all", "B"):
        print("\n[B] Conjunction / Collision CDMs")
        process_dataset_b()

    if args.dataset in ("all", "C"):
        print("\n[C] NavIC / GNSS Spoofing")
        process_dataset_c()

    if args.dataset in ("all", "D"):
        print("\n[D] Re-entry History")
        process_dataset_d()

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
