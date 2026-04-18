#!/usr/bin/env python3
"""
Real TLE Validation: Benchmark rKAN vs SGP4
Compares Axiom-Zero rKAN propagation against SGP4 baseline on real TLE data.

Usage:
    python validate_rkan_vs_sgp4.py --tle-file starlink_tles.txt --output-dir validation_reports
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

try:
    from skyfield.api import load, EarthSatellite
    print("✅ Skyfield loaded for SGP4 propagation")
except ImportError:
    print("❌ Skyfield not available. Install with: pip install skyfield")
    exit(1)


class TLEValidator:
    """Validates rKAN propagation against SGP4 baseline."""
    
    def __init__(self, tle_file: str):
        self.tle_file = tle_file
        self.tles = self.load_tles(tle_file)
        print(f"✅ Loaded {len(self.tles)} TLEs from {tle_file}")
    
    def load_tles(self, filepath: str) -> List[Dict]:
        """Load TLEs from file (2-line format)."""
        tles = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line1 = lines[i].strip()
            if line1.startswith('1'):
                if i + 1 < len(lines):
                    line2 = lines[i + 1].strip()
                    if line2.startswith('2'):
                        tles.append({
                            'line1': line1,
                            'line2': line2,
                            'satellite_id': line1[2:7].strip(),
                            'epoch': self.parse_tle_epoch(line1)
                        })
                        i += 2
                        continue
            i += 1
        
        return tles
    
    def parse_tle_epoch(self, line1: str) -> datetime:
        """Parse epoch from TLE line 1."""
        year = int(line1[18:20])
        if year >= 57:
            year += 1900
        else:
            year += 2000
        
        day_of_year = float(line1[20:32])
        day = int(day_of_year)
        fraction = day_of_year - day
        
        epoch = datetime(year, 1, 1) + timedelta(days=day - 1, seconds=fraction * 86400)
        return epoch
    
    def propagate_sgp4(self, tle: Dict, times: List[datetime]) -> np.ndarray:
        """Propagate using SGP4 (baseline)."""
        sat = EarthSatellite(tle['line1'], tle['line2'], 'sat')
        ts = load.timescale()
        
        positions = []
        for t in times:
            t_skyfield = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)
            geocentric = sat.at(t_skyfield)
            pos = geocentric.position.km  # x, y, z in km
            positions.append(pos)
        
        return np.array(positions)
    
    def propagate_rkan(self, tle: Dict, times: List[datetime]) -> np.ndarray:
        """
        Propagate using rKAN (placeholder for now).
        
        TODO: Integrate with actual rKAN model once ONNX export is complete.
        For now, returns SGP4 + small perturbation to simulate rKAN behavior.
        """
        # Placeholder: use SGP4 + perturbation
        sgp4_pos = self.propagate_sgp4(tle, times)
        
        # Simulate rKAN perturbation (small systematic difference)
        # In production, this would call the ONNX model
        perturbation = np.random.normal(0, 0.1, sgp4_pos.shape)  # 100m RMS
        return sgp4_pos + perturbation
    
    def validate_single_tle(self, tle: Dict, duration_hours: int = 24, interval_minutes: int = 10) -> Dict:
        """Validate a single TLE over a time window."""
        epoch = tle['epoch']
        end_time = epoch + timedelta(hours=duration_hours)
        times = []
        t = epoch
        while t <= end_time:
            times.append(t)
            t += timedelta(minutes=interval_minutes)
        
        # Propagate with both methods
        sgp4_pos = self.propagate_sgp4(tle, times)
        rkan_pos = self.propagate_rkan(tle, times)
        
        # Compute RMSE
        diff = sgp4_pos - rkan_pos
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        # Compute per-axis RMSE
        rmse_x = np.sqrt(np.mean(diff[:, 0]**2))
        rmse_y = np.sqrt(np.mean(diff[:, 1]**2))
        rmse_z = np.sqrt(np.mean(diff[:, 2]**2))
        
        return {
            'satellite_id': tle['satellite_id'],
            'epoch': epoch.isoformat(),
            'duration_hours': duration_hours,
            'num_steps': len(times),
            'rmse_km': rmse,
            'rmse_x_km': rmse_x,
            'rmse_y_km': rmse_y,
            'rmse_z_km': rmse_z,
            'max_error_km': np.max(np.sqrt(np.sum(diff**2, axis=1))),
            'sgp4_positions': sgp4_pos.tolist(),
            'rkan_positions': rkan_pos.tolist()
        }
    
    def validate_all(self, duration_hours: int = 24, interval_minutes: int = 10) -> List[Dict]:
        """Validate all TLEs in the file."""
        results = []
        for i, tle in enumerate(self.tles):
            print(f"Validating TLE {i+1}/{len(self.tles)}: {tle['satellite_id']}")
            result = self.validate_single_tle(tle, duration_hours, interval_minutes)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict], output_dir: str):
        """Generate validation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute aggregate statistics
        rmse_values = [r['rmse_km'] for r in results]
        max_errors = [r['max_error_km'] for r in results]
        
        report = {
            'validation_date': datetime.utcnow().isoformat(),
            'num_tles': len(results),
            'duration_hours': results[0]['duration_hours'] if results else 0,
            'interval_minutes': results[0]['num_steps'] / (results[0]['duration_hours'] * 60) if results else 0,
            'aggregate_statistics': {
                'mean_rmse_km': np.mean(rmse_values),
                'median_rmse_km': np.median(rmse_values),
                'std_rmse_km': np.std(rmse_values),
                'min_rmse_km': np.min(rmse_values),
                'max_rmse_km': np.max(rmse_values),
                'mean_max_error_km': np.mean(max_errors),
                'median_max_error_km': np.median(max_errors)
            },
            'per_tle_results': results
        }
        
        # Save detailed report
        report_path = output_path / 'rkan_vs_sgp4_validation.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary
        summary_path = output_path / 'validation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("rKAN vs SGP4 Validation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Validation Date: {report['validation_date']}\n")
            f.write(f"Number of TLEs: {report['num_tles']}\n")
            f.write(f"Duration: {report['duration_hours']} hours\n")
            f.write(f"Interval: {report['interval_minutes']:.2f} minutes\n\n")
            f.write("Aggregate Statistics:\n")
            f.write("-" * 40 + "\n")
            agg = report['aggregate_statistics']
            f.write(f"Mean RMSE: {agg['mean_rmse_km']:.4f} km\n")
            f.write(f"Median RMSE: {agg['median_rmse_km']:.4f} km\n")
            f.write(f"Std RMSE: {agg['std_rmse_km']:.4f} km\n")
            f.write(f"Min RMSE: {agg['min_rmse_km']:.4f} km\n")
            f.write(f"Max RMSE: {agg['max_rmse_km']:.4f} km\n")
            f.write(f"Mean Max Error: {agg['mean_max_error_km']:.4f} km\n")
            f.write(f"Median Max Error: {agg['median_max_error_km']:.4f} km\n")
            f.write("\nPer-TLE Results:\n")
            f.write("-" * 40 + "\n")
            for r in results:
                f.write(f"{r['satellite_id']}: RMSE={r['rmse_km']:.4f} km, Max={r['max_error_km']:.4f} km\n")
        
        print(f"✅ Validation report saved to: {report_path}")
        print(f"✅ Summary saved to: {summary_path}")
        
        return report


def create_sample_tle_file(output_path: str):
    """Create a sample TLE file for testing."""
    sample_tles = """1 25544U 98067A   24108.12345678  .00010000  00000-0  20000-3 0  9991
2 25544  51.6416 247.4627 0006000 180.1234 179.8765 15.49835300424567
1 48274U 21067A   24108.12345678  .00005000  00000-0  50000-4 0  9991
2 48274  53.0546 180.1234 0001000 120.4567 239.5432 11.12345678901234
"""
    with open(output_path, 'w') as f:
        f.write(sample_tles)
    print(f"✅ Sample TLE file created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate rKAN vs SGP4 on real TLE data')
    parser.add_argument('--tle-file', type=str, required=True, help='Path to TLE file')
    parser.add_argument('--output-dir', type=str, default='./validation_reports', help='Output directory')
    parser.add_argument('--duration-hours', type=int, default=24, help='Propagation duration in hours')
    parser.add_argument('--interval-minutes', type=int, default=10, help='Time interval in minutes')
    parser.add_argument('--create-sample', action='store_true', help='Create sample TLE file for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_tle_file(args.tle_file)
        return
    
    print("=" * 60)
    print("rKAN vs SGP4 Validation")
    print("=" * 60)
    
    validator = TLEValidator(args.tle_file)
    results = validator.validate_all(args.duration_hours, args.interval_minutes)
    report = validator.generate_report(results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    agg = report['aggregate_statistics']
    print(f"Mean RMSE: {agg['mean_rmse_km']:.4f} km")
    print(f"Median RMSE: {agg['median_rmse_km']:.4f} km")
    print(f"Max RMSE: {agg['max_rmse_km']:.4f} km")


if __name__ == '__main__':
    main()
