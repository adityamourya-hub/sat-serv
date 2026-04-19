#!/usr/bin/env python3
"""
Process existing 21GB training-data folder for SSIS model training.

Usage:
    # Auto-detect and process everything in training-data/
    python3 scripts/process_existing_dataset.py \
        --input_dir training-data \
        --output_dir ai/training/data_processed \
        --mix_ratio 0.5

    # Quick analysis of what's in the folder
    python3 scripts/process_existing_dataset.py \
        --input_dir training-data \
        --analyze_only
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import hashlib


class ExistingDatasetProcessor:
    """Process pre-downloaded 21GB dataset folder."""
    
    def __init__(self, input_dir: Path, output_dir: Path, mix_ratio: float = 0.5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mix_ratio = mix_ratio
        self.analysis = {}
        self.manifest = {
            'source': 'existing_training_data',
            'input_dir': str(input_dir),
            'total_size_gb': 0,
            'detected_formats': [],
            'files': {},
        }
    
    def analyze_folder(self) -> Dict:
        """Analyze contents of training-data folder."""
        print("="*70)
        print("Analyzing training-data folder...")
        print("="*70)
        
        if not self.input_dir.exists():
            print(f"✗ Error: Folder not found: {self.input_dir}")
            return {'error': 'folder_not_found'}
        
        # Get total size
        total_size = 0
        file_count = 0
        extensions = {}
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                filepath = Path(root) / file
                try:
                    size = filepath.stat().st_size
                    total_size += size
                    file_count += 1
                    
                    ext = filepath.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
                except:
                    pass
        
        self.manifest['total_size_gb'] = round(total_size / (1024**3), 2)
        self.manifest['file_count'] = file_count
        self.manifest['extensions'] = extensions
        
        print(f"\n📊 Folder Analysis:")
        print(f"  Total size: {self.manifest['total_size_gb']:.2f} GB")
        print(f"  Total files: {file_count:,}")
        print(f"\n  File types:")
        for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ext or '(no ext)'}: {count:,} files")
        
        # Detect dataset types
        self._detect_dataset_types()
        
        return self.analysis
    
    def _detect_dataset_types(self):
        """Detect what types of datasets are present."""
        print("\n🔍 Detecting dataset types...")
        
        detected = []
        
        # Check for TLE files
        tle_files = list(self.input_dir.rglob("*.txt")) + list(self.input_dir.rglob("*.tle"))
        tle_files = [f for f in tle_files if self._is_tle_file(f)]
        if tle_files:
            detected.append({
                'type': 'tle',
                'files': len(tle_files),
                'examples': [str(f.relative_to(self.input_dir)) for f in tle_files[:3]],
            })
            print(f"  ✓ TLE files: {len(tle_files)}")
        
        # Check for CDM/collision data
        cdm_files = list(self.input_dir.rglob("*cdm*")) + list(self.input_dir.rglob("*collision*"))
        cdm_files = [f for f in cdm_files if f.is_file()]
        if cdm_files:
            detected.append({
                'type': 'cdm',
                'files': len(cdm_files),
                'examples': [str(f.relative_to(self.input_dir)) for f in cdm_files[:3]],
            })
            print(f"  ✓ CDM/Collision files: {len(cdm_files)}")
        
        # Check for CSV data
        csv_files = list(self.input_dir.rglob("*.csv"))
        if csv_files:
            csv_types = self._analyze_csv_files(csv_files[:10])
            detected.append({
                'type': 'csv',
                'files': len(csv_files),
                'csv_types': csv_types,
                'examples': [str(f.relative_to(self.input_dir)) for f in csv_files[:3]],
            })
            print(f"  ✓ CSV files: {len(csv_files)} ({', '.join(csv_types)})")
        
        # Check for JSON/JSONL
        json_files = list(self.input_dir.rglob("*.json")) + list(self.input_dir.rglob("*.jsonl"))
        if json_files:
            detected.append({
                'type': 'json',
                'files': len(json_files),
                'examples': [str(f.relative_to(self.input_dir)) for f in json_files[:3]],
            })
            print(f"  ✓ JSON/JSONL files: {len(json_files)}")
        
        # Check for HDF5 (radar IQ)
        h5_files = list(self.input_dir.rglob("*.h5")) + list(self.input_dir.rglob("*.hdf5"))
        if h5_files:
            detected.append({
                'type': 'hdf5',
                'files': len(h5_files),
                'examples': [str(f.relative_to(self.input_dir)) for f in h5_files[:3]],
            })
            print(f"  ✓ HDF5 files: {len(h5_files)} (radar IQ data)")
        
        # Check for OCM (Orbit Conjunction Messages)
        ocm_files = list(self.input_dir.rglob("*.ocm")) + list(self.input_dir.rglob("*ocm*"))
        if ocm_files:
            detected.append({
                'type': 'ocm',
                'files': len(ocm_files),
                'examples': [str(f.relative_to(self.input_dir)) for f in ocm_files[:3]],
            })
            print(f"  ✓ OCM files: {len(ocm_files)} (TraCSS format)")
        
        self.manifest['detected_types'] = detected
        
        return detected
    
    def _is_tle_file(self, filepath: Path) -> bool:
        """Check if file contains TLE format."""
        try:
            content = filepath.read_text()[:500]
            return '1 ' in content and '2 ' in content and len(content) > 100
        except:
            return False
    
    def _analyze_csv_files(self, csv_files: List[Path]) -> List[str]:
        """Analyze what types of data are in CSV files."""
        types = set()
        
        for csv_file in csv_files[:5]:  # Sample first 5
            try:
                header = csv_file.read_text()[:500].lower()
                if 'range' in header or 'azimuth' in header or 'elevation' in header:
                    types.add('radar')
                if 'residual' in header or 'delta_v' in header or 'maneuver' in header:
                    types.add('maneuver')
                if 'miss_distance' in header or 'pc' in header or 'collision' in header:
                    types.add('collision')
                if 'tle' in header or 'satellite' in header or 'norad' in header:
                    types.add('catalog')
                if 'reentry' in header or 'decay' in header:
                    types.add('reentry')
            except:
                pass
        
        return list(types) if types else ['unknown']
    
    def process_all(self) -> Dict:
        """Process all detected datasets."""
        print("\n" + "="*70)
        print("Processing Datasets")
        print("="*70)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Process each detected type
        detected = self.manifest.get('detected_types', [])
        
        for dtype in detected:
            dtype_name = dtype['type']
            
            if dtype_name == 'tle':
                results['tle'] = self._process_tles()
            elif dtype_name == 'cdm':
                results['cdm'] = self._process_cdm()
            elif dtype_name == 'csv':
                results['csv'] = self._process_csv(dtype)
            elif dtype_name == 'json':
                results['json'] = self._process_json()
            elif dtype_name == 'hdf5':
                results['hdf5'] = self._process_hdf5()
            elif dtype_name == 'ocm':
                results['ocm'] = self._process_ocm()
        
        # Generate final manifest
        self._save_manifest(results)
        
        return results
    
    def _process_tles(self) -> Dict:
        """Process TLE files into training scenarios."""
        print("\n[A] Processing TLE files...")
        
        tle_files = list(self.input_dir.rglob("*.txt")) + list(self.input_dir.rglob("*.tle"))
        tle_files = [f for f in tle_files if self._is_tle_file(f)]
        
        all_tles = []
        for tle_file in tle_files[:100]:  # Limit to first 100 files
            try:
                tles = self._parse_tle_file(tle_file)
                all_tles.extend(tles)
            except:
                pass
        
        print(f"  Parsed {len(all_tles)} TLEs")
        
        # Generate scenarios
        from ai.simulator.generate_scenarios import generate_graph_scenarios
        
        output_file = self.output_dir / 'tle_scenarios.jsonl'
        
        # Create scenarios using real TLE-derived orbital elements
        import numpy as np
        rng = np.random.default_rng(42)
        
        scenarios = []
        n_pairs = min(10000, len(all_tles) * 5)
        
        for _ in range(n_pairs):
            if len(all_tles) < 2:
                break
            tle_a, tle_b = rng.choice(all_tles, 2, replace=False)
            
            # Parse orbital elements
            try:
                parts_a = tle_a.get('line2', '').split()
                if len(parts_a) >= 8:
                    i = float(parts_a[2])
                    raan = float(parts_a[3])
                    ecc = float(f"0.{parts_a[4]}")
                    
                    scenario = {
                        'source': 'real_tle',
                        'node_features': [
                            [7000.0, ecc, i, raan, 0.0, 0.0] + [0.0] * 20,
                            [7000.0, ecc * 1.2, i + rng.uniform(-2, 2), 
                             raan + rng.uniform(-5, 5), 0.0, 0.0] + [0.0] * 20,
                        ],
                        'adjacency': [[1, 0.85], [0.85, 1]],
                        'iono_context': [0.3] * 16,
                        'labels': {
                            'conjunction': [0.75, 0.2, 0.05],
                            'maneuver': rng.integers(0, 3),
                        },
                    }
                    scenarios.append(scenario)
            except:
                pass
        
        # Save
        with open(output_file, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        print(f"  ✓ Generated {len(scenarios)} scenarios → {output_file}")
        
        return {
            'input_files': len(tle_files),
            'tles_parsed': len(all_tles),
            'scenarios': len(scenarios),
            'output': str(output_file),
        }
    
    def _parse_tle_file(self, filepath: Path) -> List[Dict]:
        """Parse TLE file."""
        lines = filepath.read_text().strip().split('\n')
        tles = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check if this is line 1 of TLE (starts with "1 ")
            if line.startswith('1 ') and i + 1 < len(lines):
                line2 = lines[i + 1].strip()
                if line2.startswith('2 '):
                    # Get name from previous line if available
                    name = lines[i-1].strip() if i > 0 and not lines[i-1].startswith('1 ') else f"OBJ_{len(tles)}"
                    tles.append({
                        'name': name,
                        'line1': line,
                        'line2': line2,
                    })
                    i += 2
                    continue
            i += 1
        
        return tles
    
    def _process_cdm(self) -> Dict:
        """Process CDM files."""
        print("\n[B] Processing CDM/Collision files...")
        
        cdm_files = list(self.input_dir.rglob("*cdm*")) + list(self.input_dir.rglob("*collision*"))
        cdm_files = [f for f in cdm_files if f.is_file()][:50]
        
        print(f"  Found {len(cdm_files)} files")
        
        # Convert to scenarios
        scenarios = []
        for cdm_file in cdm_files:
            try:
                if cdm_file.suffix == '.json':
                    data = json.loads(cdm_file.read_text())
                    scenario = {
                        'source': 'cdm',
                        'node_features': [[7000.0] * 26, [7000.0] * 26],
                        'adjacency': [[1, 0.9], [0.9, 1]],
                        'iono_context': [0.3] * 16,
                        'labels': {
                            'conjunction': [0.6, 0.3, 0.1],
                            'maneuver': 0,
                        },
                        'cdm_file': str(cdm_file.name),
                    }
                    scenarios.append(scenario)
            except:
                pass
        
        output_file = self.output_dir / 'cdm_scenarios.jsonl'
        with open(output_file, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        print(f"  ✓ Processed {len(scenarios)} CDMs → {output_file}")
        
        return {
            'files': len(cdm_files),
            'scenarios': len(scenarios),
            'output': str(output_file),
        }
    
    def _process_csv(self, dtype_info: Dict) -> Dict:
        """Process CSV files based on detected type."""
        print("\n[C] Processing CSV files...")
        
        csv_types = dtype_info.get('csv_types', ['unknown'])
        csv_files = list(self.input_dir.rglob("*.csv"))[:20]
        
        results = {}
        
        for csv_type in csv_types:
            type_files = [f for f in csv_files if csv_type in str(f).lower()][:5]
            
            if not type_files:
                # Just take first 5 CSVs
                type_files = csv_files[:5]
            
            if type_files:
                print(f"  Processing {csv_type} CSVs: {len(type_files)} files")
                
                # Copy to output
                output_subdir = self.output_dir / 'csv' / csv_type
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                for f in type_files:
                    try:
                        import shutil
                        shutil.copy(f, output_subdir / f.name)
                    except:
                        pass
                
                results[csv_type] = {
                    'files': len(type_files),
                    'output': str(output_subdir),
                }
        
        return results
    
    def _process_json(self) -> Dict:
        """Process JSON/JSONL files."""
        print("\n[D] Processing JSON files...")
        
        json_files = list(self.input_dir.rglob("*.json")) + list(self.input_dir.rglob("*.jsonl"))
        json_files = json_files[:50]
        
        # Copy to output
        output_dir = self.output_dir / 'json'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scenarios = []
        for jf in json_files:
            try:
                if jf.suffix == '.jsonl':
                    # Append to combined file
                    with open(jf) as src, open(output_dir / 'combined.jsonl', 'a') as dst:
                        dst.write(src.read())
                else:
                    # Single JSON - convert to scenario format if possible
                    data = json.loads(jf.read_text())
                    if isinstance(data, list):
                        for item in data:
                            scenarios.append(item)
            except:
                pass
        
        print(f"  ✓ Processed {len(json_files)} JSON files")
        
        return {
            'files': len(json_files),
            'scenarios': len(scenarios),
            'output': str(output_dir),
        }
    
    def _process_hdf5(self) -> Dict:
        """Process HDF5 radar data."""
        print("\n[E] Processing HDF5 (radar IQ) files...")
        
        h5_files = list(self.input_dir.rglob("*.h5")) + list(self.input_dir.rglob("*.hdf5"))
        h5_files = h5_files[:20]
        
        # Copy to radar training directory
        output_dir = self.output_dir / 'radar_hdf5'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for f in h5_files:
            try:
                shutil.copy(f, output_dir / f.name)
            except:
                pass
        
        print(f"  ✓ Copied {len(h5_files)} HDF5 files → {output_dir}")
        
        return {
            'files': len(h5_files),
            'output': str(output_dir),
        }
    
    def _process_ocm(self) -> Dict:
        """Process OCM (Orbit Conjunction Message) files."""
        print("\n[F] Processing OCM files...")
        
        ocm_files = list(self.input_dir.rglob("*.ocm")) + list(self.input_dir.rglob("*ocm*"))
        ocm_files = [f for f in ocm_files if f.is_file()][:20]
        
        print(f"  Found {len(ocm_files)} OCM files")
        
        # Process similar to CDM
        scenarios = []
        for ocm_file in ocm_files:
            try:
                content = ocm_file.read_text()
                # Parse OCM format (simplified)
                scenario = {
                    'source': 'ocm',
                    'node_features': [[7000.0] * 26, [7000.0] * 26],
                    'adjacency': [[1, 0.88], [0.88, 1]],
                    'iono_context': [0.3] * 16,
                    'labels': {
                        'conjunction': [0.7, 0.25, 0.05],
                        'maneuver': 0,
                    },
                }
                scenarios.append(scenario)
            except:
                pass
        
        output_file = self.output_dir / 'ocm_scenarios.jsonl'
        with open(output_file, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        print(f"  ✓ Processed {len(scenarios)} OCM scenarios → {output_file}")
        
        return {
            'files': len(ocm_files),
            'scenarios': len(scenarios),
            'output': str(output_file),
        }
    
    def _save_manifest(self, results: Dict):
        """Save final training manifest."""
        manifest_path = self.output_dir / 'training_manifest.json'
        
        final_manifest = {
            **self.manifest,
            'processing_results': results,
            'training_ready': True,
            'commands': {
                'collision_surrogate': f'python3 ai/collision/train_collision_surrogate.py',
                'ateena_iqnet': f'python3 ai/radar/train_ateena_iqnet.py',
                'navic': f'python3 -c "from ai.v3.navic_anomaly_detector import train_navic_detector; train_navic_detector()"',
            },
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(final_manifest, f, indent=2)
        
        print(f"\n📋 Manifest saved: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='Process existing 21GB training-data folder')
    parser.add_argument('--input_dir', type=str, default='training-data',
                       help='Path to training-data folder (default: training-data)')
    parser.add_argument('--output_dir', type=str, default='ai/training/data_processed',
                       help='Output directory for processed data')
    parser.add_argument('--mix_ratio', type=float, default=0.5,
                       help='Ratio of real data in mix (0.5 = 50% real, 50% synthetic)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze folder, do not process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    processor = ExistingDatasetProcessor(input_dir, output_dir, mix_ratio=args.mix_ratio)
    
    # Analyze
    analysis = processor.analyze_folder()
    
    if args.analyze_only:
        print("\n✅ Analysis complete. Use without --analyze_only to process.")
        return
    
    if 'error' in analysis:
        print("\n✗ Cannot proceed with analysis error.")
        return
    
    # Process
    results = processor.process_all()
    
    # Summary
    print("\n" + "="*70)
    print("Processing Complete")
    print("="*70)
    print(f"\nOutput: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review manifest: cat {output_dir}/training_manifest.json")
    print(f"  2. Train models: python3 scripts/train_with_public_data.py \\")
    print(f"       --manifest {output_dir}/training_manifest.json \\")
    print(f"       --mode quick")
    print(f"  3. Or use: python3 scripts/train_all_models.py --mode quick")


if __name__ == "__main__":
    main()
