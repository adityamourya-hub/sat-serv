#!/usr/bin/env python3
"""
Dataset Ingestion Pipeline for SSIS.

Usage:
    # Detect and process all datasets in folders A, B, C, D
    python3 scripts/ingest_datasets.py --input_dir datasets/ --output_dir ai/training/data/

    # Process specific folders
    python3 scripts/ingest_datasets.py \
        --folders A B C D \
        --input_dir /path/to/datasets/ \
        --output_dir ai/training/data/

This script auto-detects dataset types and converts to SSIS training format.
Supported formats:
    - TLE (3-line format) → orbital scenarios
    - CDM (JSON/XML) → conjunction scenarios
    - CSV (RADAR/OPTICAL) → sensor observations
    - JSONL (SSIS native) → pass-through
    - HDF5 (IQ data) → radar training
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
import shutil


def detect_dataset_type(folder_path: Path) -> str:
    """
    Auto-detect dataset type from folder contents.
    Returns: 'tle', 'cdm', 'csv_sensor', 'jsonl', 'hdf5', 'unknown'
    """
    files = list(folder_path.iterdir())
    if not files:
        return 'empty'
    
    # Check file extensions
    extensions = {f.suffix.lower() for f in files if f.is_file()}
    
    # Sample first few files for content inspection
    sample_files = [f for f in files if f.is_file()][:5]
    
    # TLE detection: 3-line format
    for f in sample_files:
        if f.suffix in ['.txt', '.tle', '']:
            content = f.read_text()[:200]
            if content.startswith('1 ') and 'U' in content[:50]:
                return 'tle'
    
    # CDM detection: JSON/XML with collision data
    for f in sample_files:
        if f.suffix == '.json':
            try:
                data = json.loads(f.read_text())
                if 'MISS_DISTANCE' in str(data) or 'PC' in str(data) or 'collision' in str(data).lower():
                    return 'cdm'
            except:
                pass
    
    # JSONL detection
    if '.jsonl' in extensions or any(f.suffix == '.jsonl' for f in files):
        return 'jsonl'
    
    # HDF5 detection (radar data)
    if '.h5' in extensions or '.hdf5' in extensions:
        return 'hdf5'
    
    # CSV detection
    if '.csv' in extensions:
        content = sample_files[0].read_text()[:500] if sample_files else ""
        if 'range' in content.lower() or 'azimuth' in content.lower() or 'rcs' in content.lower():
            return 'csv_sensor'
        if 'residual' in content.lower() or 'delta_v' in content.lower():
            return 'csv_maneuver'
        return 'csv'
    
    return 'unknown'


class TLEIngestor:
    """Convert TLE files to SSIS orbital scenarios."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.scenarios = []
    
    def ingest(self, folder: Path) -> Dict:
        """Process all TLE files in folder."""
        print(f"  [TLE] Processing {folder.name}...")
        
        tle_files = list(folder.glob("*.txt")) + list(folder.glob("*.tle"))
        all_tles = []
        
        for tle_file in tle_files:
            tles = self._parse_tle_file(tle_file)
            all_tles.extend(tles)
        
        print(f"    Loaded {len(all_tles)} TLEs")
        
        # Generate conjunction scenarios from TLE pairs
        scenarios = self._generate_scenarios(all_tles)
        
        # Save
        output_file = self.output_dir / f"{folder.name}_scenarios.jsonl"
        with open(output_file, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        return {
            'type': 'tle',
            'input_files': len(tle_files),
            'tles': len(all_tles),
            'scenarios': len(scenarios),
            'output': str(output_file),
        }
    
    def _parse_tle_file(self, filepath: Path) -> List[Dict]:
        """Parse 3-line TLE format."""
        lines = filepath.read_text().strip().split('\n')
        tles = []
        
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                line0 = lines[i].strip() if not lines[i].startswith('1 ') else None
                line1 = lines[i] if line0 else lines[i]
                line2 = lines[i+1] if line0 else lines[i+1]
                
                if line1.startswith('1 ') and line2.startswith('2 '):
                    tles.append({
                        'name': line0 or f"OBJECT_{len(tles)}",
                        'line1': line1,
                        'line2': line2,
                    })
        
        return tles
    
    def _generate_scenarios(self, tles: List[Dict]) -> List[Dict]:
        """Generate synthetic conjunction scenarios from TLE pairs."""
        import numpy as np
        
        scenarios = []
        # Create pairs for training (sample subset for efficiency)
        n_pairs = min(1000, len(tles) * (len(tles) - 1) // 2)
        
        rng = np.random.default_rng(42)
        for _ in range(n_pairs):
            tle_a, tle_b = rng.choice(tles, 2, replace=False)
            
            # Extract orbital elements (simplified)
            try:
                i_a = float(tle_a['line2'][8:16])
                raan_a = float(tle_a['line2'][17:25])
                
                scenario = {
                    'source': 'tle_ingestion',
                    'node_features': [
                        # Simplified 26-dim features for both objects
                        [7000.0, 0.001, i_a, raan_a, 0.0, 0.0] + [0.0] * 20,
                        [7000.0, 0.001, i_a + 0.1, raan_a + 0.5, 0.0, 0.0] + [0.0] * 20,
                    ],
                    'adjacency': [[1, 0.8], [0.8, 1]],
                    'iono_context': [0.3] * 16,
                    'labels': {
                        'conjunction': [0.8, 0.1, 0.1],  # Safe, Warning, Critical
                        'maneuver': 0,
                    },
                    'source_tles': [tle_a['name'], tle_b['name']],
                }
                scenarios.append(scenario)
            except:
                pass
        
        return scenarios


class CDMIngestor:
    """Convert CDM files to SSIS training format."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def ingest(self, folder: Path) -> Dict:
        """Process CDM files."""
        print(f"  [CDM] Processing {folder.name}...")
        
        cdm_files = list(folder.glob("*.json")) + list(folder.glob("*.xml"))
        scenarios = []
        
        for cdm_file in cdm_files:
            try:
                scenario = self._parse_cdm(cdm_file)
                if scenario:
                    scenarios.append(scenario)
            except Exception as e:
                print(f"    Warning: Failed to parse {cdm_file}: {e}")
        
        # Save
        output_file = self.output_dir / f"{folder.name}_cdm.jsonl"
        with open(output_file, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        return {
            'type': 'cdm',
            'input_files': len(cdm_files),
            'scenarios': len(scenarios),
            'output': str(output_file),
        }
    
    def _parse_cdm(self, filepath: Path) -> Optional[Dict]:
        """Extract CDM data to SSIS format."""
        content = filepath.read_text()
        data = json.loads(content)
        
        # Extract relevant fields
        miss_distance = data.get('MISS_DISTANCE', {}).get('VALUE', 1000.0)
        pc = data.get('COLLISION_PROBABILITY', 0.0)
        
        # Convert to scenario format
        return {
            'source': 'cdm',
            'node_features': [[0.0] * 26 for _ in range(2)],  # Placeholder
            'adjacency': [[1, 0.9], [0.9, 1]],
            'iono_context': [0.3] * 16,
            'labels': {
                'conjunction': [1.0 - pc, pc * 0.5, pc * 0.5],
                'maneuver': 0,
            },
            'cdm_metadata': {
                'miss_distance_km': miss_distance,
                'pc': pc,
            },
        }


class CSVSensorIngestor:
    """Convert CSV sensor data to training format."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def ingest(self, folder: Path) -> Dict:
        """Process CSV sensor data."""
        print(f"  [CSV] Processing {folder.name}...")
        
        csv_files = list(folder.glob("*.csv"))
        all_data = []
        
        for csv_file in csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                print(f"    Warning: Failed to read {csv_file}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Detect sensor type
            columns = set(combined.columns.str.lower())
            if 'iq_real' in columns or 'iq_imag' in columns:
                sensor_type = 'radar_iq'
            elif 'range' in columns and 'rcs' in columns:
                sensor_type = 'radar_track'
            elif 'magnitude' in columns or 'photometric' in columns:
                sensor_type = 'optical'
            else:
                sensor_type = 'unknown'
            
            # Save processed
            output_file = self.output_dir / f"{folder.name}_{sensor_type}.csv"
            combined.to_csv(output_file, index=False)
            
            return {
                'type': 'csv_sensor',
                'sensor_type': sensor_type,
                'files': len(csv_files),
                'samples': len(combined),
                'output': str(output_file),
            }
        
        return {'type': 'csv_sensor', 'files': 0, 'samples': 0}


class DatasetManager:
    """Main orchestrator for dataset ingestion."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'scenarios').mkdir(exist_ok=True)
        (self.output_dir / 'cache').mkdir(exist_ok=True)
    
    def process_folders(self, folders: List[str]) -> Dict:
        """Process specified folders (A, B, C, D)."""
        results = {}
        
        for folder_name in folders:
            folder_path = self.input_dir / folder_name
            
            if not folder_path.exists():
                print(f"⚠ Folder not found: {folder_path}")
                results[folder_name] = {'error': 'not_found'}
                continue
            
            print(f"\n📁 Processing folder: {folder_name}")
            
            # Detect type
            dataset_type = detect_dataset_type(folder_path)
            print(f"   Detected type: {dataset_type}")
            
            # Route to appropriate ingestor
            if dataset_type == 'tle':
                ingestor = TLEIngestor(self.output_dir / 'scenarios')
                result = ingestor.ingest(folder_path)
            
            elif dataset_type == 'cdm':
                ingestor = CDMIngestor(self.output_dir / 'scenarios')
                result = ingestor.ingest(folder_path)
            
            elif dataset_type in ['csv_sensor', 'csv_maneuver', 'csv']:
                ingestor = CSVSensorIngestor(self.output_dir / 'processed')
                result = ingestor.ingest(folder_path)
            
            elif dataset_type == 'jsonl':
                # Pass-through with validation
                result = self._passthrough_jsonl(folder_path, folder_name)
            
            elif dataset_type == 'hdf5':
                result = self._passthrough_hdf5(folder_path, folder_name)
            
            else:
                result = {'type': 'unknown', 'error': f'Cannot process {dataset_type}'}
            
            results[folder_name] = result
        
        return results
    
    def _passthrough_jsonl(self, folder: Path, name: str) -> Dict:
        """Validate and copy JSONL files."""
        jsonl_files = list(folder.glob("*.jsonl"))
        
        total_records = 0
        for f in jsonl_files:
            with open(f) as fp:
                for line in fp:
                    try:
                        json.loads(line)
                        total_records += 1
                    except:
                        pass
        
        # Copy to output
        output_file = self.output_dir / 'scenarios' / f"{name}.jsonl"
        with open(output_file, 'w') as out:
            for f in jsonl_files:
                with open(f) as inp:
                    out.write(inp.read())
        
        return {
            'type': 'jsonl',
            'files': len(jsonl_files),
            'records': total_records,
            'output': str(output_file),
        }
    
    def _passthrough_hdf5(self, folder: Path, name: str) -> Dict:
        """Copy HDF5 radar data."""
        h5_files = list(folder.glob("*.h5")) + list(folder.glob("*.hdf5"))
        
        output_dir = self.output_dir / 'processed'
        for f in h5_files:
            shutil.copy(f, output_dir / f"{name}_{f.name}")
        
        return {
            'type': 'hdf5',
            'files': len(h5_files),
            'output': str(output_dir),
        }
    
    def generate_training_manifest(self, results: Dict) -> Path:
        """Generate manifest for training pipeline."""
        manifest = {
            'version': '1.0',
            'sources': results,
            'training_ready': True,
            'recommendations': self._generate_recommendations(results),
        }
        
        manifest_path = self.output_dir / 'training_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Suggest training strategies based on data."""
        recs = []
        
        has_tle = any(r.get('type') == 'tle' for r in results.values())
        has_cdm = any(r.get('type') == 'cdm' for r in results.values())
        has_sensor = any(r.get('type') == 'csv_sensor' for r in results.values())
        
        if has_tle:
            recs.append("Train collision surrogate with TLE-derived scenarios")
            recs.append("Train re-entry predictor with decaying orbits")
        
        if has_cdm:
            recs.append("Fine-tune collision GNN with real CDM data")
            recs.append("Validate Pc predictions against CDM benchmarks")
        
        if has_sensor:
            recs.append("Train ATeena IQNet with real radar tracks")
            recs.append("Calibrate RCS models for object characterization")
        
        return recs


def main():
    parser = argparse.ArgumentParser(description='Ingest datasets for SSIS training')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Root directory containing folders A, B, C, D')
    parser.add_argument('--output_dir', type=str, default='ai/training/data',
                       help='Output directory for processed data')
    parser.add_argument('--folders', nargs='+', default=['A', 'B', 'C', 'D'],
                       help='Folder names to process (default: A B C D)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("SSIS Dataset Ingestion Pipeline")
    print("="*60)
    print(f"Input:  {input_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Folders: {args.folders}")
    
    # Process
    manager = DatasetManager(input_dir, output_dir)
    results = manager.process_folders(args.folders)
    
    # Generate manifest
    manifest_path = manager.generate_training_manifest(results)
    
    # Summary
    print("\n" + "="*60)
    print("Ingestion Summary")
    print("="*60)
    
    total_scenarios = 0
    for folder, result in results.items():
        status = "✓" if 'error' not in result else "✗"
        print(f"{status} {folder}: {result.get('type', 'unknown')}")
        if 'scenarios' in result:
            print(f"    Scenarios: {result['scenarios']}")
            total_scenarios += result['scenarios']
        if 'samples' in result:
            print(f"    Samples: {result['samples']}")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    print(f"\nTotal scenarios: {total_scenarios}")
    print(f"Manifest: {manifest_path}")
    
    # Print recommendations
    recs = manager._generate_recommendations(results)
    if recs:
        print("\n📋 Training Recommendations:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
    
    print("\n✅ Ingestion complete. Ready for training.")
    print(f"\nNext steps:")
    print(f"  1. Review manifest: cat {manifest_path}")
    print(f"  2. Train models: python3 scripts/train_all_models.py --mode quick")
    print(f"  3. Or train specific: python3 scripts/train_with_manifest.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
