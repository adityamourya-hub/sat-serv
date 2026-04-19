#!/usr/bin/env python3
"""
Download small-to-medium public datasets for SSIS training.

Sources:
  A. CelesTrak TLEs (no registration) → Collision, Maneuver, Cataloguing
  B. ESA Collision Challenge (~221 MB) → Collision Surrogate GNN
  C. FGI-JSDR / TEXBAT → NavIC/GNSS Anomaly Detection
  D. Aerospace Reentry → Re-entry Predictor

Usage:
    python3 scripts/download_public_datasets.py --all
    python3 scripts/download_public_datasets.py --source celestrak --categories stations active
    python3 scripts/download_public_datasets.py --source esa_collision
    python3 scripts/download_public_datasets.py --source fgi_gnss
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path
import json
from typing import List, Optional
import ssl
import sys

# Create SSL context that allows us to download from various sources
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class DatasetDownloader:
    """Manages downloading public SSA datasets."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'tle').mkdir(exist_ok=True)
        (self.output_dir / 'collision').mkdir(exist_ok=True)
        (self.output_dir / 'gnss').mkdir(exist_ok=True)
        (self.output_dir / 'reentry').mkdir(exist_ok=True)
        
        self.manifest = {
            'downloaded': [],
            'manual_steps': [],
        }
    
    def download_file(self, url: str, output_path: Path, desc: str) -> bool:
        """Download a file with progress."""
        print(f"  Downloading: {desc}")
        print(f"  From: {url}")
        print(f"  To: {output_path}")
        
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
                if block_num % 100 == 0:
                    sys.stdout.write(f"\r    Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
            print(f"\n  ✓ Saved ({output_path.stat().st_size / 1024:.1f} KB)")
            return True
            
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            return False
    
    def download_celestrak(self, categories: List[str]) -> dict:
        """
        Download TLEs from CelesTrak (no registration required).
        
        Categories: stations, active, starlink, visual, gps-ops, etc.
        Full list: https://www.celestrak.org/NORAD/elements/
        """
        print("\n[A] Downloading CelesTrak TLEs...")
        
        base_url = "https://www.celestrak.org/NORAD/elements/"
        results = {'success': [], 'failed': []}
        
        for category in categories:
            url = f"{base_url}{category}.txt"
            output_path = self.output_dir / 'tle' / f"{category}.txt"
            
            if self.download_file(url, output_path, f"CelesTrak: {category}"):
                results['success'].append(category)
                self.manifest['downloaded'].append({
                    'source': 'celestrak',
                    'category': category,
                    'path': str(output_path),
                    'format': 'tle',
                })
            else:
                results['failed'].append(category)
        
        # Also try the GP (General Perturbations) format
        print("\n  Downloading supplemental TLE categories...")
        supplemental = ['gpz', 'gpz-ops']
        for cat in supplemental:
            url = f"{base_url}{cat}.txt"
            output_path = self.output_dir / 'tle' / f"{cat}.txt"
            if self.download_file(url, output_path, f"CelesTrak: {cat}"):
                self.manifest['downloaded'].append({
                    'source': 'celestrak',
                    'category': cat,
                    'path': str(output_path),
                    'format': 'tle',
                })
        
        return results
    
    def download_esa_collision(self) -> dict:
        """
        Download ESA Collision Avoidance Challenge dataset.
        ~221 MB, contains ~162k CDMs for training collision surrogate.
        
        Source: https://kelvins.esa.int/collision-avoidance-challenge/data/
        """
        print("\n[B] Downloading ESA Collision Challenge dataset...")
        print("  Note: This is ~221 MB and may take a few minutes")
        
        # Zenodo mirror (more reliable)
        url = "https://zenodo.org/record/4463683/files/train_data.zip?download=1"
        zip_path = self.output_dir / 'collision' / 'esa_train_data.zip'
        
        result = {'success': False, 'path': None}
        
        if self.download_file(url, zip_path, "ESA train_data.zip"):
            print("  Extracting...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir / 'collision' / 'esa_train')
                print("  ✓ Extracted")
                
                self.manifest['downloaded'].append({
                    'source': 'esa_collision_challenge',
                    'path': str(self.output_dir / 'collision' / 'esa_train'),
                    'format': 'csv',
                    'size_mb': zip_path.stat().st_size / 1024 / 1024,
                })
                result['success'] = True
                result['path'] = str(self.output_dir / 'collision' / 'esa_train')
            except Exception as e:
                print(f"  ✗ Extraction failed: {e}")
        
        return result
    
    def download_fgi_gnss(self) -> dict:
        """
        FGI GNSS Jamming & Spoofing Dataset.
        Free download from Finnish Geospatial Research Institute.
        
        Source: https://etsin.fairdata.fi/dataset/367379a8-7d78-4b08-91f0-8027ce7a621b
        """
        print("\n[C] FGI GNSS Dataset...")
        print("  Note: Requires manual download from:")
        print("  https://etsin.fairdata.fi/dataset/367379a8-7d78-4b08-91f0-8027ce7a621b")
        print("  Click 'Access / Download' → Download all files")
        
        self.manifest['manual_steps'].append({
            'source': 'fgi_jsdr',
            'name': 'FGI-JSDR GNSS Jamming/Spoofing Dataset',
            'url': 'https://etsin.fairdata.fi/dataset/367379a8-7d78-4b08-91f0-8027ce7a621b',
            'instructions': 'Click Access/Download, download all files (~10 GB)',
            'target_dir': str(self.output_dir / 'gnss' / 'fgi'),
            'format': 'raw_iq',
        })
        
        # Create placeholder directory
        (self.output_dir / 'gnss' / 'fgi').mkdir(exist_ok=True)
        
        return {'manual': True, 'target': str(self.output_dir / 'gnss' / 'fgi')}
    
    def download_texbat(self) -> dict:
        """
        TEXBAT GPS Spoofing Dataset (University of Texas).
        Smaller dataset, good for initial testing.
        
        Source: https://radionavlab.ae.utexas.edu/texbat/
        """
        print("\n[C2] TEXBAT GPS Spoofing Dataset...")
        print("  Note: Requires email request or academic access")
        print("  Contact: radionavlab@austin.utexas.edu")
        print("  Alternative: Use Jammertest 2024 (Zenodo, open access)")
        
        # Jammertest 2024 as alternative (open access)
        print("\n  Downloading Jammertest 2024 (open alternative)...")
        url = "https://zenodo.org/records/15911359"
        
        self.manifest['manual_steps'].append({
            'source': 'texbat',
            'name': 'TEXBAT GPS Spoofing',
            'url': 'https://radionavlab.ae.utexas.edu/texbat/',
            'instructions': 'Email radionavlab@austin.utexas.edu for access',
            'alternative': {
                'name': 'Jammertest 2024',
                'url': 'https://zenodo.org/records/15911359',
                'access': 'Open (no registration)',
            },
            'target_dir': str(self.output_dir / 'gnss' / 'texbat'),
        })
        
        return {'manual': True}
    
    def download_reentry(self) -> dict:
        """
        Aerospace Corporation Reentry Database.
        CSV format, easy to use.
        
        Source: https://aerospace.org/reentries
        """
        print("\n[D] Aerospace Reentry Database...")
        print("  Note: Manual download from:")
        print("  https://aerospace.org/reentries")
        print("  Look for 'Download CSV' button")
        
        self.manifest['manual_steps'].append({
            'source': 'aerospace_reentry',
            'name': 'Aerospace Corp Reentry Database',
            'url': 'https://aerospace.org/reentries',
            'instructions': 'Navigate to reentries page, click Download CSV',
            'target_dir': str(self.output_dir / 'reentry'),
            'format': 'csv',
        })
        
        # Create placeholder
        (self.output_dir / 'reentry').mkdir(exist_ok=True)
        
        return {'manual': True}
    
    def download_tracss(self) -> dict:
        """
        TraCSS Dataset for Conjunction Assessment (NOAA, 2026).
        ~21 GB total, but has smaller CSV subsets.
        
        Source: https://space.commerce.gov/dataset-for-conjunction-assessment-verification/
        """
        print("\n[B2] TraCSS Conjunction Dataset...")
        print("  Note: Requires registration via Google Form")
        print("  URL: https://space.commerce.gov/dataset-for-conjunction-assessment-verification/")
        print("  Contact: Anna.Parikka@noaa.gov if issues")
        print("\n  Start with smaller CSV files (~500 MB) before full 21 GB OCM data")
        
        self.manifest['manual_steps'].append({
            'source': 'tracss',
            'name': 'TraCSS Conjunction Assessment',
            'url': 'https://space.commerce.gov/dataset-for-conjunction-assessment-verification/',
            'instructions': 'Fill Google Form on page, wait for approval',
            'size': '~21 GB (OCM) + smaller CSV subsets',
            'target_dir': str(self.output_dir / 'collision' / 'tracss'),
            'priority': 'high',  # Has answer key, excellent for validation
        })
        
        return {'manual': True}
    
    def save_manifest(self):
        """Save download manifest for training scripts."""
        manifest_path = self.output_dir / 'download_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"\n📋 Manifest saved: {manifest_path}")
        return manifest_path


def main():
    parser = argparse.ArgumentParser(description='Download public SSA datasets')
    parser.add_argument('--output_dir', type=str, default='ai/data/public',
                       help='Output directory for downloaded data')
    parser.add_argument('--all', action='store_true',
                       help='Download all auto-downloadable sources')
    parser.add_argument('--source', type=str,
                       choices=['celestrak', 'esa_collision', 'fgi_gnss', 'texbat', 
                               'reentry', 'tracss'],
                       help='Download specific source')
    parser.add_argument('--categories', nargs='+', 
                       default=['stations', 'active', 'starlink', 'visual'],
                       help='CelesTrak categories to download')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    downloader = DatasetDownloader(output_dir)
    
    print("="*70)
    print("SSIS Public Dataset Downloader")
    print("="*70)
    print(f"Output: {output_dir.absolute()}")
    print()
    print("Sources:")
    print("  [A] CelesTrak TLEs → Collision, Maneuver, Cataloguing")
    print("  [B] ESA Collision Challenge → GNN training (~221 MB)")
    print("  [C] FGI/TEXBAT → NavIC Anomaly Detection (manual)")
    print("  [D] Aerospace Reentry → Re-entry prediction (manual)")
    print()
    
    if args.all:
        # Download all auto-downloadable
        downloader.download_celestrak(args.categories)
        downloader.download_esa_collision()
        downloader.download_fgi_gnss()
        downloader.download_texbat()
        downloader.download_reentry()
        downloader.download_tracss()
        
    elif args.source == 'celestrak':
        downloader.download_celestrak(args.categories)
        
    elif args.source == 'esa_collision':
        downloader.download_esa_collision()
        
    elif args.source == 'fgi_gnss':
        downloader.download_fgi_gnss()
        
    elif args.source == 'texbat':
        downloader.download_texbat()
        
    elif args.source == 'reentry':
        downloader.download_reentry()
        
    elif args.source == 'tracss':
        downloader.download_tracss()
    
    else:
        # Default: download quick-start datasets
        print("Quick-start: Downloading CelesTrak + ESA datasets...")
        downloader.download_celestrak(['stations', 'active', 'visual'])
        downloader.download_esa_collision()
        print("\nFor more datasets, use --all or --source <name>")
    
    # Save manifest
    manifest_path = downloader.save_manifest()
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    
    auto = len(downloader.manifest['downloaded'])
    manual = len(downloader.manifest['manual_steps'])
    
    print(f"Auto-downloaded: {auto} files")
    for item in downloader.manifest['downloaded']:
        print(f"  ✓ {item['source']}: {item.get('category', 'dataset')}")
    
    print(f"\nManual download required: {manual} sources")
    for item in downloader.manifest['manual_steps']:
        print(f"  ⏳ {item['source']}: {item['name']}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review manifest:")
    print(f"   cat {manifest_path}")
    print("\n2. Complete manual downloads (if any)")
    print("\n3. Process all datasets:")
    print(f"   python3 scripts/ingest_public_datasets.py --manifest {manifest_path}")
    print("\n4. Train models:")
    print(f"   python3 scripts/train_with_public_data.py --manifest {manifest_path}")
    
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    main()
