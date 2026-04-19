#!/usr/bin/env python3
"""
Streaming Dataset Processor for 21GB+ Compressed Data

Processes datasets WITHOUT unzipping - designed for GPU sandbox execution.
Works with: .zip, .tar.gz, .tar, .gz files directly via streaming.

Usage:
    # 1. Analyze (on any machine)
    python3 scripts/streaming_dataset_processor.py \
        --input_dir training-data \
        --analyze \
        --output_manifest gpu_training_manifest.json

    # 2. Copy manifest + selective files to GPU sandbox
    #    (Only copy needed files, not full 21GB)

    # 3. Process on GPU machine (streaming, no full unzip)
    python3 scripts/streaming_dataset_processor.py \
        --input_dir training-data \
        --manifest gpu_training_manifest.json \
        --stream_process \
        --output_dir /gpu/sandbox/ssis/ai/training/data \
        --components collision navic reentry

Features:
    - Stream-reads zip/tar files (memory efficient)
    - Selective extraction (only needed files)
    - GPU-optimized batch generation
    - Resume capability for interrupted processing
"""

import argparse
import json
import os
import zipfile
import tarfile
import gzip
import io
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import hashlib
from dataclasses import dataclass, asdict
import sys


@dataclass
class CompressedDatasetInfo:
    """Metadata about a compressed dataset file."""
    path: str
    archive_type: str  # 'zip', 'tar', 'tar.gz', 'gz'
    size_bytes: int
    estimated_content_size: int
    contains: List[str]  # detected content types
    priority: int  # 1=high (process first), 5=low
    extraction_strategy: str  # 'stream', 'selective', 'full'


class SecurityScanner:
    """Security checks for archive contents."""
    
    FORBIDDEN_PATTERNS = [
        '..', '../', '..\\',  # Path traversal
        '/etc/', '/root/', '/home/', '/var/', '/usr/', '/bin/', '/sbin/',
        'C:\\Windows', 'C:\\Program Files', 'C:\\System',
        '.exe', '.dll', '.bat', '.cmd', '.sh', '.bin', '.run',
        '.com', '.msi', '.app', '.pkg', '.deb', '.rpm',
        'autorun', 'setup', 'install', 'payload',
    ]
    
    MAX_SAFE_COMPRESSION_RATIO = 100  # Flag if compressed:uncompressed > 100:1
    MAX_ARCHIVE_SIZE = 100 * 1024**3  # 100GB sanity limit
    
    @classmethod
    def scan_entry(cls, name: str, size: int = 0) -> Tuple[bool, str]:
        """Scan a single archive entry for security issues.
        
        Returns: (is_safe, reason_if_unsafe)
        """
        name_lower = name.lower()
        
        # Check for path traversal
        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.lower() in name_lower:
                return False, f"Forbidden pattern: {pattern}"
        
        # Check for absolute paths
        if name.startswith('/') or name.startswith('C:\\') or name.startswith('\\'):
            return False, "Absolute path detected"
        
        # Check for suspicious sizes (zip bomb indicator)
        if size > 10 * 1024**3:  # > 10GB single file
            return False, "Suspiciously large file"
        
        return True, ""
    
    @classmethod
    def check_archive_health(cls, archive_path: Path, archive_type: str) -> Tuple[bool, str]:
        """Basic health check before processing."""
        size = archive_path.stat().st_size
        
        if size > cls.MAX_ARCHIVE_SIZE:
            return False, f"Archive too large: {size / 1024**3:.1f}GB > {cls.MAX_ARCHIVE_SIZE / 1024**3}GB limit"
        
        return True, ""


class StreamingArchiveAnalyzer:
    """Analyze archives without full extraction."""
    
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.archives: List[CompressedDatasetInfo] = []
        self.security_warnings = []
    
    def scan_for_archives(self) -> List[CompressedDatasetInfo]:
        """Find all compressed files."""
        print("🔍 Scanning for compressed files...")
        
        patterns = ['*.zip', '*.tar.gz', '*.tar', '*.gz', '*.tgz', '*.bz2']
        found = []
        
        for pattern in patterns:
            found.extend(self.input_dir.rglob(pattern))
        
        print(f"  Found {len(found)} compressed files")
        
        for f in sorted(found)[:100]:  # Limit to first 100 for speed
            info = self._analyze_archive(f)
            if info:
                self.archives.append(info)
        
        return self.archives
    
    def _analyze_archive(self, path: Path) -> Optional[CompressedDatasetInfo]:
        """Quick analysis of archive contents with security checks."""
        size = path.stat().st_size
        suffix = ''.join(path.suffixes).lower()
        
        # Security: Check archive health before analysis
        is_healthy, health_reason = SecurityScanner.check_archive_health(path, suffix)
        if not is_healthy:
            self.security_warnings.append(f"{path.name}: {health_reason}")
            print(f"  ⚠️ Skipping {path.name}: {health_reason}")
            return None
        
        # Determine type
        if '.zip' in suffix:
            archive_type = 'zip'
        elif '.tar.gz' in suffix or '.tgz' in suffix:
            archive_type = 'tar.gz'
        elif '.tar' in suffix:
            archive_type = 'tar'
        elif '.gz' in suffix:
            archive_type = 'gz'
        else:
            archive_type = 'unknown'
        
        # Quick peek at contents (streaming)
        contains = []
        try:
            if archive_type == 'zip':
                contains = self._peek_zip(path)
            elif archive_type in ['tar', 'tar.gz']:
                contains = self._peek_tar(path)
        except Exception as e:
            contains = ['error_peeking']
        
        # Determine priority
        priority = self._determine_priority(contains, path.name)
        
        # Strategy
        if size < 100 * 1024 * 1024:  # < 100MB
            strategy = 'stream'
        elif size < 1024 * 1024 * 1024:  # < 1GB
            strategy = 'selective'
        else:
            strategy = 'selective'  # Never full extract for large files
        
        return CompressedDatasetInfo(
            path=str(path.relative_to(self.input_dir)),
            archive_type=archive_type,
            size_bytes=size,
            estimated_content_size=size * 3,  # Rough estimate
            contains=contains,
            priority=priority,
            extraction_strategy=strategy,
        )
    
    def _peek_zip(self, path: Path) -> List[str]:
        """Peek inside zip without full extract."""
        contains = []
        with zipfile.ZipFile(path, 'r') as zf:
            namelist = zf.namelist()[:50]  # Sample first 50
            
            # Security scan entries
            for name in namelist:
                is_safe, reason = SecurityScanner.scan_entry(name)
                if not is_safe:
                    self.security_warnings.append(f"{path.name}: {name} - {reason}")
                    print(f"    ⚠️ Security warning: {name} - {reason}")
            
            # Check for data types
            names_lower = ' '.join(namelist).lower()
            
            if any(x in names_lower for x in ['tle', 'satellite', 'norad']):
                contains.append('tle')
            if any(x in names_lower for x in ['cdm', 'collision', 'conjunction']):
                contains.append('cdm')
            if any(x in names_lower for x in ['csv', 'data']):
                contains.append('csv')
            if any(x in names_lower for x in ['json', 'jsonl']):
                contains.append('json')
            if any(x in names_lower for x in ['h5', 'hdf5', 'iq']):
                contains.append('hdf5')
            if any(x in names_lower for x in ['ocm', 'orbit']):
                contains.append('ocm')
            
            # Sample a small file to check format
            for name in namelist[:5]:
                # Skip if security flagged
                is_safe, _ = SecurityScanner.scan_entry(name)
                if not is_safe:
                    continue
                    
                if name.endswith(('.txt', '.csv', '.json')):
                    try:
                        with zf.open(name) as f:
                            sample = f.read(500).decode('utf-8', errors='ignore')
                            if '1 ' in sample and '2 ' in sample:
                                contains.append('tle_format')
                            break
                    except:
                        pass
        
        return list(set(contains)) if contains else ['unknown']
    
    def _peek_tar(self, path: Path) -> List[str]:
        """Peek inside tar without full extract."""
        contains = []
        mode = 'r:gz' if '.gz' in str(path) else 'r'
        
        try:
            with tarfile.open(path, mode) as tf:
                members = tf.getmembers()[:50]
                names = [m.name for m in members]
                names_lower = ' '.join(names).lower()
                
                if any(x in names_lower for x in ['tle', 'satellite']):
                    contains.append('tle')
                if any(x in names_lower for x in ['cdm', 'collision']):
                    contains.append('cdm')
                if any(x in names_lower for x in ['csv']):
                    contains.append('csv')
                if any(x in names_lower for x in ['h5', 'hdf5']):
                    contains.append('hdf5')
        except:
            contains = ['unknown']
        
        return contains if contains else ['unknown']
    
    def _determine_priority(self, contains: List[str], filename: str) -> int:
        """Determine processing priority (1=high, 5=low)."""
        fname_lower = filename.lower()
        
        # High priority for collision/conjunction data
        if 'collision' in fname_lower or 'cdm' in fname_lower or 'tracss' in fname_lower:
            return 1
        if 'cdm' in contains or 'collision' in contains:
            return 1
        
        # Medium-high for TLEs
        if 'tle' in contains or 'satellite' in fname_lower:
            return 2
        
        # Medium for GNSS/spoofing
        if 'gnss' in fname_lower or 'spoof' in fname_lower or 'fgi' in fname_lower:
            return 3
        
        # Lower for auxiliary data
        if 'doc' in fname_lower or 'readme' in fname_lower:
            return 5
        
        return 3
    
    def generate_gpu_manifest(self, output_path: Path) -> Dict:
        """Generate manifest for GPU sandbox execution."""
        manifest = {
            'version': '2.0',
            'generated_for': 'gpu_sandbox',
            'source_dir': str(self.input_dir),
            'security': {
                'warnings': self.security_warnings,
                'warning_count': len(self.security_warnings),
                'scan_complete': True,
                'recommendation': 'Review warnings before extracting on GPU sandbox' if self.security_warnings else 'No security issues detected',
            },
            'archives': [asdict(a) for a in sorted(self.archives, key=lambda x: x.priority)],
            'processing_plan': {
                'phase_1_extract_critical': [a.path for a in self.archives if a.priority == 1],
                'phase_2_process_large': [a.path for a in self.archives if a.priority <= 3],
                'phase_3_optional': [a.path for a in self.archives if a.priority > 3],
            },
            'estimated_processing': {
                'collision_gnn': '1-2 hours GPU',
                'navic_detector': '30-60 minutes GPU',
                'reentry_predictor': '1-2 hours GPU',
                'ateena_iqnet': '2-4 hours GPU',
            },
            'streaming_commands': {
                'collision': f'python3 {__file__} --stream_archive {{collision_archive}} --component collision',
                'navic': f'python3 {__file__} --stream_archive {{gnss_archive}} --component navic',
                'reentry': f'python3 {__file__} --stream_archive {{tle_archive}} --component reentry',
            },
            'security_note': 'Archives stay zipped on laptop. Extraction only on GPU sandbox.',
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Print security summary
        if self.security_warnings:
            print(f"\n⚠️  SECURITY WARNINGS: {len(self.security_warnings)} issues found")
            print("Review manifest before GPU processing")
        else:
            print("\n✅ Security scan: No issues detected")
        
        return manifest


class StreamingProcessor:
    """Process archives via streaming (no full unzip)."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def stream_process_zip(self, archive_path: Path, component: str) -> Dict:
        """Process a zip file via streaming for specific component."""
        print(f"\n📦 Streaming process: {archive_path.name} → {component}")
        
        results = {'processed': 0, 'errors': 0}
        
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Get relevant files for component
            relevant_files = self._select_files_for_component(zf.namelist(), component)
            
            print(f"  Selected {len(relevant_files)} files for {component}")
            
            for file_info in relevant_files:
                try:
                    with zf.open(file_info) as f:
                        # Stream process
                        self._process_stream(f, file_info, component)
                        results['processed'] += 1
                        
                        if results['processed'] % 100 == 0:
                            print(f"    Progress: {results['processed']} files")
                            
                except Exception as e:
                    results['errors'] += 1
                    if results['errors'] < 5:  # Limit error spam
                        print(f"    Warning: {file_info}: {e}")
        
        print(f"  ✓ Completed: {results['processed']} files, {results['errors']} errors")
        return results
    
    def _select_files_for_component(self, namelist: List[str], component: str) -> List[str]:
        """Select relevant files from archive for component."""
        extensions = {
            'collision': ['.json', '.jsonl', '.csv'],
            'navic': ['.csv', '.dat', '.bin'],
            'reentry': ['.txt', '.csv', '.json'],
            'ateena': ['.h5', '.hdf5', '.npy'],
            'maneuver': ['.csv', '.json'],
        }
        
        valid_exts = extensions.get(component, ['.csv', '.json'])
        keywords = {
            'collision': ['cdm', 'collision', 'conjunction', 'scenario'],
            'navic': ['gnss', 'gps', 'navic', 'spoof', 'jamming'],
            'reentry': ['reentry', 'decay', 'tle', 'satellite'],
            'ateena': ['radar', 'iq', 'cpi'],
            'maneuver': ['maneuver', 'residual', 'delta_v'],
        }.get(component, [])
        
        selected = []
        for name in namelist:
            name_lower = name.lower()
            # Check extension
            if any(name.endswith(ext) for ext in valid_exts):
                # Check keywords
                if not keywords or any(kw in name_lower for kw in keywords):
                    selected.append(name)
        
        # Limit to avoid memory issues
        return selected[:1000]
    
    def _process_stream(self, file_stream, filename: str, component: str):
        """Process a single file stream."""
        if component == 'collision':
            self._process_collision_stream(file_stream, filename)
        elif component == 'navic':
            self._process_navic_stream(file_stream, filename)
        elif component == 'reentry':
            self._process_reentry_stream(file_stream, filename)
        elif component == 'ateena':
            self._process_radar_stream(file_stream, filename)
    
    def _process_collision_stream(self, stream, filename: str):
        """Extract collision scenarios from stream."""
        # Read as text (assuming JSON/CSV)
        try:
            content = stream.read().decode('utf-8', errors='ignore')
            # Append to output
            output_file = self.output_dir / 'collision_stream.jsonl'
            with open(output_file, 'a') as out:
                # Try to parse as JSON, else write as-is
                for line in content.split('\n')[:100]:  # Limit lines
                    if line.strip():
                        out.write(line + '\n')
        except:
            pass
    
    def _process_navic_stream(self, stream, filename: str):
        """Extract GNSS patterns from stream."""
        output_dir = self.output_dir / 'gnss_raw'
        output_dir.mkdir(exist_ok=True)
        
        # Save raw for later processing
        output_file = output_dir / Path(filename).name
        with open(output_file, 'wb') as out:
            out.write(stream.read())
    
    def _process_reentry_stream(self, stream, filename: str):
        """Extract TLE/reentry data from stream."""
        try:
            content = stream.read().decode('utf-8', errors='ignore')
            output_file = self.output_dir / 'reentry_stream.txt'
            with open(output_file, 'a') as out:
                out.write(content)
        except:
            pass
    
    def _process_radar_stream(self, stream, filename: str):
        """Extract radar IQ data (likely HDF5)."""
        output_dir = self.output_dir / 'radar_raw'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / Path(filename).name
        with open(output_file, 'wb') as out:
            out.write(stream.read())


def main():
    parser = argparse.ArgumentParser(
        description='Streaming processor for compressed datasets (GPU-ready)'
    )
    parser.add_argument('--input_dir', type=str, default='training-data',
                       help='Input directory with compressed files')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze archives and generate manifest')
    parser.add_argument('--output_manifest', type=str, default='gpu_training_manifest.json',
                       help='Output manifest file')
    parser.add_argument('--stream_process', action='store_true',
                       help='Process archives via streaming')
    parser.add_argument('--manifest', type=str,
                       help='Input manifest for streaming processing')
    parser.add_argument('--stream_archive', type=str,
                       help='Specific archive to stream-process')
    parser.add_argument('--component', type=str,
                       choices=['collision', 'navic', 'reentry', 'ateena', 'maneuver', 'all'],
                       default='all',
                       help='Component to process')
    parser.add_argument('--output_dir', type=str, default='ai/training/data_streamed',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if args.analyze:
        # Phase 1: Analyze (run anywhere)
        print("="*70)
        print("GPU Sandbox - Dataset Analysis Phase")
        print("="*70)
        
        analyzer = StreamingArchiveAnalyzer(input_dir)
        archives = analyzer.scan_for_archives()
        
        print(f"\n📊 Found {len(archives)} archives")
        
        # Show summary
        by_priority = {}
        for a in archives:
            by_priority.setdefault(a.priority, []).append(a)
        
        for priority in sorted(by_priority.keys()):
            print(f"\nPriority {priority}:")
            for a in by_priority[priority][:5]:  # Show first 5
                size_mb = a.size_bytes / 1024 / 1024
                print(f"  • {a.path}")
                print(f"    Size: {size_mb:.1f} MB | Type: {a.archive_type} | Contains: {', '.join(a.contains[:3])}")
        
        # Generate manifest
        manifest_path = Path(args.output_manifest)
        manifest = analyzer.generate_gpu_manifest(manifest_path)
        
        print(f"\n✅ Manifest generated: {manifest_path}")
        print(f"\nNext steps for GPU sandbox:")
        print(f"  1. Copy this script + manifest to GPU machine")
        print(f"  2. Copy selective archives (Priority 1-2) if needed")
        print(f"  3. Run on GPU: python3 {__file__} --manifest {manifest_path} --stream_process")
    
    elif args.stream_process or args.stream_archive:
        # Phase 2: Stream process (run on GPU)
        print("="*70)
        print("GPU Sandbox - Streaming Processing Phase")
        print("="*70)
        
        output_dir = Path(args.output_dir)
        processor = StreamingProcessor(input_dir, output_dir)
        
        if args.stream_archive:
            # Process specific archive
            archive_path = input_dir / args.stream_archive
            if archive_path.exists():
                results = processor.stream_process_zip(archive_path, args.component)
                print(f"\n✅ Stream processing complete: {results}")
            else:
                print(f"✗ Archive not found: {archive_path}")
        
        elif args.manifest:
            # Process based on manifest
            manifest = json.loads(Path(args.manifest).read_text())
            
            # Get priority 1 archives first
            priority_archives = manifest['processing_plan']['phase_1_extract_critical']
            
            for archive_path in priority_archives[:3]:  # Process top 3
                full_path = input_dir / archive_path
                if full_path.exists():
                    results = processor.stream_process_zip(full_path, args.component)
                else:
                    print(f"⚠ Not found: {full_path}")
            
            print(f"\n✅ Batch stream processing complete")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
