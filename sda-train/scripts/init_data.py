"""
Training Data Initialization for 
Seeds synthetic training data using Go Truth Engine and Python Pixel Engine
Implements curriculum learning: Easy → Medium → Hard → Tactical
"""

import numpy as np
import time
from pathlib import Path
import json
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from render_sim import PixelRenderer, render_curriculum


class DataSeeder:
 """
 Manages synthetic data generation for training
 """
 
 def __init__(self, 
 output_dir: str = "data/synthetic",
 resolution: tuple = (1024, 1024)):
 """
 Initialize data seeder
 
 Args:
 output_dir: Directory to save synthetic frames
 resolution: Frame resolution (height, width)
 """
 self.output_dir = Path(output_dir)
 self.output_dir.mkdir(parents=True, exist_ok=True)
 
 self.resolution = resolution
 self.renderer = PixelRenderer(resolution=resolution)
 
 # Statistics tracking
 self.stats = {
 'total_frames': 0,
 'curriculum_levels': {0: 0, 1: 0, 2: 0, 3: 0},
 'start_time': time.time()
 }
 
 def generate_truth_states(self, 
 num_targets: int = 1,
 num_debris: int = 20,
 level: int = 1) -> List[Dict]:
 """
 Generate synthetic truth states (simulating Go engine output)
 
 In production, this would connect to Go via ZMQ/Plasma
 
 Args:
 num_targets: Number of primary targets
 num_debris: Number of debris objects
 level: Curriculum level (affects debris density)
 
 Returns:
 List of satellite state dictionaries
 """
 states = []
 
 # Adjust debris count by level
 if level == 0:
 num_debris = 0
 elif level == 1:
 num_debris = 20
 elif level == 2:
 num_debris = 100
 else: # level >= 3
 num_debris = 200
 
 # Generate primary target (adversary stalker)
 for i in range(num_targets):
 # Simulate LEO orbit at ~400km altitude
 r = 6371000 + 400000 # Earth radius + 400km in meters
 v = 7500 # Approximate orbital velocity (m/s)
 
 states.append({
 'sat_id': 99999 + i,
 'x': r,
 'y': 50000 + i * 1000,
 'z': 50000,
 'vx': 0,
 'vy': v,
 'vz': 0,
 'mag': 3.5 + np.random.randn() * 0.2 # Bright target
 })
 
 # Generate debris
 for i in range(num_debris):
 # Random position around target
 r = 6371000 + 400000 + np.random.randn() * 10000
 
 # Random velocity perturbation
 v = 7500 + np.random.randn() * 50
 
 # Dimmer for higher levels (harder to detect)
 if level <= 1:
 mag = 8.0 + np.random.rand() * 4.0
 elif level == 2:
 mag = 10.0 + np.random.rand() * 3.0
 else:
 mag = 11.0 + np.random.rand() * 3.0
 
 states.append({
 'sat_id': i + 1,
 'x': r,
 'y': 50000 + np.random.randn() * 50000,
 'z': 50000 + np.random.randn() * 50000,
 'vx': np.random.randn() * 10,
 'vy': v + np.random.randn() * 10,
 'vz': np.random.randn() * 10,
 'mag': mag
 })
 
 return states
 
 def seed_single_frame(self,
 frame_idx: int,
 level: int = 1,
 save_metadata: bool = True) -> Dict:
 """
 Generate and save a single training frame
 
 Args:
 frame_idx: Frame index
 level: Curriculum difficulty level
 save_metadata: Whether to save metadata JSON
 
 Returns:
 Dictionary with frame info
 """
 # Generate truth states
 truth = self.generate_truth_states(level=level)
 
 # Render dirty frame
 frame = self.renderer.render_curriculum_frame(truth, level)
 
 # Save frame
 frame_path = self.output_dir / f"frame_{frame_idx:06d}_level{level}.npy"
 np.save(frame_path, frame)
 
 # Save metadata
 metadata = {
 'frame_idx': frame_idx,
 'level': level,
 'resolution': self.resolution,
 'num_objects': len(truth),
 'timestamp': time.time(),
 'truth_states': truth
 }
 
 if save_metadata:
 meta_path = self.output_dir / f"frame_{frame_idx:06d}_level{level}.json"
 with open(meta_path, 'w') as f:
 json.dump(metadata, f, indent=2)
 
 return metadata
 
 def seed_curriculum_phase(self,
 level: int,
 num_frames: int,
 start_idx: int = 0) -> List[Dict]:
 """
 Generate a full curriculum phase
 
 Args:
 level: Curriculum level (0-3)
 num_frames: Number of frames to generate
 start_idx: Starting frame index
 
 Returns:
 List of metadata dictionaries
 """
 level_names = {
 0: "EASY (Clean, Bright, No Stars)",
 1: "MEDIUM (Star Fields + Noise)",
 2: "HARD (Dense Debris, Dim Sleepers)",
 3: "TACTICAL (Radiation Hits, Extreme)"
 }
 
 print(f"\n[CURRICULUM] Level {level}: {level_names.get(level, 'UNKNOWN')}")
 print(f" Generating {num_frames} frames...")
 
 metadata_list = []
 
 for i in range(num_frames):
 frame_idx = start_idx + i
 
 metadata = self.seed_single_frame(frame_idx, level=level)
 metadata_list.append(metadata)
 
 # Progress update
 if (i + 1) % 10 == 0:
 print(f" Progress: {i + 1}/{num_frames} frames")
 
 # Update stats
 self.stats['curriculum_levels'][level] += num_frames
 self.stats['total_frames'] += num_frames
 
 return metadata_list
 
 def seed_training_data(self,
 num_samples_per_level: int = 250,
 curriculum_ratio: List[float] = None) -> Dict:
 """
 Generate complete training dataset with all curriculum levels
 
 Args:
 num_samples_per_level: Base number of samples per level
 curriculum_ratio: Optional custom ratio for each level
 
 Returns:
 Statistics dictionary
 """
 print("=" * 60)
 print(" - Synthetic Data Generation")
 print("Siliguri Corridor Digital Twin")
 print("=" * 60)
 
 start_time = time.time()
 
 # Default equal distribution
 if curriculum_ratio is None:
 curriculum_ratio = [1.0, 1.0, 1.0, 1.0]
 
 all_metadata = []
 frame_idx = 0
 
 # Generate each curriculum level
 for level in range(4):
 num_frames = int(num_samples_per_level * curriculum_ratio[level])
 
 metadata = self.seed_curriculum_phase(level, num_frames, frame_idx)
 all_metadata.extend(metadata)
 
 frame_idx += num_frames
 
 # Save master metadata
 master_meta_path = self.output_dir / "master_metadata.json"
 with open(master_meta_path, 'w') as f:
 json.dump({
 'total_frames': self.stats['total_frames'],
 'curriculum_distribution': self.stats['curriculum_levels'],
 'generation_time': time.time() - start_time,
 'resolution': self.resolution,
 'frames': [
 {
 'frame_idx': m['frame_idx'],
 'level': m['level'],
 'num_objects': m['num_objects']
 }
 for m in all_metadata
 ]
 }, f, indent=2)
 
 # Print summary
 elapsed = time.time() - start_time
 print("\n" + "=" * 60)
 print("Generation Complete!")
 print("=" * 60)
 print(f"Total frames: {self.stats['total_frames']}")
 print(f"Curriculum distribution:")
 for level, count in self.stats['curriculum_levels'].items():
 print(f" Level {level}: {count} frames")
 print(f"Output directory: {self.output_dir.absolute()}")
 print(f"Total time: {elapsed:.1f}s ({self.stats['total_frames']/elapsed:.1f} fps)")
 
 return self.stats
 
 def validate_dataset(self) -> Dict:
 """
 Validate generated dataset
 
 Returns:
 Validation statistics
 """
 print("\n[VALIDATE] Checking dataset integrity...")
 
 validation_stats = {
 'valid_frames': 0,
 'invalid_frames': 0,
 'total_objects': 0,
 'issues': []
 }
 
 # Check all frame files
 frame_files = sorted(self.output_dir.glob("frame_*.npy"))
 
 for frame_file in frame_files:
 try:
 # Load frame
 frame = np.load(frame_file)
 
 # Check frame properties
 if frame.shape != self.resolution:
 validation_stats['issues'].append(
 f"{frame_file.name}: Wrong shape {frame.shape}"
)
 validation_stats['invalid_frames'] += 1
 continue
 
 if np.isnan(frame).any():
 validation_stats['issues'].append(
 f"{frame_file.name}: Contains NaN values"
)
 validation_stats['invalid_frames'] += 1
 continue
 
 # Check corresponding metadata
 meta_file = frame_file.with_suffix('.json')
 if not meta_file.exists():
 validation_stats['issues'].append(
 f"{frame_file.name}: Missing metadata"
)
 else:
 with open(meta_file) as f:
 meta = json.load(f)
 validation_stats['total_objects'] += meta.get('num_objects', 0)
 
 validation_stats['valid_frames'] += 1
 
 except Exception as e:
 validation_stats['issues'].append(
 f"{frame_file.name}: Error loading - {str(e)}"
)
 validation_stats['invalid_frames'] += 1
 
 # Summary
 print(f" Valid frames: {validation_stats['valid_frames']}")
 print(f" Invalid frames: {validation_stats['invalid_frames']}")
 print(f" Total objects: {validation_stats['total_objects']}")
 
 if validation_stats['issues']:
 print(f" Issues found: {len(validation_stats['issues'])}")
 for issue in validation_stats['issues'][:10]: # Show first 10
 print(f" - {issue}")
 
 return validation_stats


# Main entry point
def main():
 """Generate synthetic training data for """
 
 # Initialize seeder
 seeder = DataSeeder(
 output_dir="data/synthetic",
 resolution=(1024, 1024)
)
 
 # Generate dataset
 # Curriculum: 1000 frames total, 250 per level
 stats = seeder.seed_training_data(
 num_samples_per_level=250,
 curriculum_ratio=[1.0, 1.0, 1.0, 1.0]
)
 
 # Validate
 validation = seeder.validate_dataset()
 
 print("\n[COMPLETE] Synthetic dataset ready for training!")
 print(f" Run: python train_init.py to start training")
 
 return stats


if __name__ == "__main__":
 main()
