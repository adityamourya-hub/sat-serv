import sys
import os
from pathlib import Path

# Add simulator to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulator'))
from ateena_sim import create_training_dataset

if __name__ == "__main__":
    output_dir = Path("ai/radar/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ateena_mini_dataset_500.h5"
    print(f"Generating mini dataset: {output_file}")
    
    # 500 samples is manageable and allows for some learning
    create_training_dataset(
        str(output_file),
        n_samples=500,
        n_pulses=128,
        n_range_bins=1024,
        n_elements=64,
        seed=42
    )
    print("✓ Mini dataset generated successfully.")
