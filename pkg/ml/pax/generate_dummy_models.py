#!/usr/bin/env python3
"""
Generate synthetic ONNX files for PAX tier switching benchmark.

Since torch.onnx.export heavily optimizes models, we create synthetic
files of realistic sizes for I/O benchmarking.

Tier 1: 8MB FP32 (representing ~2MB INT8 quantized pruned model)
Tier 2: 50MB FP32 (representing full Lie-PHS model)

These are placeholder files - they won't be valid ONNX models but
will have realistic file sizes for measuring disk I/O and loading latency.
"""

import os
import struct


def generate_synthetic_onnx(output_path, target_size_mb):
    """
    Generate a synthetic file of target size with ONNX-like header.
    
    This creates a file that will take realistic time to load from disk,
    even though it's not a valid ONNX model. For the benchmark, we only
    care about I/O and loading time, not actual inference.
    """
    print(f"Generating synthetic ONNX file: {output_path} (target: {target_size_mb:.2f} MB)")
    
    # ONNX magic bytes (minimal header)
    onnx_header = b'\x08\x03\x12\x00'  # Minimal ONNX header
    
    with open(output_path, 'wb') as f:
        f.write(onnx_header)
        
        # Write padding data to reach target size
        target_bytes = int(target_size_mb * 1024 * 1024)
        current_bytes = len(onnx_header)
        
        # Write in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        while current_bytes < target_bytes:
            chunk = os.urandom(min(chunk_size, target_bytes - current_bytes))
            f.write(chunk)
            current_bytes += len(chunk)
    
    # Get actual file size
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Actual size: {actual_size_mb:.2f} MB")
    
    return actual_size_mb


def generate_tier1_model(output_path):
    """
    Generate Tier 1 synthetic file: ~8MB FP32.
    """
    return generate_synthetic_onnx(output_path, 8.0)


def generate_tier2_model(output_path):
    """
    Generate Tier 2 synthetic file: ~50MB FP32.
    """
    return generate_synthetic_onnx(output_path, 50.0)


def main():
    """Generate both synthetic ONNX files."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Tier 1 synthetic file
    tier1_path = os.path.join(output_dir, "tier1_pruned.onnx")
    tier1_size = generate_tier1_model(tier1_path)
    
    # Generate Tier 2 synthetic file
    tier2_path = os.path.join(output_dir, "tier2_full.onnx")
    tier2_size = generate_tier2_model(tier2_path)
    
    print("\n" + "="*60)
    print("Synthetic ONNX files generated successfully!")
    print("="*60)
    print(f"Tier 1 (synthetic):  {tier1_size:.2f} MB - {tier1_path}")
    print(f"Tier 2 (synthetic):  {tier2_size:.2f} MB - {tier2_path}")
    print("="*60)
    print("Note: These are synthetic files for I/O benchmarking only.")
    print("      They are not valid ONNX models but have realistic file sizes.")
    print("="*60)


if __name__ == "__main__":
    main()
