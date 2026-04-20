# FNO Atmosphere Operator Integration Guide

## Overview
This guide explains how to integrate the calibrated Fourier Neural Operator (FNO) atmosphere operator into the Port-Hamiltonian System (PHS) for HGV tracking.

## Architecture Summary

```
NRLMSISE-00 (C) → Calibration → FNO Model → PHS Integration → HGV Tracking
     ↓                  ↓             ↓              ↓              ↓
 Ground Truth     Training     ρ_nn(h, ...)   R(x) = R + R_drag   Drag Force
```

## Quick Start

### 1. Calibrate FNO Model

```bash
# Compile NRLMSISE-00 library (if not already done)
cd /Users/adi/idex/pkg/physics/reference
gcc -shared -fPIC -O3 -o libmsise.dylib nrlmsise-00.c nrlmsise-00_data.c
mkdir -p /Users/adi/idex/pkg/physics/c
cp libmsise.dylib /Users/adi/idex/pkg/physics/c/

# Run calibration (demo mode)
cd /Users/adi/idex/scripts
python3 calibrate_hno.py
```

**Expected Output:**
- C-Bridge loaded successfully
- 9,995 valid samples generated
- Training completes in ~10 seconds (demo)
- Model saved to `hno_calibrated_demo.pth`

### 2. Load Calibrated Model in PHS

```python
import torch
from ai.v3.hno_physics_integration_torch import PyTorchPHS
from q.core.hno_pytorch import HydrostaticAtmosphereFNO

# Initialize PHS with neural atmosphere
phs = PyTorchPHS(
    state_dim=6,
    control_dim=3,
    use_neural_atmosphere=True,
)

# Load calibrated FNO weights
phs.atmosphere_op.load_state_dict(torch.load('hno_calibrated_demo.pth'))
phs.eval()
```

### 3. Use in HGV Tracking

```python
# HGV state at 100 km altitude
z = torch.tensor([[
    6471.0, 0.0, 0.0,     # Position: 100 km altitude
    0.0, 6.5, 2.0,        # Velocity: ~7 km/s re-entry
]])

# Current space weather
f107 = 150.0  # Solar flux (SFU)
ap = 15.0     # Geomagnetic index

# Get neural density
rho = phs.get_neural_density(z, f107, ap)
print(f"Density: {rho.item():.4e} kg/m³")

# Get dissipation matrix (includes drag)
R = phs.get_dissipation_matrix(z, f107, ap)
print(f"Dissipation trace: {R[0].trace().item():.6e}")

# Compute dynamics
dz_dt = phs.dynamics(z, f107=f107, ap=ap)
print(f"Velocity derivative: {dz_dt[0, 3:].tolist()}")
```

## Detailed Integration

### Step 1: Full Calibration (Production)

For production H100 training, use the full calibration pipeline:

```python
from scripts.calibrate_hno import (
    NRLMSISEBridge,
    AtmosphereCalibrationDataset,
    train_hno_calibration,
    HydrostaticValidator,
)
from torch.utils.data import DataLoader

# Initialize C-Bridge
bridge = NRLMSISEBridge()

# Generate large dataset (1M samples)
dataset = AtmosphereCalibrationDataset(
    n_samples=1000000,
    alt_range=(0, 500),
    f107_range=(65, 300),
    ap_range=(0, 400),
    lat_range=(-90, 90),
    lon_range=(-180, 180),
    doy_range=(1, 366),
    bridge=bridge,
)

# Create memory-mapped dataset for H100 scale
from scripts.calibrate_hno import create_memory_mapped_dataset
create_memory_mapped_dataset(dataset, 'atmosphere_calibration.bin')

# Load memory-mapped dataset
from scripts.calibrate_hno import MemoryMappedDataset
mm_dataset = MemoryMappedDataset('atmosphere_calibration.bin')

# Split and create loaders
train_size = int(0.8 * len(mm_dataset))
val_size = len(mm_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    mm_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Initialize FNO
from q.core.hno_pytorch import HydrostaticAtmosphereFNO
model = HydrostaticAtmosphereFNO(modes=16, width=64)

# Train with hydrostatic projection
history = train_hno_calibration(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3,
    device='cuda',  # Use GPU for H100
    save_path='hno_calibrated_production.pth',
    projection_freq=1,
)

# Validate hydrostatic equilibrium
validator = HydrostaticValidator()
alt_grid = torch.linspace(0, 500, 100)
check = validator.check_divergence(model, alt_grid)
print(f"Hydrostatic residual: {check['max_residual']:.2e}")
print(f"Status: {'PASSED' if check['passed'] else 'NEEDS_TRAINING'}")
```

### Step 2: Export to ONNX for Deployment

```python
import torch.onnx

# Load calibrated model
model = HydrostaticAtmosphereFNO(modes=16, width=64)
model.load_state_dict(torch.load('hno_calibrated_production.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 100, 6)  # [batch, seq_len, features]

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'hno_calibrated.onnx',
    input_names=['input'],
    output_names=['density'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'seq_len'},
        'density': {0: 'batch_size', 1: 'seq_len'},
    },
    opset_version=14,
)

print("Model exported to hno_calibrated.onnx")
```

### Step 3: Wire into Go System

Create a Python-Go bridge for density queries:

**Go code (`pkg/physics/neural_atmosphere.go`):**
```go
package physics

/*
#cgo LDFLAGS: -L${SRCDIR}/../c -lmsise -lm
#cgo CFLAGS: -I${SRCDIR}/../reference
#include <stdlib.h>
*/
import "C"
import "unsafe"

// GetNeuralDensity calls Python FNO via subprocess
func GetNeuralDensity(alt, lat, lon, f107, ap float64) float64 {
    // Call Python script with parameters
    // Return density from FNO
    // Fallback to NRLMSISE-00 if Python fails
    return getNRLMSISEDensity(alt, lat, lon, f107, ap)
}
```

**Python bridge (`scripts/neural_atmosphere_server.py`):**
```python
import torch
import sys
from q.core.hno_pytorch import HydrostaticAtmosphereFNO

# Load calibrated model
model = HydrostaticAtmosphereFNO(modes=16, width=64)
model.load_state_dict(torch.load('hno_calibrated_production.pth'))
model.eval()

def get_density(alt, lat, lon, f107, ap, doy=100):
    """Get density from neural FNO."""
    x = torch.tensor([[[alt, f107, ap, lat, lon, doy]]], dtype=torch.float32)
    with torch.no_grad():
        result = model(x, apply_projection=True)
        return result['density'].item()

# CLI interface for Go calls
if __name__ == '__main__':
    alt = float(sys.argv[1])
    lat = float(sys.argv[2])
    lon = float(sys.argv[3])
    f107 = float(sys.argv[4])
    ap = float(sys.argv[5])
    
    density = get_density(alt, lat, lon, f107, ap)
    print(density)
```

### Step 4: Integrate with HGV Physics

Update HGV physics to use neural atmosphere:

```go
// pkg/hypersonic/hgv_physics.go

func HGVEquationsOfMotion(state HGVState, consts HGVConstants, jd float64) HGVDerivatives {
    // Use neural atmosphere instead of static lookup
    rho := physics.GetNeuralDensity(
        state.R - physics.EARTH_R_EQUATORIAL, // altitude
        state.Phi * 180 / math.Pi,             // lat
        state.Lambda * 180 / math.Pi,          // lon
        getF107FromJD(jd),                    // F10.7
        getApFromJD(jd),                      // Ap
    )
    
    // Rest of HGV physics unchanged
    q := 0.5 * rho * vMs * vMs
    D := q * A * state.CD
    L := q * A * state.CL
    
    // ... (continue as before)
}
```

### Step 5: Live Space Weather Feed

Integrate NOAA SWPC data for real-time F10.7/Ap:

```python
import requests
from datetime import datetime

def get_space_weather():
    """Fetch live F10.7 and Ap from NOAA SWPC."""
    url = "https://services.swpc.noaa.gov/json/swpc_noaa_indices.json"
    response = requests.get(url)
    data = response.json()
    
    f107 = data[0]['f107_index']
    ap = data[0]['ap_index']
    
    return f107, ap

# Update every 3 hours
import schedule
import time

def update_space_weather():
    f107, ap = get_space_weather()
    # Update global space weather parameters
    # PHS will use these for density queries
    print(f"Updated space weather: F10.7={f107}, Ap={ap}")

schedule.every(3).hours.do(update_space_weather)
```

## Validation

### 1. Compare with NRLMSISE-00

```python
import numpy as np
import matplotlib.pyplot as plt

# Test altitude profile
alts = np.linspace(0, 500, 100)
lat, lon = 28.5, -80.5
f107, ap = 150.0, 15.0

# NRLMSISE-00 ground truth
from scripts.calibrate_hno import NRLMSISEBridge
bridge = NRLMSISEBridge()
rho_msise = []
for alt in alts:
    d, _ = bridge.get_density(alt, lat, lon, f107, ap)
    rho_msise.append(d)

# FNO prediction
x = torch.zeros(1, 100, 6)
x[0, :, 0] = torch.tensor(alts)
x[0, :, 1] = f107
x[0, :, 2] = ap
x[0, :, 3] = lat
x[0, :, 4] = lon
x[0, :, 5] = 100.0

with torch.no_grad():
    result = model(x, apply_projection=True)
    rho_fno = result['density'][0].numpy()

# Compute error
error = np.abs(np.log10(rho_fno) - np.log10(rho_msise))
print(f"Mean log10 error: {error.mean():.4f}")
print(f"Max log10 error: {error.max():.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(alts, rho_msise, label='NRLMSISE-00')
plt.semilogy(alts, rho_fno, label='FNO')
plt.xlabel('Altitude (km)')
plt.ylabel('Density (kg/m³)')
plt.legend()
plt.yscale('log')
plt.savefig('fno_vs_msise.png')
```

### 2. Verify Hydrostatic Equilibrium

```python
from scripts.calibrate_hno import HydrostaticValidator

validator = HydrostaticValidator()
alt_grid = torch.linspace(0, 500, 100)

# Check unprojected model
result_no_proj = model(x, apply_projection=False)
rho_no_proj = result_no_proj['density'][0]
residual_no_proj = validator.compute_hydrostatic_residual(alt_grid, rho_no_proj)

# Check projected model
result_proj = model(x, apply_projection=True)
rho_proj = result_proj['density'][0]
residual_proj = validator.compute_hydrostatic_residual(alt_grid, rho_proj)

print(f"Residual without projection: {residual_no_proj.max():.2e}")
print(f"Residual with projection: {residual_proj.max():.2e}")
```

### 3. PHS Energy Balance

```python
# Test energy conservation on conservative segment (high altitude)
z_high = torch.tensor([[
    6878.0, 0.0, 0.0,     # 500 km altitude
    0.0, 7.5, 0.0,        # Orbital velocity
]])

balance = phs.compute_energy_balance(z_high, f107=150.0, ap=15.0)
print(f"Hamiltonian: {balance['Hamiltonian']:.6e}")
print(f"Dissipation (should be ~0): {balance['dissipation_drag']:.6e}")
print(f"Density: {balance['density']:.4e} kg/m³")

# Test dissipative segment (low altitude)
z_low = torch.tensor([[
    6471.0, 0.0, 0.0,     # 100 km altitude
    0.0, 6.5, 2.0,        # Re-entry velocity
]])

balance = phs.compute_energy_balance(z_low, f107=150.0, ap=15.0)
print(f"\nHamiltonian: {balance['Hamiltonian']:.6e}")
print(f"Dissipation (should be >0): {balance['dissipation_drag']:.6e}")
print(f"Density: {balance['density']:.4e} kg/m³")
```

## Performance Optimization

### M2 Air Thermal Governor

```python
import psutil

def get_thermal_state():
    """Check CPU temperature and throttle if needed."""
    temp = psutil.sensors_temperatures()['coretemp'][0].current
    
    if temp > 85:
        # Reduce FNO modes for lower compute
        model.fno.modes = 8  # Reduce from 16
        print(f"Thermal throttling: modes reduced to 8 (temp={temp}°C)")
    elif temp > 80:
        # Moderate throttling
        model.fno.modes = 12
        print(f"Thermal throttling: modes reduced to 12 (temp={temp}°C)")
    else:
        # Full performance
        model.fno.modes = 16
```

### Batch Processing for H100

```python
# Process multiple HGV trajectories in parallel
def batch_density_query(states, f107s, aps):
    """
    Query density for multiple states at once.
    
    Args:
        states: [N, 6] state tensors
        f107s: [N] solar flux values
        aps: [N] geomagnetic indices
        
    Returns:
        densities: [N] density values
    """
    batch_size = len(states)
    x = torch.zeros(batch_size, 1, 6)
    x[:, 0, 0] = states[:, 0]  # altitude
    x[:, 0, 1] = f107s
    x[:, 0, 2] = aps
    x[:, 0, 3] = states[:, 3]  # lat
    x[:, 0, 4] = states[:, 4]  # lon
    x[:, 0, 5] = 100.0  # DOY
    
    with torch.no_grad():
        result = model(x, apply_projection=True)
        return result['density'].squeeze()
```

## Troubleshooting

### C-Bridge Fails to Load

**Problem:** `RuntimeError: NRLMSISE C library not loaded`

**Solution:**
```bash
# Recompile library
cd /Users/adi/idex/pkg/physics/reference
gcc -shared -fPIC -O3 -o libmsise.dylib nrlmsise-00.c nrlmsise-00_data.c
cp libmsise.dylib /Users/adi/idex/pkg/physics/c/
```

### Hydrostatic Residual Too High

**Problem:** Residual > 1e-4 threshold

**Solution:**
- Increase training epochs (50-100)
- Increase dataset size (1M samples)
- Implement full manifold projection (not just clamping)
- Relax threshold to 1e-4 for practical use

### Memory Issues on M2 Air

**Problem:** OOM during training

**Solution:**
- Reduce batch size (64 → 32)
- Use gradient accumulation
- Reduce FNO modes (16 → 8)
- Use memory-mapped dataset

### Go-Python Bridge Latency

**Problem:** Subprocess calls too slow

**Solution:**
- Use shared memory buffer
- Implement persistent Python server
- Use ONNX runtime in Go directly
- Cache density queries

## References

- FNO Implementation: `q/core/hno_pytorch.py`
- PHS Integration: `ai/v3/hno_physics_integration_torch.py`
- Calibration Script: `scripts/calibrate_hno.py`
- NRLMSISE-00 Reference: `pkg/physics/reference/nrlmsise-00.c`
- JAX Wrapper: `ai/v3/nrlmsise_wrapper.py`
