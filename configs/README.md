# SSIS Configuration Directory

Configuration files for the Space Situational Intelligence System (SSIS) / .

## File Overview

| File | Format | Purpose | Hot Reload |
|------|--------|---------|------------|
| `config.yaml` | YAML | Core operational parameters (engine, physics, security) | No |
| `config.prod.yaml` | YAML | Production template with GPU/CUDA settings | No |
| `model_registry.yaml` | YAML | Neural network manifest with ONNX paths | No |
| `pax_config.json` | JSON | PAX controller thresholds (thermal, tier switching) | Partial* |
| `sensors.yaml` | YAML | Sensor network topology (optical + radar) | No |
| `threat_library.yaml` | YAML | Sovereign HGV threat intelligence database | No |

*PAX hot reload: Manual via `controller.ReloadConfig()` only. Automatic background reloading not implemented.

## Configuration Details

### config.yaml
Core SSIS engine configuration:
- **Engine**: Batch size, worker count, provider (CPU/CUDA/ONEDNN)
- **Physics**: KL divergence threshold, propagation step, position sigma
- **Sensing**: Detection threshold, pulse count, sampling rate
- **Security**: mTLS, JWT expiration, audit level
- **Proximity**: Threat radius, FOV angles, reachability horizons

### model_registry.yaml
Neural network model manifest for the PAX tiered execution system:

**Existing Models (Ready for Inference):**
- `dphno_hamiltonian` (Tier 2) - `/ai/v3/axiom_se3_ghnn.onnx` ✓
- `dphno_hamiltonian_fp16` (Tier 1) - `/ai/v3/axiom_se3_ghnn_fp16.onnx` ✓

**Stub Implementations (ONNX files missing):**
- `maneuver_detector` - Returns "not yet bound" error
- `collision_risk` - Returns "not yet bound" error
- `rapid_ranker` - Returns "not yet bound" error
- `blindspot_forecaster` - Returns "not yet bound" error
- `icnn_hamiltonian` - Model not trained/exported

### pax_config.json
PAX (Progressive Acceleration eXecution) controller thresholds:

```json
{
  "thermal": {
    "upshift_threshold": 72.0,     // Max temp for T1→T2 up-shift (°C)
    "downshift_threshold": 80.0,    // Emergency T2→T1 down-shift (°C)
    "cooldown_duration_sec": 30.0   // Lock duration after thermal down-shift
  },
  "tier_switching": {
    "up_threshold": 0.70,      // R² to shift UP (T1→T2)
    "down_threshold": 0.85,    // R² to shift DOWN (T2→T1)
    "cooldown_frames": 10      // Frames to wait after gear shift
  }
}
```

**PAX Tier System:**
- Tier 0: Go-native physics (fastest, ballistic only)
- Tier 1: INT8-ONNX (thermal-efficient, pruned models)
- Tier 2: FP32-DPHNO (full accuracy, HGV capable)
- Tier 3: JAX-Discovery (imagination/what-if scenarios)

### sensors.yaml
Indian SSA sensor network:

| Sensor | Type | Location | Status | Range |
|--------|------|----------|--------|-------|
| `leh_optical` | Optical | Ladakh (34.15°N, 77.58°E) | **active** | 36,000 km |
| `mount_abu_optical` | Optical | Rajasthan (24.59°N, 72.72°E) | **active** | 36,000 km |
| `nellore_radar` | Radar | Andhra Pradesh (13.43°N, 79.92°E) | **active** | 2,000 km |
| `tawang_optical` | Optical | Arunachal (27.59°N, 91.86°E) | standby | 36,000 km |
| `sriharikota_radar` | Radar | Sriharikota (13.73°N, 80.20°E) | **maintenance** | 2,500 km |

**Coverage Gap**: Western sector (Gujarat) has no optical sensor.

### threat_library.yaml
Air-gapped sovereign intelligence database for HGV identification.

**Classification**: `SOVEREIGN_INTEL` / `AIR_GAPPED_ONLY`
**Authority**: IAF SSA Directorate

Contains aerodynamic signatures for:
- DF-ZF (China, Threat Level 9)
- Starry Sky-2 (China, Threat Level 8)
- Avangard (Russia, Threat Level 10)
- Generic tactical HGVs (Threat Level 7)
- Next-gen HGV-2026 (Threat Level 10, hypothetical)

## Known Issues

1. **Stub Models**: 5 models in registry have no ONNX files - only stub implementations exist
2. **Hot Reload**: PAX automatic background reloading not implemented; use manual `ReloadConfig()`
3. **Sensor Maintenance**: Sriharikota radar currently in maintenance mode

## Deployment Checklist

- [ ] Copy `config.prod.yaml` to `config.yaml` for production GPU deployment
- [ ] Verify ONNX files exist for models marked `stub_implementation: false`
- [ ] Confirm sensor network status matches operational needs
- [ ] Review threat library classification before air-gapped deployment
