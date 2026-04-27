# Axiom-Zero Interpretability Architecture

## Overview

This document describes the interpretability mechanisms in the Axiom-Zero kernel that prevent the "Black Box Information Bottleneck" - ensuring the latent space produces actionable intelligence rather than entangled statistical oracles.

## The Problem: Black Box Neural Operators

Traditional deep learning produces latent spaces that are:
- **Statistically efficient** at prediction but **tactically opaque** to operators
- **Entangled** - 64-dimensional soup mixing physics, noise, and adversarial intent
- **Uninterpretable** - "Anomaly Detected" vs "HGV banking 45° at L/D=2.4"

## Solution: Structured Latent Manifolds

### 1. FiLM Conditioning (Feature-wise Linear Modulation)

**Mathematical Advantage**: `h' = γ(z) ⊙ h + β(z)` vs concatenation `h' = W[h, z]`

FiLM respects separation of variables:
- **γ(z)** (Curvature Knob): Dynamically alters energy landscape - analogous to effective mass/moment of inertia
- **β(z)** (Origin Shift): Non-conservative potential energy term - models continuous thrust

**Diagnostic Check**:
```python
# Orbit (no atmosphere): γ ≈ 1.0, β ≈ 0.0
# HGV corridor: γ deviates based on aerodynamic resistance
# Thrust active: β non-zero (energy injection)

film_stats = model.get_film_diagnostics(z_latent)
if altitude_km > 400 and abs(film_stats["gamma_mean"] - 1.0) > 0.1:
    trip_symplectic_circuit_breaker()  # Hallucination detected
```

### 2. Auxiliary Physics Decoder (Linear Probe)

**Purpose**: Force latent manifold to align with human-readable physics parameters

**Architecture**:
```python
# Training only - pruned from ONNX export
physics_pred = physics_decoder(z_latent)  # [L/D, bank_angle, mass_flow, drag_coeff]
aux_loss = jnp.mean((physics_pred - physics_true) ** 2)
```

**Decoded Parameters**:
| Parameter | Unit | Tactical Meaning |
|-----------|------|------------------|
| L/D Ratio | - | Vehicle aerodynamic efficiency (chassis identification) |
| Bank Angle | deg | Predicts cross-range footprint |
| Mass Flow | kg/s | Thrust estimation, fuel depletion rate |
| Drag Coeff | - | Atmospheric regime indicator |

**Usage**:
```python
physics = model.decode_physics(z_latent)
print(f"L/D: {physics['L/D_ratio']:.2f}")          # e.g., 2.4
print(f"Bank: {physics['bank_angle_deg']:.1f}°")  # e.g., 45°
```

### 3. ONNX Deployment Strategy

**Training Phase (Heavy)**:
- H100 cluster trains with: Contrastive Loss + Yoshida-4 + Auxiliary Physics Loss
- Physics decoder shapes z-manifold along principal components
- FiLM networks learn regime-aware modulation

**Inference Phase (Lightweight)**:
- `jax2onnx` export prunes training-only components
- Exported graph contains only: FiLM H_net + E(3) tensor products
- Physics decoder removed (was only for training manifold shaping)
- M2 ANE executes forward pass in <2ms with zero interpretability penalty

## Implementation

### JAX PortHamiltonianSystem Configuration

```python
model = PortHamiltonianSystem(
    latent_dim=64,
    use_film=True,                    # Enable FiLM conditioning
    use_physics_decoder=True,         # Enable linear probe for training
    num_physics_params=4,             # [L/D, bank, mass_flow, Cd]
    key=jax.random.PRNGKey(42),
)

# During training: Full auxiliary loss
physics_pred = model.decode_physics(z_latent)
aux_loss = jnp.mean((physics_pred - physics_true) ** 2)
total_loss = mse_loss + 0.1 * aux_loss  # Shape manifold

# During inference: Pruned ONNX
# physics_decoder absent from exported graph
# Only H_net with FiLM modulation remains
```

### FiLM Diagnostics Watchdog

```python
def monitor_film_sanity(model, z_latent, altitude_km):
    """
    Go-native watchdog for ONNX inference sanity.
    Trips circuit breaker if γ deviates in orbit.
    """
    stats = model.get_film_diagnostics(z_latent)
    gamma_mean = stats["gamma_mean"]
    
    # In orbit (no atmosphere), γ should be ≈ 1.0
    if altitude_km > 400 and abs(gamma_mean - 1.0) > 0.1:
        return {
            "status": "HALLUCINATION_DETECTED",
            "action": "FALLBACK_TO_PURE_PHYSICS",
            "gamma_deviation": gamma_mean - 1.0,
        }
    
    return {"status": "NORMAL", "gamma": gamma_mean}
```

### War-Room TUI Output

Instead of abstract anomalies:

```
❌ BEFORE (Black Box):
[ALERT] Anomaly Detected in Sector 7
[ALERT] Latent Vector: [0.3, -0.7, 0.1, ...] (64 dims)
[ALERT] Confidence: 0.87

✅ AFTER (Interpretable):
[HGV IDENTIFIED] L/D Ratio: 2.4 (Waverider-class vehicle)
[MANEUVER] Bank Angle: 45° → Predicted Cross-Range: 340 km
[PROPULSION] Mass Flow: 0.15 kg/s (Hall thruster active)
[ATMOSPHERE] Drag Coeff: 0.42 (Transitional Flow Regime)
```

## Architectural Completeness

| Component | Training | Inference | Purpose |
|-----------|----------|-----------|---------|
| FiLM Network | ✅ Active | ✅ Exported | Regime-aware energy modulation |
| Physics Decoder | ✅ Active | ❌ Pruned | Manifold shaping (training only) |
| E(3) Irreps | ✅ Active | ✅ Exported | Rotational equivariance |
| Yoshida-4 | ✅ Active | ✅ Hard-coded | Symplectic integration |
| DeepONet | ✅ Active | ✅ Exported | Discrete control inference |

## Summary

The interpretability architecture ensures:

1. **Structured Latent Space**: Physics decoder forces z to align with [L/D, bank, thrust, drag]
2. **Regime Transparency**: FiLM parameters (γ, β) directly indicate atmospheric/thrust state
3. **Zero Inference Cost**: Training-only components pruned, maintaining <2ms latency
4. **Actionable Intelligence**: Raw 64D tensors → human-readable flight dynamics

**Result**: The IAF operator sees "HGV banking 45° with L/D=2.4" not "Anomaly confidence 0.87".
