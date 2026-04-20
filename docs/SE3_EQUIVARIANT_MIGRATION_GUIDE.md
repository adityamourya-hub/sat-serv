# SE(3)-Equivariant HNN Migration Guide

## Critical Physics Violation Identified

**Current Issue:** The Hamiltonian neural network (`lc_phs_trainer.py:499`) uses a destructive projection hack:

```
h' = γ(z) ⊙ h + β(z)
```

The rotation "invariance" hack in `hnn_bridge.go` projects everything to a unit vector n̂ ∈ S² plus a scalar h. This is a **destructive projection** that scrubs off-manifold physics:

- Small altitude perturbations carrying drag signals → zeroed
- Density gradients → lost
- Velocity shear → lost
- Network learns **invariance** instead of **equivariance**
- Cannot output bank-angle fields that transform correctly under SO(3)

**Result:** Garbage gradients during satellite slew, wrong control coupling g, brittle behavior under high-velocity maneuvers.

## First Principles

Physical quantities live in irreducible representations of the Euclidean group E(3) = SE(3) ⋊ parity:

- **l=0 scalars:** Hamiltonian density h, density ρ, speed of sound γ
- **l=1 vectors:** Position r, velocity v, Earth rotation ω_E, thrust
- **l=2 tensors:** J₂ gravity gradient contributions, inertia tensor

Any model that does not respect how these transform under rotation/translation is **not learning physics** — it is memorizing a fixed coordinate system.

## Solution: SE(3)-Equivariant Backbone

Replace the entire stack with irreducible-representation (irrep) math using **e3nn** library.

### Architecture

```
Input (irrep tensors) → e3nn backbone → Scalar Hamiltonian (l=0)

l=0 scalars: h, ρ, Mach, altitude
l=1 vectors: r, v, ω_E, thrust  
l=2 tensors: J₂ gradients, inertia
```

### Benefits

1. **No destructive projection** - all directional information preserved
2. **Physics by construction** - symmetries baked into architecture
3. **Automatic equivariance** - outputs transform correctly under SO(3)
4. **Correct gradients** - ∂H/∂x is equivariant vector field by construction
5. **Energy conservation** - emerges from Noether's theorem

### Implementation Files

- `ai/v3/e3nn_equivariant_hnn.py` - Complete SE(3)-equivariant HNN implementation
- `pkg/physics/hnn_bridge.go` - Needs migration from S² projection to irrep format

## Migration Steps

### Phase 1: Setup (Today)

```bash
# Clone e3nn
cd /Users/adi/idex/ai/v3
git clone https://github.com/e3nn/e3nn.git
cd e3nn
pip install -e .

# Verify installation
python -c "import e3nn; print('e3nn installed')"
```

### Phase 2: Port Features (This Week)

Convert current HNN features to irrep tensor format:

```python
# Current (destructive projection)
n = [x, y, z] / r  # Unit vector on S²
h = r - R_earth    # Altitude scalar

# New (proper irrep)
scalars = [h, ρ, Mach, altitude]  # l=0
vectors = [r, v, ω_E, thrust]     # l=1
tensors = [J₂_gradient, inertia] # l=2
```

### Phase 3: Train (2 Weeks)

Train on simulated orbital + aerodynamic data:

```python
model = SE3EquivariantPHS(config)

# Loss: energy conservation + symplectic constraints
loss = symplectic_loss(H, R, g) + equivariance_loss
```

### Phase 4: Benchmark (1 Week)

Test under random SO(3) rotations:

```python
# Rotate input by random rotation matrix
R = random_rotation_matrix()
state_rotated = R @ state

# Current projection hack: collapses
# New equivariant backbone: stable
H_rotated = model(state_rotated)
assert torch.allclose(H, H_rotated, atol=1e-6)
```

### Phase 5: Integration (2 Weeks)

Integrate with Go propagator:

```go
// Replace ProjectToS2 with irrep serialization
func (b *HNNBridge) SerializeIrrep(state State6D) []byte {
    // Serialize scalars, vectors, tensors separately
    // No destructive projection
}
```

## R-net: PSD Equivariant Operator

Parameterization ensures positive semi-definiteness structurally:

```
R = Σ_k α_k({invariants}) · (u_k u_kᵀ)

where:
- u_k are equivariant vector features (l=1)
- α_k are learned non-negative scalars from invariants
- PSD is automatic (sum of PSD outer products)
- Equivariance is automatic (u_k transforms correctly)
```

## g-net: Equivariant Control

Parameterized as equivariant vector field conditioned on bank angle (axis-angle irrep-1):

```python
g = f(state_vectors, bank_angle_axis_angle)

# When spacecraft rotates:
# state_vectors → R @ state_vectors
# bank_angle → R @ bank_angle
# g → R @ g  (correct transformation!)
```

## Library Recommendations

**Core: e3nn (PyTorch)**
- https://github.com/e3nn/e3nn
- Production standard for E(3)-equivariant nets
- Modular tensor products, spherical harmonics
- Runs on Apple MPS with minor patches

**High-Impact Forks:**
- QuantumLab-ZY/HamGNN - E(3)-equivariant GNN for Hamiltonian matrices
- tensorfieldnetworks/tensorfieldnetworks - Original TFN implementation
- shiangfang/e3nn-models - Generalized energy models on NequIP + e3nn

**Integration:**
- Feed e3nn outputs into nyx-space (Rust) or SSAPy propagator
- Use equivariant features upstream of PyDMD for hypersonic tripwires
- Quantize (Torch → ONNX) for radiation-hard OBC

## Flight Path

1. **Today:** Clone e3nn, install, verify
2. **This week:** Port features to irrep format
3. **30 days:** Train, benchmark, integrate with Go
4. **Flight:** Quantize, deploy on ISRO smallsat

## Timeline

- **Week 1:** e3nn installation, feature porting
- **Week 2:** Training on orbital data
- **Week 3:** SO(3) rotation benchmarking
- **Week 4:** Go integration and testing
- **Week 5-6:** Quantization and OBC deployment

## Success Criteria

1. ✅ e3nn installed and working
2. ✅ Features converted to irrep format
3. ✅ Model trained with <1% energy drift
4. ✅ Stable under random SO(3) rotations
5. ✅ Integrated with Go propagator
6. ✅ Quantized for OBC deployment

## References

- e3nn paper: "Equivariant Graph Neural Networks for 3D Graphs"
- NequIP: "Billion-scale fitting of physics-based neural network interatomic potentials"
- HamGNN: "Hamiltonian prediction with E(3)-equivariant GNNs"
- SEGNN: "Geometric and physical quantities improve E(3) message passing"

---

**This is not an incremental fix. It is first-principles orbital truth.**

Symmetries are baked into the architecture, not learned or projected away.
Photons, gravity gradients, and aerodynamic forces transform under E(3) whether you like it or not.
Your SSA satellite must respect that or it becomes a blind, brittle target.

**Execution starts now.**
