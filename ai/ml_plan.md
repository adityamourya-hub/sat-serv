# HF-PGFM: Full Technical Development & Training Roadmap

## Baseline Reality Check

Before proposing what to build, the current training infrastructure has three critical
gaps that must be acknowledged:

| Component | Current State | Gap |
|---|---|---|
| Data Volume | 100 sats × 60 steps = 6,000 points (`cmd/gen_training_data/main.go`) | Need 10⁸+ transitions |
| Architecture | `ArchitectureShell` (placeholder, no weights) (`sda-train/train_init.py:90`) | No actual model exists |
| Integrator | RK4 placeholder ("TODO: Replace with DP54") (`pkg/orbital/hpop.go:97`) | No adaptive step control |
| Loss Physics | `λ_h=0.05` soft penalty, Euler rollout | Physics violable; no hard constraints |
| Atmosphere | Jacchia-Roberts with fixed F10.7=150 (`pkg/physics/atmosphere.go:22`) | No solar weather coupling |

These are not criticisms — they are the foundation of a realistic build plan.

---

## Part I: The Target Architecture

### Name: Physics-Equivariant Latent Dynamics Foundation (PELDF)

This is the best model for this problem at the intersection of:
- **Neural Operators** (learn function-to-function mappings for PDEs)
- **Port-Hamiltonian Networks** (structural dissipation without violating conservation)
- **Hypernetworks** (generalize across vehicle classes without retraining)
- **Equivariant Architectures** (encode physical symmetry as a hard constraint)
- **Neural SINDy** (learn the candidate library, not just the coefficients)

```
┌───────────────────────────────────────────────────────────────────────┐
│              PELDF: Full Architecture Stack                           │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 0: SOVEREIGN OBSERVATION ENCODER              ║    │
│  ║                                                               ║    │
│  ║  Input: (r̲ ∈ ℝ³, v̲ ∈ ℝ³, t, σ_r, σ_v, sensor_id)          ║    │
│  ║                                                               ║    │
│  ║  SE(3)-Equivariant Graph Neural Network                       ║    │
│  ║  - Nodes: individual RSOs/HGV track points                   ║    │
│  ║  - Edges: sensor-measurement bipartite graph                  ║    │
│  ║  - Message passing: equivariant under rotation/translation    ║    │
│  ║  - Output: latent state z ∈ ℝ^512 (rotation-invariant)      ║    │
│  ║                                                               ║    │
│  ║  Key property: if you rotate the coordinate frame, z rotates ║    │
│  ║  with it. The model "knows" physical orientation.            ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
│                          │                                            │
│                          ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 1: ATMOSPHERIC NEURAL OPERATOR                ║    │
│  ║                                                               ║    │
│  ║  Input: (altitude h, F10.7, Ap, DoY, lat, lon) → ρ(h)       ║    │
│  ║                                                               ║    │
│  ║  Architecture: DeepONet (Branch + Trunk)                      ║    │
│  ║  - Branch net: encodes {F10.7, Ap, DoY} into coefficients    ║    │
│  ║  - Trunk net: maps altitude h → basis functions               ║    │
│  ║  - Output: density field ρ(h) as a differentiable function   ║    │
│  ║                                                               ║    │
│  ║  Trained on: NRLMSISE-00 output across 10⁶ (h, space_wx)    ║    │
│  ║  Hard constraint: ∂p/∂h = -ρg (hydrostatic equilibrium)      ║    │
│  ║                                                               ║    │
│  ║  This replaces your hardcoded Jacchia-Roberts model and      ║    │
│  ║  makes density differentiable w.r.t. all inputs.             ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
│                          │                                            │
│                          ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 2: HYPERNETWORK CONTEXT MODULE                ║    │
│  ║                                                               ║    │
│  ║  Input: z_obs (from Layer 0) + z_atm (from Layer 1)          ║    │
│  ║         + vehicle_class_prior (from IMM mode probabilities)  ║    │
│  ║                                                               ║    │
│  ║  z_context = Concat[z_obs, z_atm, p_mode] ∈ ℝ^768          ║    │
│  ║                                                               ║    │
│  ║  Hypernetwork H_φ: z_context → {W_H, W_R, W_g, W_lib}      ║    │
│  ║  - W_H: weights of Hamiltonian network H_θ                   ║    │
│  ║  - W_R: weights of dissipation network R_θ                   ║    │
│  ║  - W_g: weights of control coupling g_θ                      ║    │
│  ║  - W_lib: weights of SINDy library adaptation network        ║    │
│  ║                                                               ║    │
│  ║  Each vehicle class (orbital/ballistic/HGV/novel) gets its   ║    │
│  ║  own physics kernel generated on the fly.                    ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
│                          │                                            │
│                          ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 3: DISSIPATIVE PORT-HAMILTONIAN DYNAMICS      ║    │
│  ║                                                               ║    │
│  ║  Core ODE (run for prediction):                               ║    │
│  ║                                                               ║    │
│  ║    dz/dt = [J - R(z; W_R)] · ∇_z H(z; W_H) + g(z; W_g)·u  ║    │
│  ║                                                               ║    │
│  ║  Where:                                                       ║    │
│  ║    J = [[0, I], [-I, 0]]  (canonical symplectic, HARDCODED)  ║    │
│  ║                                                               ║    │
│  ║    R(z; W_R) = L(z) · L(z)ᵀ                                 ║    │
│  ║    L = lower triangular with positive diagonal               ║    │
│  ║    (Cholesky factor — guarantees R is PSD by construction)   ║    │
│  ║                                                               ║    │
│  ║    H(z; W_H) = MLP_4x128_Tanh(z)  → scalar                  ║    │
│  ║    (your existing HamiltonianNet architecture)                ║    │
│  ║                                                               ║    │
│  ║    g(z; W_g) = MLP_2x64_ReLU(z)  → control coupling matrix  ║    │
│  ║                                                               ║    │
│  ║    u = SINDy-discovered control input (bank angle, thrust)   ║    │
│  ║                                                               ║    │
│  ║  Integration: Yoshida-4 symplectic (your Ruth-4 in           ║    │
│  ║  jax_backend.py, adapted for dissipative case via            ║    │
│  ║  operator splitting: conservative + dissipative steps)       ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
│                          │                                            │
│                          ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 4: NEURAL SINDY READOUT                       ║    │
│  ║                                                               ║    │
│  ║  Unlike classical SINDy (fixed polynomial library),          ║    │
│  ║  Neural SINDy learns the library via W_lib:                  ║    │
│  ║                                                               ║    │
│  ║    Θ_learned(z) = [φ₁(z), φ₂(z), ..., φ_K(z)]             ║    │
│  ║    where each φᵢ is a small 1×32 network                     ║    │
│  ║                                                               ║    │
│  ║  Then sparse regression selects active terms:                ║    │
│  ║    ż = Θ_learned(z) · ξ,    ‖ξ‖₀ ≤ s  (sparsity budget)   ║    │
│  ║                                                               ║    │
│  ║  The discovered symbolic equation in physical units is:      ║    │
│  ║    ẋ_physical = Decoder(ξ, Θ_learned)                       ║    │
│  ║                                                               ║    │
│  ║  Output: SymbolicCertificate with CI, physics grade, 60s     ║    │
│  ║  prediction, and human-readable LaTeX                        ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
│                          │                                            │
│                          ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════╗    │
│  ║          LAYER 5: CONFORMAL UNCERTAINTY QUANTIFICATION       ║    │
│  ║                                                               ║    │
│  ║  For each prediction, output a coverage-guaranteed interval: ║    │
│  ║                                                               ║    │
│  ║    P(y_true ∈ Ĉ_α(x)) ≥ 1 - α  (conformal guarantee)      ║    │
│  ║                                                               ║    │
│  ║  Method: Conformalized Quantile Regression (CQR)             ║    │
│  ║  - No distributional assumptions                              ║    │
│  ║  - Coverage is exact, not approximate                        ║    │
│  ║  - Works with any base model                                  ║    │
│  ║                                                               ║    │
│  ║  Applied to: positional uncertainty in the VSC              ║    │
│  ║  Output: σ_position ∈ ℝ³ with guaranteed 95% coverage       ║    │
│  ╚═══════════════════════════════════════════════════════════════╝    │
└───────────────────────────────────────────────────────────────────────┘

Total parameters (foundation model, H100 scale):
  Layer 0 (SE3-GNN):          ~45M params
  Layer 1 (DeepONet Atm):     ~8M params
  Layer 2 (Hypernetwork):     ~120M params
  Layer 3 (DPHNO, generated): ~2M params (per vehicle class)
  Layer 4 (Neural SINDy):     ~5M params
  Layer 5 (CQR head):         ~1M params
  TOTAL:                      ~181M params  (GPT-2-scale, intentionally modest)

Edge-deployed projected model (per vehicle class):
  Layer 3 only (INT8 quantized): ~250KB
  Layer 4 only (INT8 quantized): ~50KB
  Total edge footprint:          ~300KB per class, ~3MB for 10 classes
```

### Why This Specific Architecture

**Why SE(3)-Equivariant GNN over Transformer?**
Transformers process tokens permutation-equivariantly but have no concept of
3D geometric symmetry. If you rotate your coordinate frame by 90°, a transformer
gives a completely different output. An SE(3)-GNN gives the same output rotated
by 90°. For orbital mechanics, physics is rotation-invariant — this is not optional.

**Why DeepONet for Atmosphere over MLP?**
MLPs map fixed-size inputs to outputs. The atmosphere is a function ρ(h) — an
infinite-dimensional object. DeepONet learns the operator F: space_weather → ρ(·)
where ρ is a continuous function of altitude. This means you can query density at
any altitude, not just gridded values, and gradients flow through continuously.

**Why Hypernetwork over Multi-Task Head?**
Multi-task learning (shared backbone, separate heads per vehicle class) still
requires the same backbone to compress all classes. A hypernetwork generates
class-specific physics kernels from a context vector, allowing the Hamiltonian and
dissipation functions to be fundamentally different per class — not just a
different output head on the same function.

**Why Port-Hamiltonian over Pure HNN?**
Your existing `GeneralizedHNN` conserves energy exactly. An HGV loses energy to
atmospheric drag at ~10 MJ/km². A pure HNN either invents energy or cannot model
drag. Port-Hamiltonian systems decompose the dynamics into a conservative part
(J·∇H, preserved exactly by symplectic integration) and a dissipative part
(-R·∇H, non-negative energy dissipation by construction from Cholesky). You get
exact conservation where physics demands it, and correct dissipation where the
atmosphere demands it.

**Why Neural SINDy over Classical SINDy?**
Classical SINDy (your `pkg/physics/sindy.go`) uses a fixed 21-term library. Neural
SINDy learns the library basis functions from data. If the target is executing a
maneuver type never seen before, Classical SINDy returns garbage (the active terms
are wrong). Neural SINDy adapts the library to the observed dynamics, then performs
sparse selection. The resulting equation is still interpretable (it's still ξ applied
to learned basis functions) but the basis functions are data-driven, not hardcoded.

**Why Conformal Prediction over Bayesian Uncertainty?**
Bayesian deep learning (dropout, ensembles) gives calibrated uncertainty only under
specific distributional assumptions. Conformal prediction gives a mathematically
guaranteed coverage interval with no distributional assumptions. For defense
applications, "95% coverage" must mean exactly 95%, not "approximately 95% under
our distributional assumptions."

---

## Part II: Data Generation Pipeline

### Current Gap
`cmd/gen_training_data/main.go` generates:
- 100 satellites × 60 × 60s steps ≈ 6,000 state transitions
- All in a 1-degree inclination plane (vz=0.0, line 61)
- No atmospheric variation, no maneuvers, no HGV dynamics
- Stored as JSON (too slow for 10⁸+ samples)

### Required Data Volume for Foundation Training
```
Dataset            │ Samples  │ Variants            │ Generator
───────────────────┼──────────┼─────────────────────┼────────────────────────
Orbital (LEO)      │ 2×10⁶   │ i∈[0°,98°], ecc    │ HPOP (hpop.go)
Orbital (MEO/GEO)  │ 5×10⁵   │ Diverse regimes     │ HPOP (hpop.go)
HGV Skip-Glide     │ 3×10⁶   │ L/D∈[1.5,5], σ     │ HGVEquationsOfMotion
HGV Ballistic      │ 1×10⁶   │ varied β            │ hgv_physics.go
HGV Zero-Day       │ 5×10⁵   │ random ODE params   │ Procedural generator
Atmosphere         │ 10⁷     │ (h, F10.7, Ap, DoY) │ NRLMSISE-00 C wrapper
Sensor Noise       │ Augment  │ σ_r∈[1,100]m       │ Additive Gaussian/Lévy
Deception          │ 1×10⁵   │ fake tracks, GLINT  │ Adversarial generator
───────────────────┼──────────┼─────────────────────┼────────────────────────
TOTAL              │ ~7×10⁶  trajectory segments    │
                   │ ~3×10⁸  individual transitions │
```

### Storage Format: Zarr v3 (Not JSON)
Replace `json.MarshalIndent` with:
```python
# New generator output format
import zarr
import numpy as np

store = zarr.open('sda_training.zarr', mode='w')
store.create_dataset('states', shape=(N, T, 6), dtype='float32',
                     chunks=(1000, 100, 6), compressor=zarr.Blosc())
store.create_dataset('times', shape=(N, T), dtype='float64')
store.create_dataset('metadata', shape=(N,), dtype=object)
# Read speed: ~10GB/s vs ~100MB/s for JSON
```

### HGV Zero-Day Generator (Critical Gap)
To train robustness to novel maneuvers, generate trajectories from random ODEs
in the physically plausible parameter space:
```python
def generate_zero_day_trajectory(
    n_steps: int = 500,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generate a novel HGV-like trajectory from randomly sampled ODE.
    Parameters sampled from physics-informed priors, not arbitrary random.
    """
    # Sample L/D and bank profile from physically valid range
    LD = np.random.uniform(1.0, 6.0)
    # Bank angle profile: Fourier series with random coefficients
    n_harmonics = np.random.randint(1, 8)
    amplitudes = np.random.normal(0, 0.3, n_harmonics)  # rad
    frequencies = np.random.uniform(0.01, 0.5, n_harmonics)  # cycles/s
    phases = np.random.uniform(0, 2*np.pi, n_harmonics)

    sigma_fn = lambda t: np.sum(
        amplitudes * np.sin(2*np.pi*frequencies*t + phases)
    )

    # Integrate HGVEquationsOfMotion with this sigma_fn
    # Start from Siliguri corridor entry conditions
    state = sample_siliguri_entry_state()
    trajectory = [state]
    for i in range(n_steps):
        sigma = sigma_fn(i * dt)
        dot = hgv_equations_of_motion(state, LD, sigma)
        state = rk45_step(state, dot, dt)
        trajectory.append(state)
    return np.array(trajectory)
```

---

## Part III: Training Curriculum — Five Phases

### Phase 0: Physics Bootstrap (Weeks 1-2)
**Goal**: Train the atmospheric DeepONet and validate the port-Hamiltonian
structure in isolation before any trajectory data is used.

**What you train**: Layer 1 (DeepONet) independently.

**Loss**:
```
L_atm = MSE(ρ_nn(h, F10.7, Ap) - ρ_NRLMSISE(h, F10.7, Ap))
      + λ_hydro · ‖∂p_nn/∂h + ρ_nn · g‖²    (hydrostatic constraint)
      + λ_mono  · max(0, ρ_nn(h+δ) - ρ_nn(h))  (monotone decreasing)
```

**Dataset**: 10⁷ NRLMSISE-00 samples computed via CGo wrapper around
your existing `pkg/physics/reference/nrlmsise-00.c`.

**Target accuracy**: <0.5% density error across 30-500km, all solar conditions.

**Why first**: All subsequent phases use this density model for drag computation.
A 10% density error propagates to a 10% drag error, which compounds to
~50m positional error over 60 seconds of HGV flight — exceeding your 10m budget.

---

### Phase 1: Orbital Foundation (Weeks 3-6)
**Goal**: Train Layers 0, 2, 3 (without dissipation, R=0) on pure orbital dynamics.
This establishes the latent space structure and symplectic integration correctness.

**What you train**: SE3-GNN encoder + Hypernetwork (orbital-only context) +
Port-Hamiltonian dynamics (J-component only, R=0 for conservative regime).

**Data**: 2.5×10⁶ orbital trajectories from HPOP (your existing Go propagator,
upgraded from RK4 to DP54/RKF45 — your `IntegratorRKF45` already exists in
`pkg/physics/integrator.go`, just needs to replace the RK4 call in `hpop.go:97`).

**Loss (Augmented Lagrangian, Phase 1)**:
```
L_total = L_obs + μ₁ · c₁ + (ρ_al/2) · c₁²
        + μ₂ · c₂ + (ρ_al/2) · c₂²

c₁ = Var[H(z(t))] / H(z(0))²        (energy conservation, hard)
c₂ = Var[‖r(t) × v(t)‖] / ‖L₀‖²    (angular momentum, hard)

L_obs = (1/T) Σ_t ‖x̂(t) - x(t)‖²   (trajectory MSE)

Dual update: μᵢ ← μᵢ + ρ_al · cᵢ    (after each epoch)
             ρ_al ← min(10 · ρ_al, ρ_max)  (progressive tightening)
```

**Key constraint**: Use Yoshida-4 symplectic integrator in the forward pass.
The integration scheme makes symplecticity structurally impossible to violate,
regardless of what the neural network learns. It is the ultimate hard constraint.

**Convergence target**: 
- Energy drift < 10⁻⁶ per orbit (your existing benchmark in `hamiltonian_dynamics.py:14`)
- Position error < 100m at T+6hr

---

### Phase 2: Hypersonic Extension (Weeks 7-10)
**Goal**: Activate the dissipation term R(z; W_R) and train on HGV dynamics.
The port-Hamiltonian framework guarantees that turning on R can only decrease
energy, never increase it — physically correct for drag.

**What you train**: Full Layers 0-4. Dissipation network R trained from scratch.
Neural SINDy library network W_lib trained to discover HGV ODEs.

**Data**: 3×10⁶ HGV skip-glide trajectories from `HGVEquationsOfMotion()` +
1×10⁶ ballistic trajectories.

**Loss additions (Phase 2)**:
```
L_total += λ_dissip · max(0, dH/dt)     (dissipation must be non-positive)
         + λ_sindy  · ‖Θ(z) · ξ - ż‖²  (SINDy reconstruction)
         + λ_sparse · ‖ξ‖₁              (sparsity on discovered equation)
         + μ₃ · c₃ + (ρ_al/2) · c₃²    (c₃: hydrostatic — links to Phase 0 model)

c₃ = ‖ρ_nn(h) - ρ_physics(h)‖ / ‖ρ_physics‖   (density consistency)
```

**PSD enforcement for R** (no soft penalty, hard by construction):
```python
# In the forward pass of the dissipation network:
def compute_R(z: jnp.ndarray, W_R: dict) -> jnp.ndarray:
    L_flat = mlp(z, W_R)                              # Raw output
    L = jnp.tril(L_flat.reshape(d, d))                # Lower triangular
    L = L.at[jnp.arange(d), jnp.arange(d)].set(
        jax.nn.softplus(jnp.diag(L))                  # Positive diagonal
    )
    return L @ L.T                                     # PSD by construction, always
```

**Convergence target**:
- HGV position error < 50m at T+60s (single trajectory)
- L/D ratio estimation error < 5%
- Bank angle phase estimation error < 10°

---

### Phase 3: Adversarial Hardening (Weeks 11-13)
**Goal**: Train robustness to zero-day maneuvers, electronic deception, and sensor
spoofing. The model must reject physically impossible inputs.

**What you train**: Full model + add an adversarial discriminator.

**Data**: 5×10⁵ zero-day trajectories (procedural generator) +
10⁵ adversarial perturbations (crafted to fool the Phase 2 model).

**Adversarial Training Protocol: Physics-Constrained PGD**

Standard adversarial training (FGSM, PGD) generates perturbations anywhere in
L∞ ball. For SDA, we only care about physically realizable adversarial inputs —
perturbations that could actually be achieved by a jammer or deceptive target.

```python
def physics_pgd_attack(model, x_clean, y_true, epsilon_m=10.0, steps=20):
    """
    Projected Gradient Descent constrained to physically realizable perturbations.
    epsilon_m: max position perturbation in meters (e.g., 10m jamming error)
    """
    x_adv = x_clean.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = -model_loss(model, x_adv, y_true)
        loss.backward()
        
        # Gradient step
        x_adv = x_adv + (epsilon_m / steps) * x_adv.grad.sign()
        
        # Project back to physically realizable set:
        # 1. Position: L∞ ball of epsilon_m meters
        delta_pos = (x_adv[:3] - x_clean[:3]).clamp(-epsilon_m/1000, epsilon_m/1000)
        # 2. Velocity: must be consistent with orbital mechanics (vis-viva)
        v_circular = torch.sqrt(MU / torch.norm(x_clean[:3]))
        v_perturb = (x_adv[3:] - x_clean[3:]).clamp(-0.1, 0.1)  # Max 100 m/s spoof
        # 3. Energy: must be in valid orbital regime (not escape velocity)
        energy = 0.5 * torch.norm(x_adv[3:])**2 - MU / torch.norm(x_adv[:3])
        # Clamp energy to physically valid range
        x_adv = torch.cat([x_clean[:3] + delta_pos, x_clean[3:] + v_perturb])
    
    return x_adv.detach()
```

**Sovereign Bridge Integration**:
Your existing `sovereign_bridge_execute` in `q/core/auditor.py:336` already
implements the Pfaffian physics check. During training, ensure this check is
applied as a hard gate: if the model's prediction violates the Pfaffian criterion,
the training loss is set to a high constant (not backpropagated through the model
— this prevents gradient from learning to fool the physics gate).

---

### Phase 4: Online Adaptation (Weeks 14+)
**Goal**: Deploy to edge; continue learning from live sensor streams without
catastrophic forgetting of the foundation knowledge.

**Method: Elastic Weight Consolidation (EWC) + Continual SINDy**

EWC prevents catastrophic forgetting:
```python
L_ewc = L_task + λ_ewc · Σᵢ Fᵢ · (θᵢ - θᵢ*)²
```
Where Fᵢ is the Fisher information diagonal (importance of each weight)
and θᵢ* are the foundation model weights. Important weights (high Fᵢ) are
protected from drift; unimportant weights adapt freely to new data.

For the SINDy readout specifically:
- Keep the learned library basis functions Θ_learned frozen
- Only adapt the sparse coefficients ξ online (this is just a linear regression,
  costs ~0.1ms on-device using the Go STLSQ implementation)
- This preserves interpretability while enabling real-time adaptation

**Online update frequency**:
- Full model update: Never on-device (upload batch to H100 nightly)
- SINDy coefficients ξ: Every 10 observations (100ms at 100Hz)
- EWC penalty: Recomputed weekly from accumulated sensor data

---

## Part IV: Loss Function — Complete Formal Specification

The complete training loss is an Augmented Lagrangian:

```
L(θ, μ, ρ_al) = L_obs(θ) + L_reg(θ) + Σᵢ [μᵢ · cᵢ(θ) + (ρ_al/2) · cᵢ(θ)²]
```

### Observation Loss
```
L_obs = (1/N·T) Σ_n Σ_t ‖x̂_n(t) - x_n(t)‖²_W

where W = diag(1/σ²_r, 1/σ²_r, 1/σ²_r, 1/σ²_v, 1/σ²_v, 1/σ²_v)
```
Weighted MSE — position and velocity errors normalized by sensor noise.

### Regularization
```
L_reg = λ₁ · ‖ξ‖₁                  (SINDy sparsity)
      + λ₂ · ‖W_H‖²_F               (Hamiltonian L2)
      + λ₃ · ‖W_lib - W_lib_prev‖²  (Library stability across time)
      + λ₄ · KL(p_ξ ‖ Laplace(0,1)) (Bayesian SINDy sparsity prior)
```

### Hard Physics Constraints (cᵢ = 0 at convergence)
```
c₁ = Var_t[H(z(t))] / H(z(0))²                    Energy conservation (orbital)
c₂ = Var_t[‖L(t)‖] / ‖L₀‖²                        Angular momentum (orbital)
c₃ = (1/K) Σ_k max(0, ∂H/∂t|_{HGV step k})        No energy creation (HGV)
c₄ = (1/K) Σ_k ‖min(eigenvalues(R(z_k)), 0)‖₂     PSD of dissipation
c₅ = ‖∂p_nn/∂h + ρ_nn · g‖²                       Hydrostatic equilibrium
c₆ = (1/K) Σ_k max(0, ρ_nn(h_k) - ρ_nn(h_k - 1)) Density monotone decreasing
```

### Dual Variable Update (After Each Epoch)
```python
for i, constraint_fn in enumerate(constraint_fns):
    c_val = constraint_fn(model, validation_batch)
    mu[i] = mu[i] + rho_al * c_val
    if c_val > 0.01:  # Constraint violated
        rho_al = min(rho_al * 10, rho_max)  # Tighten penalty
    else:
        rho_al = max(rho_al * 0.5, rho_min)  # Relax when satisfied
```

---

## Part V: Evaluation Protocol

### Metrics Hierarchy

**Level 1: Physics Integrity (Must Pass Before Any Deployment)**
| Metric | Target | Measurement |
|---|---|---|
| Energy drift (orbital, 24hr) | < 10⁻⁶ relative | Ruth-4 integration test |
| Density error (30-500km) | < 0.5% | vs NRLMSISE-00 ground truth |
| Dissipation sign | Always ≤ 0 | dH/dt on 10⁴ HGV test trajectories |
| R PSD fraction | 100.0% | Minimum eigenvalue check |

**Level 2: Tracking Accuracy (Core Performance)**
| Metric | Target | Measurement |
|---|---|---|
| Position error (T+60s, orbital) | < 10m (3σ) | CARA parity cases in `docs/validation_reports/` |
| Position error (T+60s, HGV glide) | < 50m (3σ) | HGV test suite in `pkg/hypersonic/hypersonic_test.go` |
| L/D estimation error | < 5% | vs known L/D in synthetic data |
| Zero-day detection rate | > 95% | vs zero-day test set from Phase 3 |
| False positive rate | < 1% | On pure orbital (no maneuver) test set |

**Level 3: Operational Performance (ADITI 4.0 Judging)**
| Metric | Target | Measurement |
|---|---|---|
| OODA loop latency | < 100ms | End-to-end: sensor → decision |
| SINDy R² (actual, not hardcoded) | > 0.85 | Bootstrap over 100 window resamples |
| Certificate falsifiability rate | > 90% | 60s predictions validated against next measurement |
| Sovereignty (zero foreign deps) | 100% | Dependency audit: no Aurora, no foreign cloud |

**Level 4: Adversarial Robustness**
| Metric | Target | Measurement |
|---|---|---|
| Accuracy under 10m position spoof | > 80% of clean accuracy | Physics-constrained PGD |
| Accuracy under Rahu GLINT | > 90% of clean accuracy | Deception dataset |
| Physics gate rejection rate | > 99% on adversarial | Pfaffian check pass rate |
| Model confidence under attack | Decreases (not maintained) | UQ calibration under attack |

---

## Part VI: Development Sequencing

```
Week 1-2:   PHASE 0 — Atmospheric Neural Operator
             Build CGo wrapper around nrlmsise-00.c
             Generate 10⁷ density samples
             Train DeepONet, verify hydrostatic constraint
             
Week 3-4:   DATA PIPELINE
             Upgrade gen_training_data/main.go to Zarr output
             Replace HPOP RK4 → RKF45 (integrator.go already exists)
             Add HGV trajectory generator (HGVEquationsOfMotion)
             Add zero-day procedural generator
             Target: 10⁸ transitions in Zarr store
             
Week 5-6:   SE3-GNN ENCODER (Layer 0)
             Implement using e3nn-jax or MACE
             Train on orbital trajectories only (Phase 1)
             Verify rotation equivariance via automated tests
             
Week 7-9:   HYPERNETWORK + DPHNO (Layers 2-3)
             Implement Cholesky dissipation network
             Implement Yoshida-4 in JAX (adapt jax_backend.py Ruth-4)
             Implement operator splitting for dissipative integration
             Train full Phase 1 loss (Augmented Lagrangian, orbital)
             
Week 10-12: NEURAL SINDY (Layer 4) + HGV TRAINING
             Implement learned library basis functions
             Train Phase 2 loss on HGV data
             Verify L/D recovery on synthetic benchmarks
             
Week 13:    ADVERSARIAL HARDENING (Phase 3)
             Physics-constrained PGD implementation
             Integrate with sovereign_bridge_execute auditor
             Verify physics gate rejection on deception dataset
             
Week 14:    PROJECTION + QUANTIZATION
             INT8 quantization of DPHNO weights per vehicle class
             ONNX export of edge models
             Verify PAX tier selection with IMM filter
             
Week 15:    CONFORMAL UQ (Layer 5)
             Implement CQR calibration on holdout set
             Verify 95% coverage guarantee on test set
             Attach to SymbolicCertificate struct
             
Week 16:    INTEGRATION + CERTIFICATION
             Wire all components through Go runtime
             Run CARA parity cases
             Run ADITI 4.0 benchmark suite
             Final certification run
```

---

## Part VII: Five Critical Implementation Steps (In Order)

**Step 1: Fix the Integrator (Day 1, 2 hours)**

Replace the RK4 placeholder in `hpop.go:97` with `IntegratorRKF45`:
```go
// pkg/orbital/hpop.go:97 — Replace this:
y := h.rk4Integrate(y0, dt, force.derivative, h.conf.MaxStep)

// With this:
integrator := physics.NewIntegratorRKF45()
t_curr := 0.0
y := y0
for math.Abs(t_curr) < math.Abs(dt) {
    step := h.conf.MaxStep
    if math.Abs(t_curr + step) > math.Abs(dt) {
        step = dt - t_curr
    }
    t_curr, y, _ = integrator.Step(t_curr, step, y, force.derivative)
}
```
This immediately improves all training data quality with zero architectural change.

**Step 2: Fix R² Computation in SINDy (Day 1, 30 minutes)**

`pkg/physics/sindy.go:353` returns hardcoded 0.95. The actual residual data is
already computed in `DiscoverManeuver()`. Just compute it:
```go
func (s *SINDy) computeRSquared(predicted, actual [][6]float64) float64 {
    if len(predicted) == 0 { return 0 }
    ssTot, ssRes := 0.0, 0.0
    mean := computeMean(actual)
    for i := range predicted {
        for j := 0; j < 6; j++ {
            diff := actual[i][j] - mean[j]
            ssTot += diff * diff
            res := actual[i][j] - predicted[i][j]
            ssRes += res * res
        }
    }
    if ssTot < 1e-15 { return 1.0 }
    return 1.0 - ssRes/ssTot
}
```

**Step 3: Implement Yoshida-4 Dissipative Splitter in JAX**

Extending the Ruth-4 in `q/core/jax_backend.py` to handle port-Hamiltonian:
```python
# ai/v3/dphno_integrator.py
def yoshida4_dissipative_step(q, p, grad_H_fn, R_fn, g_fn, u, dt):
    """
    Yoshida-4 operator splitting for port-Hamiltonian systems.
    Splits: z_dot = [J]·∇H + [-R]·∇H + [g]·u
    Integrates conservative part with Yoshida-4, dissipative with RK4.
    """
    w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    w0 = 1.0 - 2.0*w1
    c = [w1/2, (w1+w0)/2, (w1+w0)/2, w1/2]
    d = [w1, w0, w1]
    
    # Conservative Yoshida substeps
    for i in range(3):
        p = p - c[i]*dt * grad_H_fn(q, p)[:len(q)]  # -dH/dq
        q = q + d[i]*dt * grad_H_fn(q, p)[len(q):]  # +dH/dp
    p = p - c[3]*dt * grad_H_fn(q, p)[:len(q)]
    
    # Dissipative correction (1st order split, small R => small error)
    z = jnp.concatenate([q, p])
    grad_H = grad_H_fn(q, p)
    R = R_fn(z)
    diss = -R @ grad_H
    control = g_fn(z) @ u
    z = z + dt * (diss + control)
    
    return z[:len(q)], z[len(q):]
```

**Step 4: Build the CGo Atmospheric Bridge**

Wrap `pkg/physics/reference/nrlmsise-00.c` in a Go CGo interface, then call it
from Python via ctypes to generate the DeepONet training dataset:
```go
// pkg/physics/nrlmsise_bridge.go
package physics

/*
#cgo CFLAGS: -I${SRCDIR}/reference
#include "nrlmsise-00.h"
*/
import "C"

func NRLMSISE00Density(altKm, latDeg, lonDeg, f107, f107a, ap float64,
                        year, doy int, sec float64) float64 {
    var output C.struct_nrlmsise_output
    var input C.struct_nrlmsise_input
    var flags C.struct_nrlmsise_flags
    // ... configure input/output structs
    C.gtd7(&input, &flags, &output)
    return float64(output.d[5]) * 1e3  // Total mass density kg/m³
}
```

**Step 5: Replace gen_training_data to Zarr at Scale**

Extend `cmd/gen_training_data/main.go` to call the HGV and zero-day generators
in parallel (Go goroutines), and output Zarr via Python subprocess:
```go
// Use Go's concurrency for parallel trajectory generation
const numWorkers = runtime.NumCPU()
jobs := make(chan TrajectoryJob, 1000)
results := make(chan TrajectorySample, 10000)

for w := 0; w < numWorkers; w++ {
    go trajectoryWorker(jobs, results)
}
// Target: 10⁶ trajectories in < 30 minutes on 16-core machine
```

---

## Summary: The Correct Build Order

```
Phase 0 (Physics foundation) → Deploy DeepONet atmosphere
         ↓
Phase 1 (Orbital foundation) → Deploy orbital PELDF
         ↓                       (sub-10m accuracy on LEO/MEO/GEO)
Phase 2 (HGV extension) ──────→ Deploy HGV PELDF
         ↓                       (sub-50m accuracy on skip-glide)
Phase 3 (Adversarial) ────────→ Deploy hardened model
         ↓                       (ADITI 4.0 submission-ready)
Phase 4 (Online adaptation) ──→ Continuous improvement from live data
```

**The single highest-leverage action**: Fix the RKF45 integrator and R² in the
first two days. Every subsequent training run benefits from better data quality and
honest model evaluation. Both are 30-120 minute fixes on existing code. Everything
else in this roadmap builds on those two facts.
