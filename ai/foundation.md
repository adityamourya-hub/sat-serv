# Foundational Physics Model Architecture: First-Principles Technical Analysis

## Context

This document analyzes the current AI/physics model stack in `/Users/adi/idex` (orbital + HGV intelligence) alongside `/Users/adi/ganga1.0` (robotics), derives from first principles what a *foundational* physics model architecture ought to be, evaluates how far the existing code is from that target, and specifies a development path.

**Why now.** The current system is a strong local maximum for orbital maneuver detection but has structural defects that will prevent it from generalizing to HGVs, zero-day maneuvers, or cross-domain (satellite → robot) transfer. The architecture proposal already in [model_arch.md](/Users/adi/idex/ai/model_arch.md) points in the right direction but places the hypernetwork/PAX-tiering ahead of base-model correctness, and misdiagnoses several code-level issues. This plan corrects those.

**Intended outcome.** A prioritized, executable path from the current fragmented stack to a unified **Lie-Group Port-Hamiltonian Neural Operator (LG-PHNO)** foundation model trained under hard constraints, with SINDy demoted to residual-inverse control inference and FNO/DeepONet reserved for genuinely infinite-dimensional fields (atmosphere, contact).

---

## 1. Verified State of the Current Architecture

Directly confirmed by reading source (not derived from docs):

| Component | File | Status | Notes |
|---|---|---|---|
| Port-Hamiltonian system (JAX) | [ai/v3/port_hamiltonian.py](/Users/adi/idex/ai/v3/port_hamiltonian.py) | **Correct** | Cholesky PSD `R`, canonical `J`, control coupling `g`, Yoshida-4 integrator with proper q/p splitting. Curriculum-staged training described but not wired. |
| GHNN (PyTorch) | [ai/neural_ode/ghnn.py](/Users/adi/idex/ai/neural_ode/ghnn.py) | Partial | Scalar H(q,p), autograd vector field, ONNX export. Conservative-only. No dissipation or control port. |
| "Symplectic" training loop | [ai/neural_ode/train.py:142-177](/Users/adi/idex/ai/neural_ode/train.py#L142-L177) | **Broken** | Claims Yoshida-4 but applies `c1..c4` to the full state vector with no `d` coefficients and no q/p split. This is a 4-stage explicit RK-like integrator, *not* symplectic. Energy conservation claim is false. |
| Forest-Ruth integrator | [q/core/jax_backend.py:510-552](/Users/adi/idex/q/core/jax_backend.py#L510-L552) | Correct | Proper q/p alternation with `C_R4`/`D_R4` coefficients. Not reused by PyTorch trainer. |
| Classical SINDy | [pkg/physics/sindy.go](/Users/adi/idex/pkg/physics/sindy.go) | Partial | Fixed 21-term library (polynomial + single-freq trig + v·v). Real R² computation at [:353-417](/Users/adi/idex/pkg/physics/sindy.go#L353-L417) — the `0.95` claim in prior docs was incorrect. |
| Sliding SINDy | [pkg/physics/sliding_sindy.go](/Users/adi/idex/pkg/physics/sliding_sindy.go) | Partial | Windowed refit, intent mapping. No bootstrap CIs; no causal provenance. |
| HGV 6-DOF physics | [pkg/hypersonic/hgv_physics.go](/Users/adi/idex/pkg/hypersonic/hgv_physics.go) | Classical | Spherical-coordinate EOM with bank-angle `σ` hidden. Not port-Hamiltonian. |
| HGV IMM filter | [pkg/hypersonic/hgv_imm_filter.go](/Users/adi/idex/pkg/hypersonic/hgv_imm_filter.go) | Partial | 3-mode Markov with static transition matrix; Kalman gain hardcoded 0.3. Zero-day maneuvers have no mode. |
| Atmosphere | [pkg/physics/atmosphere.go](/Users/adi/idex/pkg/physics/atmosphere.go) | Partial | Jacchia-Roberts valid >100 km; hardcoded `F10.7=150, Ap=15`. NRLMSISE-00 C reference present but unwired. |
| Astrojax world model | [ai/v3/astrojax_world_model.py:121-162](/Users/adi/idex/ai/v3/astrojax_world_model.py#L121-L162) | Correct RK4 | Prior doc's "Euler" claim was incorrect — this file uses RK4. Exponential drag single-scale-height is the real issue. |
| HNO (FNO backbone) | [q/core/hno.py](/Users/adi/idex/q/core/hno.py) | Placeholder | Spectral layers stacked; no physics constraint. Not load-bearing yet. |
| Orchestration | [pkg/ml/hybrid/orchestrator.go](/Users/adi/idex/pkg/ml/hybrid/orchestrator.go) | Works | Confidence-gated ML→classical fallback. No UQ. |

**ganga1.0 cross-reference.** Identical symplectic primitives (Yoshida-4, leapfrog) in [sim/ganga_mjx/symplectic_mjx.py](/Users/adi/ganga1.0/sim/ganga_mjx/symplectic_mjx.py) and [brain/dynamics/sfmn_integrators/integrators.py](/Users/adi/ganga1.0/brain/dynamics/sfmn_integrators/integrators.py). E(n)-equivariant Hamiltonian Flow Net in [brain/dynamics/hamiltonian_flow_net.py](/Users/adi/ganga1.0/brain/dynamics/hamiltonian_flow_net.py). Energy auditor `E_hal` in [sim/ganga_mjx/energy_auditor.py](/Users/adi/ganga1.0/sim/ganga_mjx/energy_auditor.py). No cross-imports between the two repos, but the mathematical substrate is the same.

**Headline findings:**
1. A correct port-Hamiltonian exists in JAX but is not used by the PyTorch training pipeline.
2. The PyTorch training pipeline's "symplectic" integrator is not symplectic.
3. HGV physics is classical, not port-Hamiltonian — which is exactly where port-Hamiltonian matters most.
4. Physics enters training as a soft penalty (`λ_h=0.05`), freely tradeable against observation MSE.
5. All state is Cartesian ℝ⁶ — no Lie-group awareness despite HGV living naturally on `S² × ℝ³ × SO(3)` and robots on `SE(3) × 𝕋ⁿ`.

---

## 2. First-Principles Axioms for the Foundation Model

The model is a map `(state, context) → dynamics`. Before picking architecture, list the axioms any acceptable model must satisfy — structurally, not statistically:

**A1. Geometric invariance.** State lives on a manifold `M`, not ℝⁿ. For idex: `M = S² × ℝ³ × SO(3)` (HGV) or `T*(SE(3))` (orbital). For ganga1.0: `T*(SE(3) × 𝕋ⁿ)`. Learned dynamics must be equivariant to the symmetry group of `M` (rotations, translations, gauge).

**A2. Symplecticity.** On the cotangent bundle `T*M`, phase-space volume is preserved by any valid evolution of a closed Hamiltonian system. Violations are not modeling noise — they are physically impossible. Therefore the *integrator* must be symplectic, not the loss.

**A3. Passivity.** Real systems dissipate. `dH/dt ≤ uᵀy`. This is stronger than energy conservation: it says energy can only leave through controlled ports and dissipative channels. Port-Hamiltonian form `ẋ = (J − R)∇H + gu` with `R ⪰ 0` makes passivity structural, not learned.

**A4. Observability closure.** Some inputs are hidden: HGV bank angle `σ(t)`, robot contact impulses, motor torques. Any model that predicts `ẋ` from observed state alone is misspecified. The correct decomposition: learn `(J, R, H, g)` from data, then infer `u(t)` as the sparse residual on `ẋ − (J−R)∇H`.

**A5. Field coupling.** Certain physical quantities are infinite-dimensional functions: atmospheric density `ρ(h, ·)`, contact-distance field, gravity multipoles. Approximating them with MLPs wastes structure. Operator learning (FNO, DeepONet) is the right tool — but only for these fields, not for the whole dynamics.

**A6. Verifiability.** The model must emit falsifiable predictions with calibrated uncertainty. Conformal prediction, bootstrap CIs, and forward-prediction checks are required for any "sovereign proof" story.

**A7. Constraints are hard, not soft.** Conservation laws (energy, angular momentum, hydrostatic equilibrium) should be enforced by projection or Augmented Lagrangian with dual ascent, such that at convergence `c(θ) = 0` *exactly*, not `‖c(θ)‖ < ε`.

**A8. Compositionality.** A foundation model isn't a monolith. It's a family of `(H, R, g)` triples conditioned on a latent physics code `z` that encodes regime (orbital, HGV, robot, molecular) and environment (atmosphere state, contact properties). Hypernetwork conditioning is the right tool *after* the base triple is validated.

These axioms, taken together, **derive** the architecture. They are not design choices.

---

## 3. The Derived Foundational Architecture: LG-PHNO

**Lie-Group Port-Hamiltonian Neural Operator.** The name encodes the axioms:

```
Lie-Group         → A1 (manifold state, equivariance)
Port-Hamiltonian  → A2, A3 (symplectic + dissipation + control port)
Neural Operator   → A5 (field coupling for atmosphere/contact)
```

### 3.1 Forward Model

```
ẋ = [J(x) − R(x; θ_R)] · ∇_x H(x, z; θ_H) + g(x; θ_g) · u(t)
    └───────────────┘   └────────────────┘   └──────────────┘
       (J − R) on TxM   Hamiltonian on M      Control port
```

where:

- `x ∈ T*M` with `M` a Lie group (`SE(3)`, `S²×SO(3)`, etc.)
- `J(x)` is the canonical Poisson structure on `T*M` — hardcoded per domain, **not learned**. For `SE(3)` this is the Lie-Poisson bracket; for `S²×ℝ³×SO(3)` it's the corresponding bundle structure.
- `R(x; θ_R) = L(x) L(x)ᵀ`, Cholesky-parameterized, guaranteed PSD. Existing in [port_hamiltonian.py:145-178](/Users/adi/idex/ai/v3/port_hamiltonian.py#L145-L178).
- `H(x, z; θ_H)` is the Hamiltonian, conditioned on latent code `z`.
- `g(x; θ_g)` is the input port matrix. Existing in [port_hamiltonian.py:203-220](/Users/adi/idex/ai/v3/port_hamiltonian.py#L203-L220).
- `u(t)` is inferred, not observed.

### 3.2 The Five Submodules

**Submodule 1 — Lie-group symplectic integrator.** Replace Cartesian Yoshida-4 with a Lie-group variant (coadjoint-orbit-preserving). On `SE(3)`, use Munthe-Kaas / RKMK with Forest-Ruth coefficients. On `S²`, use geodesic symplectic-Euler. The PyTorch trainer currently broken at [train.py:142-177](/Users/adi/idex/ai/neural_ode/train.py#L142-L177) is replaced wholesale.

**Submodule 2 — Dissipative port-Hamiltonian core.** Existing [PortHamiltonianSystem](/Users/adi/idex/ai/v3/port_hamiltonian.py) with three additions:
- `H` conditioned on latent `z` (hypernetwork-style concatenation at input layer; do not yet generate weights — defer until Phase 3).
- Input-convex constraint on `H` for robot regime (guarantees passivity under PD control).
- Drop the `relu` in R_net ([port_hamiltonian.py:115](/Users/adi/idex/ai/v3/port_hamiltonian.py#L115)) — `relu` kills gradient symmetry; use `softplus` for positive diagonal.

**Submodule 3 — Residual-inverse SINDy for `u(t)`.** Demote classical SINDy from forward-dynamics discovery to control inference:
```
residual r(t) = ẋ_obs − (J − R)∇H   ≈  g(x)·u(t)
```
Regress `u(t)` against a library of control primitives (bang-bang, sinusoid, ramp, skip-glide modes). The existing [sliding_sindy.go](/Users/adi/idex/pkg/physics/sliding_sindy.go) window + STLSQ logic is reused, but against a different target. Add bootstrap CIs (100× resample, refit).

**Submodule 4 — Neural atmosphere operator.** Implement `ρ_nn(h, F10.7, Ap, lat, lon, DOY)` as a small FNO over altitude, fit to NRLMSISE-00 tables generated from [pkg/physics/reference/nrlmsise-00.c](/Users/adi/idex/pkg/physics/reference/nrlmsise-00.c). Enforce hydrostatic equilibrium `dp/dh = −ρg` as a **projection step** after each gradient update, not a penalty. Wire into HGV drag term through the differentiable operator boundary. The placeholder [q/core/hno.py](/Users/adi/idex/q/core/hno.py) becomes this submodule's home.

**Submodule 5 — Augmented Lagrangian trainer.** Replace [StructurePreservingLoss](/Users/adi/idex/ai/neural_ode/train.py#L93-L123):
```
L(θ, μ, ρ) = L_obs(θ) + μᵀ c(θ) + (ρ/2) ‖c(θ)‖²
μ_{k+1} = μ_k + ρ · c(θ_k)           (dual ascent)
```
Constraints `c(θ) = 0`:
- `c₁`: `Var[H(trajectory)] = 0` on conservative segments
- `c₂`: `Var[‖r × v‖] = 0` on orbital segments
- `c₃`: `∫(dp/dh + ρg)² dh = 0` on atmosphere model
- `c₄`: `λ_min(R) ≥ 0` via Cholesky (structurally satisfied)
- `c₅`: `Var[Casimirs]` on Lie-Poisson manifold

Guarantees exact satisfaction at convergence.

### 3.3 Features Unlocked by the Architecture

| Feature | Enabled by | Status today |
|---|---|---|
| HGV skip-glide tracking | PHS + SINDy residual for σ(t) | Impossible (GHNN is conservative-only) |
| Zero-day maneuver adaptation | Hidden-input `u(t)` inference | Not possible |
| Provable energy non-creation | Cholesky PSD `R` | Partial (only in JAX path) |
| Cross-domain transfer (orbit↔robot) | Lie-group invariant formulation | Not possible |
| Symbolic certificate with true confidence | Bootstrap CI + conformal prediction | Placeholder only |
| Atmospheric real-time adaptation | Neural operator + F10.7/Ap live feed | Hardcoded defaults |
| Physics violations mathematically impossible | Symplectic integrator + Aug Lagrangian | Soft penalty only |
| Foundation behavior (one model, many regimes) | Latent-conditioned H, R, g | Per-regime training |
| Falsifiable forward prediction | VSC with 60s prediction + verify-at | Not implemented |
| Edge deployment under thermal | PAX tier selection | Prior art — keeps working |

---

## 4. Gap Analysis: Current → Target

Ranked by leverage (biggest architectural win per unit effort):

### Tier 1 — Structural correctness (required before anything else)

1. **Fix the broken symplectic integrator in PyTorch trainer.** [train.py:142-177](/Users/adi/idex/ai/neural_ode/train.py#L142-L177) either imports the correct Forest-Ruth from [jax_backend.py:510-552](/Users/adi/idex/q/core/jax_backend.py#L510-L552) (via a JAX↔Torch bridge) or reimplements it natively with q/p split and `d` coefficients. This is the single highest-leverage change: the trainer currently produces weights that the architecture cannot honor.

2. **Unify on port-Hamiltonian as the default.** Promote [PortHamiltonianSystem](/Users/adi/idex/ai/v3/port_hamiltonian.py) to the canonical model. Deprecate pure-GHNN as the special case `R = 0, g = 0`. The PyTorch GHNN path in [ghnn.py](/Users/adi/idex/ai/neural_ode/ghnn.py) should be replaced with a Torch port of PHS, or the training pipeline moved to JAX.

3. **Replace `StructurePreservingLoss` with Augmented Lagrangian.** ~200 lines. Dual variable `μ` tracked per constraint, updated each epoch. Fixed `λ_h=0.05` is the structural weakness; replacing it eliminates the largest source of physics violation.

4. **Wire the HGV model into PHS form.** [hgv_physics.go](/Users/adi/idex/pkg/hypersonic/hgv_physics.go) stays as ground-truth data generator, but the learned model becomes PHS-on-`S²×ℝ³×SO(3)` with drag absorbed into `R` and bank-angle absorbed into `g·u`. This is the change that unblocks zero-day HGV tracking.

### Tier 2 — Foundation behavior (earns the "foundation" label)

5. **Latent conditioning of H.** Pass `z ∈ ℝ^d` (regime + environment code) as additional input to H_net. Train jointly on orbital + HGV + ballistic data with a contrastive regularizer on `z`. Do *not* generate weights via hypernetwork yet — latent-input conditioning is sufficient and far more stable to train.

6. **Neural atmosphere operator with hydrostatic projection.** Replace [q/core/hno.py](/Users/adi/idex/q/core/hno.py) placeholder with `ρ_nn` operator, projection-enforced hydrostatic equilibrium. Source NRLMSISE-00 data via the C reference already in the tree.

7. **SINDy as residual control inference.** Modify [sliding_sindy.go](/Users/adi/idex/pkg/physics/sliding_sindy.go) to regress on `r(t) = ẋ − (J−R)∇H` rather than `ẋ`. Library becomes control primitives, not polynomial dynamics. Add bootstrap CIs.

### Tier 3 — Verifiability layer

8. **Symbolic certificate struct + pipeline.** New file `pkg/physics/certificate.go`. Struct drafted in [model_arch.md §5](/Users/adi/idex/ai/model_arch.md). Include bootstrap CI, physics-grade (A/B/C/F), 60-second forward prediction with σ envelope, Ed25519 signature.

9. **Conformal prediction wrapper.** Wrap model outputs with conformal intervals calibrated on held-out trajectories. Provides distribution-free coverage guarantees for the certificate.

10. **Adaptive IMM transition matrix.** Replace the hardcoded Markov in [hgv_imm_filter.go](/Users/adi/idex/pkg/hypersonic/hgv_imm_filter.go) with online Dirichlet update from observed residual statistics. Zero-day maneuvers can open a new mode under sufficient evidence.

### Tier 4 — Deployment (only after Tiers 1-3 hold)

11. **Hypernetwork atlas.** Now justified: base model is correct, so generating `{θ_H, θ_R, θ_g}` from `z` produces *valid* physics instances. Structured pruning + INT8 per regime, ONNX export.

12. **Phase-Adaptive Execution.** Thermal-governed tiering from [model_arch.md §4](/Users/adi/idex/ai/model_arch.md). Pure engineering; orthogonal to the model itself.

---

## 5. Development Phases

**Phase 0 — Audit and align (1 week).**
- Read and reconcile JAX and PyTorch PHS implementations.
- Write a shared spec for `(J, R, H, g)` interfaces that both backends implement.
- Decide canonical training backend (recommend JAX for autodiff-on-gradient cleanliness; PyTorch for deployment).

**Phase 1 — Structural correctness (3 weeks).**
- Ship fixes 1–4 from Tier 1.
- Acceptance: on Keplerian test trajectories, energy drift < 1e-8 over 10⁴ steps (currently Euler-broken); on HGV test trajectories, ground-truth drag recoverable via `R(x)` within 5%.

**Phase 2 — Operator layer & residual SINDy (3 weeks).**
- Ship fixes 5–7.
- Acceptance: atmosphere operator agrees with NRLMSISE-00 within 3% across 30-500 km, live F10.7/Ap feed integrated; on synthetic HGV with known `σ(t)`, SINDy recovers bank-angle signal with R² > 0.9 and bootstrap 95% CI width < 0.1.

**Phase 3 — Verifiability (2 weeks).**
- Ship fixes 8–10.
- Acceptance: certificate generation in <50ms per window; conformal intervals achieve nominal coverage (90%) on held-out; IMM mode count grows adaptively in zero-day scenarios.

**Phase 4 — Foundation behavior (4 weeks on H100).**
- Generate 10⁶ HGV + 10⁵ orbital + 10⁴ ballistic trajectories in Zarr.
- Train LG-PHNO with latent conditioning across all regimes.
- Acceptance: single-checkpoint generalization within 10% across regimes vs regime-specific baselines.

**Phase 5 — Deployment (2 weeks).**
- Hypernetwork atlas if still justified after Phase 4 (may not be).
- ONNX quantization, PAX controller, thermal governor.

**Total: ~15 weeks of focused work.** Most Phase 1 work uses code already present but not wired.

---

## 6. Critical Files

### Modify
- [ai/neural_ode/train.py](/Users/adi/idex/ai/neural_ode/train.py) — replace broken "Yoshida-4" block (L142-177); replace `StructurePreservingLoss` (L93-123) with Augmented Lagrangian.
- [ai/v3/port_hamiltonian.py](/Users/adi/idex/ai/v3/port_hamiltonian.py) — add latent-code `z` input to H_net; replace `relu` in R_net with `softplus`; add Lie-group variant of integrator.
- [pkg/physics/sliding_sindy.go](/Users/adi/idex/pkg/physics/sliding_sindy.go) — switch target from `ẋ` to residual `r(t)`; add bootstrap CI.
- [pkg/hypersonic/hgv_imm_filter.go](/Users/adi/idex/pkg/hypersonic/hgv_imm_filter.go) — online Dirichlet update on transition matrix; compute Kalman gain from covariance (remove hardcoded 0.3).
- [pkg/physics/atmosphere.go](/Users/adi/idex/pkg/physics/atmosphere.go) — wire F10.7/Ap to live feed, route through neural operator.
- [q/core/hno.py](/Users/adi/idex/q/core/hno.py) — promote from placeholder to `ρ_nn` atmosphere operator with hydrostatic projection.

### Create
- `ai/v3/lg_phno.py` — Lie-group port-Hamiltonian model (supersedes PHS).
- `ai/v3/augmented_lagrangian.py` — constraint manager, dual ascent, projection operators.
- `ai/v3/residual_sindy.py` — control-inference SINDy wrapper.
- `pkg/physics/certificate.go` — Verifiable Symbolic Certificate struct + pipeline.
- `pkg/physics/sindy_bootstrap.go` — bootstrap CIs for SINDy coefficients.
- `pkg/physics/conformal.go` — conformal prediction wrapper.
- `ai/v3/neural_atmos.py` — NRLMSISE-00 operator surrogate.
- `ai/v3/latent_encoder.py` — regime/environment → `z` encoder.

### Reuse (existing, correct, underused)
- [q/core/jax_backend.py:510-552](/Users/adi/idex/q/core/jax_backend.py#L510-L552) — proper Forest-Ruth 4th order.
- [ai/v3/port_hamiltonian.py](/Users/adi/idex/ai/v3/port_hamiltonian.py) — Cholesky PSD R construction.
- [pkg/physics/sindy.go:353-417](/Users/adi/idex/pkg/physics/sindy.go#L353-L417) — real R² computation.
- [pkg/physics/reference/nrlmsise-00.c](/Users/adi/idex/pkg/physics/reference/nrlmsise-00.c) — atmosphere ground truth.
- [pkg/ml/hybrid/orchestrator.go](/Users/adi/idex/pkg/ml/hybrid/orchestrator.go) — fallback routing.
- [sim/ganga_mjx/symplectic_mjx.py](/Users/adi/ganga1.0/sim/ganga_mjx/symplectic_mjx.py) — cross-repo reference for Lie-group MuJoCo integration.
- [sim/ganga_mjx/energy_auditor.py](/Users/adi/ganga1.0/sim/ganga_mjx/energy_auditor.py) — `E_hal` pattern for runtime physics violation detection.

---

## 7. Verification Plan

**Unit-level.**
- Symplecticity: rollout 10⁴ steps of Keplerian orbit, `|ΔH/H| < 10⁻⁸`.
- PSD R: random states, assert `λ_min(R(x)) ≥ 0` for 10⁶ samples.
- Passivity: assert `dH/dt + ∇H ᵀ R ∇H − ∇H ᵀ g u = 0` numerically.
- Augmented Lagrangian convergence: constraint residual `‖c(θ_k)‖ → 0` as `k → ∞`.

**Module-level.**
- Atmosphere operator: Frobenius error vs NRLMSISE-00 < 3% across 30-500 km, 9 months of F10.7/Ap history.
- Residual SINDy: on synthetic HGV trajectories with known σ(t) profile, recover signal with R² > 0.9 and CI coverage > 90%.
- IMM adaptation: feed a never-before-seen maneuver signature; verify a new mode opens within 30 observations.

**Integration-level.**
- End-to-end HGV corridor test: simulate a Siliguri-corridor HGV with NRLMSISE-00 truth atmosphere, feed noisy radar observations, assert track accuracy < 500 m CEP over 5-minute window.
- Cross-regime: single checkpoint, evaluate on LEO + HGV + ballistic held-out sets; within 10% of regime-specific baselines.
- Certificate pipeline: generate VSC, forward-predict 60 s, measure fraction of predictions within stated σ envelope; expect ≥ 90%.
- Edge thermal: sustained 100 Hz inference on M2 Air for 10 min; PAX governor holds T < 85°C.

**Regression guard.**
- Existing orbital maneuver detection benchmarks must not regress. [pkg/ml/hybrid/orchestrator.go](/Users/adi/idex/pkg/ml/hybrid/orchestrator.go) fallback path stays live during migration; new LG-PHNO output is shadow-compared for N days before promotion.

---

## 8. TL;DR

The foundation is **Lie-Group Port-Hamiltonian Neural Operators**, trained with Augmented Lagrangian under hard symplectic integration, with SINDy as residual-inverse control inference and Fourier Neural Operators reserved for infinite-dimensional fields (atmosphere, contact). Every primitive needed exists in fragments across idex and ganga1.0; the work is wiring, unification, and replacing soft penalties with hard constraints — not new ML research. The hypernetwork atlas and PAX tiering proposed in [model_arch.md](/Users/adi/idex/ai/model_arch.md) are correct but belong in Phase 4–5, not Phase 1. The biggest single lever is fixing the broken symplectic integrator in the PyTorch trainer, which today undermines every downstream claim about conservation.
