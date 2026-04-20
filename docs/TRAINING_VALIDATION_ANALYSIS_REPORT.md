# ADITI 4.0 Training & Validation Pipeline Analysis Report
**Analysis Date:** April 20, 2026  
**Scope:** Loss functions, validation hooks, symplectic constraints, training architecture

---

## Executive Summary

### Critical Findings

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| **SupCon Temperature Mismatch** | HIGH | `lc_phs_trainer.py:136` vs `port_hamiltonian.py:753` | Regime disentanglement may fail - 0.07 vs 0.1 |
| **DPHNO Uses Soft Penalty** | HIGH | `dphno.py:263` | Architecture violated: using `0.1 * energy_loss` instead of Augmented Lagrangian |
| **Multiple Competing Loss Classes** | MEDIUM | 3 files with `AugmentedLagrangianLoss` | Inconsistent constraint enforcement |
| **S² Validation Uses Cartesian State** | MEDIUM | `train.py:585-587` | Manifold check `|q|=1` on unnormalized orbital state |
| **dexpinv FIXED - 3rd Order** | RESOLVED | `lie_integrator.py:119-147` | Audit issue resolved - full series expansion implemented |

---

## 1. Loss Function Architecture Analysis

### 1.1 Multiple AugmentedLagrangian Implementations

**File 1:** `ai/v3/physics_constraints.py:17-147`
- Uses **learned** Lagrange multipliers (`nn.Parameter`)
- Proper dual variable updates via gradient ascent
- **Strength:** Correct Augmented Lagrangian formulation

```python
self.lambda_energy = nn.Parameter(torch.tensor(initial_lambda))  # Learned!
```

**File 2:** `ai/neural_ode/train.py:93-168`
- Uses **scalar** Lagrange multipliers (not learned)
- Dual variables updated manually in `with torch.no_grad()`
- **Weakness:** Not integrated into gradient optimization

```python
self.lambda_energy = 0.0  # Scalar, not learned
```

**File 3:** `ai/neural_ode/train.py:314-421` (SymplecticAugmentedLoss)
- Combines Augmented Lagrangian + Symplectic Jacobian penalty
- Scheduled ramp-up for symplectic weight (quadratic ease-in)
- **Unique feature:** Jacobian computation via `torch.vmap` for H100 efficiency

### 1.2 DPHNO Trainer - Architecture Violation

**File:** `ai/v3/dphno.py:225-274`

```python
# Line 263: Using soft penalty instead of Augmented Lagrangian
total_loss = dynamics_loss + 0.1 * energy_loss  # WRONG!
```

**Expected (per architecture decision):**
```python
total_loss = dynamics_loss + energy_augmented  # Augmented Lagrangian
```

**Impact:** DPHNO models trained with soft penalty will have weaker physics constraints at convergence.

### 1.3 Symplectic Loss Implementation

**Strengths:**
- `compute_symplectic_jacobian_loss()` (train.py:240-311) correctly implements M^T * J * M = J check
- Uses `torch.vmap` for batch Jacobian computation (H100 optimized)
- Scheduled ramp-up prevents early training instability

**Weakness:**
- Lambda ramp depends on `current_epoch` tracking (line 338) - may not sync with actual training loop
- Symplectic penalty only applied if `z_initial is not None and dt is not None` - easy to accidentally disable

---

## 2. Supervised Contrastive (SupCon) Loss Analysis

### 2.1 Temperature Mismatch

| File | Temperature | Notes |
|------|-------------|-------|
| `lc_phs_trainer.py:136` | **0.07** | "Standard for disentanglement" |
| `port_hamiltonian.py:753` | **0.1** | Default parameter |

**Impact:** Different temperatures will produce different latent space clustering. τ=0.07 creates sharper clusters; τ=0.1 is more relaxed.

**Recommendation:** Standardize on τ=0.07 (from Khosla et al. paper).

### 2.2 SupCon Implementation Quality

**File:** `port_hamiltonian.py:735-790`

**Strengths:**
- Proper InfoNCE-style formulation
- Self-similarity masking with large negative (-1e9)
- Normalized cosine similarity
- Handles samples without positives gracefully

**Weakness:**
- No temperature annealing during training
- Label comparison creates [batch, batch] matrix - O(n²) memory

---

## 3. Validation Hook Analysis

### 3.1 10k-Step Symplectic Drift Check

**Implementation in `train.py:542-614`:**

```python
# Target thresholds
avg_drift < 1e-8          # Energy drift target
max_constraint_violation < 1e-10  # Manifold integrity target
```

**Implementation in `lc_phs_trainer.py:599-653`:**

```python
# Line 699
print(f"  Symplectic Drift: {avg_drift:.2e} (target < 1e-7)")
```

**Discrepancy:** Different targets (1e-8 vs 1e-7). Should standardize.

### 3.2 S² Manifold Validation Issue

**File:** `train.py:585-587`

```python
# For S²: |q| = 1 (unit sphere constraint)
if current_state.shape[-1] >= 3:
    q_norm = torch.norm(current_state[..., :3], dim=-1)
    q_err = torch.abs(q_norm - 1.0).max().item()
```

**Problem:** For orbital dynamics (LEO), position is NOT on unit sphere - it's ~6878 km from Earth center. The |q|=1 check is only valid for S² manifold (HGV/attitude), not orbital position states.

**Fix needed:** Regime-aware validation:
```python
if regime == "S2":
    q_err = abs(norm(q) - 1.0)  # Unit sphere
elif regime == "SE3":
    q_err = 0.0  # No sphere constraint for orbital
```

---

## 4. Lie Integrator & dexpinv Status

### 4.1 dexpinv - FIXED

**Audit Finding (Critical):** "dexpinv is FIRST-ORDER placeholder"  
**Current Status:** `lie_integrator.py:119-147`

```python
def dexpinv_se3(xi: torch.Tensor, eta: torch.Tensor, order: int = 3) -> torch.Tensor:
    """ADITI 4.0: Full series expansion (3rd order) to fix Lie bracket errors"""
    for k in range(1, order + 1):
        coeff = ((-1) ** k) / math.factorial(k + 1)
        nested = nested_lie_bracket(xi, k, eta)
        correction = correction + coeff * nested
```

**Resolution:** 3rd-order series expansion implemented with nested Lie brackets. Munthe-Kaas correction is now active.

### 4.2 S² Geodesic Integrator

**File:** `lie_integrator.py:252-332`

**Strengths:**
- Trigonometric geodesic update: `q_next = q cos(|v|dt) + (v/|v|) sin(|v|dt)`
- Tangent space projection for momentum: `p_next = p - (p·q)q`
- Renormalization safety: `q = q / (|q| + 1e-10)`

---

## 5. Training-Runtime Gap Analysis

| Training Constraint | Runtime (Go) | Gap |
|---------------------|--------------|-----|
| Augmented Lagrangian (energy, momentum) | ??? | Unknown - check pkg/orbital |
| Symplectic Jacobian (M^T*J*M=J) | RK4 (non-symplectic) | **CRITICAL** |
| Forest-Ruth coefficients | Forest-Ruth | Match ✓ |
| S² geodesic integrator | ??? | Unknown - verify Go side |

**Symplectic Ghost persists:** Python trains with symplectic constraints, Go uses RK4 for state integration per audit.

---

## 6. Recommendations

### Immediate (Pre-May 4)

1. **Fix DPHNO loss** (`ai/v3/dphno.py:263`):
   ```python
   # Replace soft penalty with Augmented Lagrangian
   from physics_constraints import AugmentedLagrangianLoss
   self.criterion = AugmentedLagrangianLoss()
   ```

2. **Standardize SupCon temperature** to 0.07:
   ```python
   # port_hamiltonian.py:753
   def supervised_contrastive_loss(..., temperature: float = 0.07):  # Was 0.1
   ```

3. **Fix S² validation** to be regime-aware:
   ```python
   if regime == "S2":
       q_err = check_unit_sphere(current_state)
   else:
       q_err = 0.0
   ```

### Medium-term

4. **Merge AugmentedLagrangianLoss implementations** - use `physics_constraints.py` version (learned multipliers)

5. **Add validation regime tracking** - pass regime label through validation hook

6. **Standardize drift targets** - use 1e-7 (matches Sovereign Handshake σ_H/H threshold)

---

## Appendix: File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `ai/neural_ode/train.py` | Main training pipeline | `SymplecticAugmentedLoss`, `validate_lie_integrator` |
| `ai/v3/physics_constraints.py` | Hard constraint losses | `AugmentedLagrangianLoss` (learned multipliers) |
| `ai/v3/port_hamiltonian.py` | PHS + SupCon | `supervised_contrastive_loss`, `Yoshida4Integrator` |
| `ai/v3/dphno.py` | DPHNO model + trainer | `DPHNOTrainer` (needs fixing) |
| `ai/v3/lc_phs_trainer.py` | LC-PHS trainer | `PyTorchLCPHSTrainer`, `validate_symplectic_drift` |
| `ai/v3/lie_integrator.py` | Geometric integrators | `dexpinv_se3` (fixed!), `s2_geodesic_step` |
