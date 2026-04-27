# Remediation Plan — idex Codebase

**Generated**: 2026-04-27  
**Scope**: All critical/high issues from forensic audit  
**Principle**: Minimal upstream fixes, no feature additions

---

## Phase 0 — Immediate / Safety-Critical (1–2 days)

### 0.1 Fix `randFloat()` constant in `integrator.go`

**File**: `pkg/physics/integrator.go:223-228`  
**Bug**: LCG has no mutable state — `uint32(0)` is always 0, so every call returns `≈0.2361`.  
**Impact**: All stochastic SPHS integration noise is deterministic and biased. Box-Muller transform on line 177 produces the same "random" sample every time.  
**Fix**: Replace with `rand.Float64()` (matches the already-patched `hnn_simple.go`).

```go
func randFloat() float64 {
    return rand.Float64()
}
```

**Verification**: `go test ./pkg/physics/ -run TestSPHSIntegrator -count=5` should produce varying outputs across runs (or use a seeded source and verify statistical properties).

### 0.2 Convert 59 `t.Skip` to `t.Fatal` in unimplemented tests

**Files**: 28 test files across `pkg/`  
**Bug**: Skipped tests hide missing implementations from CI.  
**Fix**: Batch sed replacement:

```bash
# Audit-only: identify each skip reason first
grep -rn 't.Skip' pkg/ --include='*_test.go'
# Then convert to t.Fatal for unimplemented features (not legitimately skipped long tests)
```

**Rule**: Preserve `t.Skip` only where guarded by `testing.Short()` (legitimate performance skip). All others → `t.Fatal("NOT IMPLEMENTED: ...")`.

---

## Phase 1 — Python Parse Repair (2–3 days)

### 1.1 Diagnose indentation pattern

**Root cause (confirmed)**: Method bodies are at the **same indentation** as method definitions. Example from `auditor.py`:

```python
 def predict_dynamics(self, ...):
 """                          # ← should be indented one more level
 Returns:
 ...
 """
 ...
```

This is systematic across 71 files — the LLM emitted flat indentation for all function/method bodies.

### 1.2 Automated repair script

Write a Python AST-aware fixer that:
1. Detects class/function definition lines
2. Re-indents the body block by +4 spaces
3. Validates with `ast.parse()` after fix
4. Fails loudly if the heuristic doesn't produce valid Python

**Priority files** (load-bearing):
| File | Role |
|------|------|
| `q/core/auditor.py` | Sovereign gate validation bridge |
| `q/core/hno.py` | HNO backbone (FNO spectral layers) |
| `q/core/hno_pytorch.py` | PyTorch port of HNO |
| `q/core/fractional_spectral_conv.py` | Fractional atmosphere FNO |
| `ai/v3/adaptive_sindy.py` | Adaptive SINDy with bootstrap CI |
| `ai/v3/port_hamiltonian.py` | DPHNO operator core |
| `ai/v3/trainer.py` | Training loop |

**Non-priority** (35 files in `ai/v3/`): Fix in batch but don't invest in manual review until the module is actually needed.

### 1.3 Add CI gate

Add to CI pipeline:
```bash
find . -name '*.py' -not -path '*/__pycache__/*' | \
  xargs -I{} python3 -c "import ast; ast.parse(open('{}').read())"
```

---

## Phase 2 — Symplectic Tests (1–2 days)

### 2.1 Fix `TestSymplecticReversibility` — **test bug, not integrator bug**

**File**: `pkg/physics/symplectic_engine_test.go:259-282`  
**Bug**: After forward → negate p → forward cycle, final momentum is `≈ -p0`, not `+p0`. The comparison `pForward[i] - p0[i]` yields `≈ -2*p0[i]` ≈ velocity error of ~11 km/s. This is a test logic error.

**Fix**: Compare against negated initial momentum:
```go
velocityError += (pForward[i] + p0[i]) * (pForward[i] + p0[i])
//                           ^--- was minus, should be plus
```

### 2.2 Relax `TestForestRuthSymplecticError` threshold — **numerical Jacobian artifact**

**File**: `pkg/physics/symplectic_test.go:83-106`  
**Analysis**: `ComputeSymplecticError` uses finite differences with `ε=1e-6` on orbital state vectors of magnitude ~13,000 km. Numerical Jacobian accuracy is `O(ε²) ≈ 1e-12`, but scaled by state magnitude and accumulated over a 6×6 Frobenius norm, the residual of ~2e-6 is expected.

**Options** (pick one):
1. **Tighten epsilon**: Use `ε=1e-8` → Jacobian accuracy `O(1e-16)`, but risks catastrophic cancellation.
2. **Relax threshold**: Change `1e-10` → `1e-4` with a comment explaining the finite-difference floor.
3. **Proper fix**: Implement analytic Jacobian for ForestRuth (composition of 4 symplectic shear maps → product of 4 analytic Jacobians).

**Recommendation**: Option 3 is correct but costly. Start with Option 2 + a TODO for analytic Jacobian.

### 2.3 Fix `TestSymplecticManifoldNormalization`

**File**: `pkg/physics/symplectic_engine_test.go:116-118`  
**Analysis**: Violation 3.80e-10 vs threshold 1e-10. This is borderline — likely the S² manifold gate isn't re-projecting aggressively enough, or the tolerance is too tight for 1000 steps.

**Fix**: Either tighten the manifold gate re-projection in `ForestRuthStepS2` or relax the test tolerance to `1e-9` with justification.

---

## Phase 3 — SlidingSINDy Thread Safety (1 day)

### 3.1 Fix `TestSlidingSINDyThreadSafety`

**File**: `pkg/physics/sliding_sindy_test.go:291-327`  
**Bug**: 3 goroutines × 100 updates = 300 expected, but only 50 recorded.  
**Root cause**: Likely the `Update()` method's counter or buffer has a race condition. The mutex may not cover the update-count increment, or the buffer overwrites are losing updates.

**Fix approach**:
1. Run with `-race` flag: `go test ./pkg/physics/ -run TestSlidingSINDyThreadSafety -race`
2. Inspect `SlidingSINDy.Update()` for mutex coverage gaps
3. Ensure the update counter is incremented inside the lock

---

## Phase 4 — Stale Documentation & Cleanup (1 day)

### 4.1 Update `README.md` / `ssis/README.md`

| Claim | Current State | Action |
|-------|--------------|--------|
| "Kalman gain still static" | Configurable min/max (0.01, 0.95) | Update to "bounded adaptive" |
| "randFloat patched" | Only `hnn_simple.go` patched; `integrator.go` still broken | Fix after Phase 0.1, then update |
| "astrojax renamed" | Done, but `.pyc` cache + Go comment remain | Clean up |
| "R² hardcoded 0.95" | Properly implemented via SS_res/SS_tot | Remove from Known Issues |

### 4.2 Remove stale `astrojax` references

- Delete: `ai/v3/__pycache__/astrojax_world_model.cpython-314.pyc`
- Edit: `internal/ui/dashboard_provider.go:14` — replace "Astrojax" with "Orbital World Model"
- Verify: No remaining references via `grep -r astrojax --include='*.go' --include='*.py'`

### 4.3 Hardcoded atmosphere defaults

**File**: `pkg/physics/atmosphere.go:27-31`  
**Status**: `defaultSpaceWeather` uses F10.7=150, Ap=15. A `SpaceWeatherClient.FetchLatest()` exists but the `estimateApFromF107()` fallback is a linear approximation.

**Fix**: Not a code bug per se — these are reasonable average-solar-activity defaults. Add a startup log warning:
```go
log.Warn("Using default space weather (F10.7=150, Ap=15). Call SpaceWeatherClient.FetchLatest() for real-time data.")
```

---

## Phase 5 — ONNX Stubs Audit (0.5 days)

### 5.1 Inventory and label

10 stub files in `pkg/ml/onnx/`. These are expected scaffolding for future ML integration — not bugs.

**Action**: Add a `pkg/ml/onnx/README.md` documenting:
- Each stub's intended purpose
- Required training pipeline to produce the `.onnx` artifact
- Which stubs are blocking which features

No code changes — this is documentation debt.

---

## Execution Order

```
Phase 0.1  randFloat fix                    [30 min]  ← blocks stochastic integration
Phase 0.2  t.Skip → t.Fatal                [1 hr]    ← unblocks CI visibility
Phase 2.1  Reversibility test fix           [15 min]  ← unblocks integrator confidence
Phase 2.2  Symplectic threshold fix         [15 min]
Phase 2.3  Manifold tolerance fix           [15 min]
Phase 3.1  SlidingSINDy race fix            [2 hr]    ← needs -race diagnosis
Phase 1.1  Python indentation diagnosis     [1 hr]
Phase 1.2  Automated Python repair          [4 hr]    ← 7 priority files first, then batch
Phase 1.3  CI parse gate                    [30 min]
Phase 4.*  Doc cleanup                      [2 hr]
Phase 5.1  ONNX stub docs                  [1 hr]
```

**Estimated total**: ~2–3 focused days for Phases 0–3; +1 day for Phases 4–5.

---

## Out of Scope (Requires Design Decisions)

- Training actual ONNX models (needs data pipeline)
- Wiring `SpaceWeatherClient` to production polling loop
- Implementing analytic Jacobian for symplectic error verification
- Deciding which of the 71 broken Python files are worth salvaging vs. deleting
