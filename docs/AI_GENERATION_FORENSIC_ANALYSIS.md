# Forensic Analysis: AI-Generated Code Detection in `idex`

**Date:** 2026-04-26  
**Scope:** Full repository — Go runtime (`pkg/`, `cmd/`, `engine/`), Python ML (`ai/`, `q/`), docs, git history.  
**Method:** Static analysis, git forensics, dependency verification, style heuristics. Read-only.

---

## 1. Verdict

**Yes — overwhelmingly AI-generated, with thin human cleanup.** Confidence: Very High (>95%).

Multiple independent signals converge: non-human git velocity, hallucinated dependency, stub-as-implementation, broken primitives, cross-language style bleed, linter-chasing commits, and internal docs that read like an LLM auditing itself.

---

## 2. Quantitative Summary

| Signal | Metric |
|--------|--------|
| Total Go + Python LOC | ~235,824 |
| Repository age | 6 days (2026-04-18 → 2026-04-24) |
| Largest commit | +50,356/−2,014 lines, 220 files (`8e6ee71`) |
| 2nd largest | +32,311/−31,842 lines, 263 files (`727e562`) |
| Stub files (`*_stub.go`) | 12 in `pkg/ml/onnx/` and `pkg/ml/lightgbm/` |
| TODO/stub/placeholder in `pkg/` | 472 matches across 134 Go files |
| Hallucinated PyPI dep | `astrojax` (does not exist) |
| Non-idiomatic JavaDoc blocks | Pervasive in `pkg/neural/`, `pkg/collision/`, `pkg/physics/` |
| Separate `requirements.txt` | 9 files, no shared lockfile |
| Authors | `adityamourya-hub` (descriptive), `Manish Maurya` (vague cleanup) |

---

## 3. Evidence Taxonomy

### 3.1 Git Velocity — Physically Impossible for Solo Human

- `8e6ee71` (2026-04-20): **220 files, +50,356/−2,014** — bulk generation event.
- `727e562` (2026-04-22): **263 files, +32,311/−31,842** — larger than most engineers write in a month.
- `c6f937a` (2026-04-23): **20 files, +3,467**.
- `6dd8bf4` (2026-04-22): **22 files, +4,226/−856**.

**Cleanup pattern:** ~15 micro-commits on 2026-04-21 titled `"Fix unused parameter warnings in X.go"` — classic "run linter, feed errors back to LLM, re-commit" loop. A human would batch these or fix inline.

### 3.2 Hallucinated Dependency: `astrojax`

`@/Users/adi/idex/ai/v3/astrojax_world_model.py:13-20`:
```python
from astrojax import propagators, forces, constants
```

`astrojax` is **not on PyPI**. It is a juggling toy / 2008 Georgia Tech nonlinear-dynamics paper. The LLM conflated it with JAX orbital mechanics. The entire 530-line module is scaffolded on this fabricated import; the `try/except` branch is performative since the `_mock_propagate()` path is the only executable surface.

### 3.3 Broken Primitives

`randFloat()` returns a **constant**:
```go
@/Users/adi/idex/pkg/neural/hnn_simple.go:474-477
func randFloat() float64 {
    return float64(123456789) / float64(2147483647) // Placeholder
}
```
Called by `randomPosition()` and `randomVelocity()` inside `GenerateTrainingData()` — every "random" training sample is identical.

`mathRand()` returns **0.5**:
```go
@/Users/adi/idex/pkg/collision/surrogate.go:400-403
func mathRand() float64 {
    return 0.5 // Placeholder
}
```

### 3.4 Stub-as-Implementation

12 `*_stub.go` files exist for headline capabilities (Axiom-Zero, rapid ranker, POL transformer, blindspot forecaster, etc.). Every function returns `fmt.Errorf("...not yet implemented")`. Example:

```go
@/Users/adi/idex/pkg/ml/onnx/rapid_ranker_stub.go:28-30
return physics.RapidRankInference{}, fmt.Errorf(
    "onnx rapid ranker backend not enabled in this build; model=%s", m.ModelPath)
```

DPHNO — the central architectural claim — has a test that skips:
```go
@/Users/adi/idex/pkg/ml/dphno/nasa42_validation_test.go:46
t.Skip("DPHNO not yet implemented - see ai/model_arch.md for architecture")
```

### 3.5 Cross-Language Style Bleed

Pervasive JavaDoc `/** */` blocks in Go, e.g. `@/Users/adi/idex/pkg/neural/hnn_simple.go:8-15`. Idiomatic Go uses `//`. This is an LLM trained on Java/C++/Python defaulting to `/** */` for "important" doc blocks.

### 3.6 Self-Incriminating Internal Docs

`@/Users/adi/idex/ai/model_arch.md` and `ai/foundation.md` contain line-numbered bug inventories:
- `pkg/physics/sindy.go:353-357` — hardcoded R²=0.95
- `ai/v3/astrojax_world_model.py:140-144` — Euler integration (docs lag code; current source uses RK4)
- `pkg/hypersonic/hgv_imm_filter.go:196` — Kalman gain hardcoded 0.3
- `pkg/physics/atmosphere.go:22` — fixed F10.7=150, Ap=15

A human architecture review does not enumerate its own bugs with line numbers inside the design document. This is an LLM prompted to "review the code and write a technical analysis," output committed verbatim.

### 3.7 Marketing Prose Over Stubbed Runtime

- **"Sovereign Handshake"** — Hamiltonian hash exists in `sindy.go` but no gRPC endpoint emits it (`engine/grpc_server.go` has empty `processQueryCommand` placeholders).
- **"Quantum-immune Majorana zero-mode"** — zero quantum computing code in the repo.
- **"10⁻¹⁵ energy conservation"** — HNN propagator uses **first-order Euler** (`@/Users/adi/idex/pkg/neural/hnn_simple.go:181-188`).
- **"Air-Gapped Ready"** — Docker Compose binds external ports; gRPC/REST listeners are active.

### 3.8 Domain Incoherence

- "Astrojax" as JAX orbital library (see §3.2).
- "Majorana zero-mode" in SDA — condensed-matter concept with zero orbital relevance.
- "PAX 4-tier thermal governor" — mentioned in docs, not wired to hardware telemetry.
- 9 separate `requirements.txt` files — typical of per-prompt module generation.

### 3.9 Vague Human Commit Messages

`Manish Maurya` commits: `"fixing issues in the code"`, `"fixing the code issues"`, `"code optimisation, ML integration ,corner cases fix"`. These bracket AI-bulk commits from `adityamourya-hub` with polished architectural messages. Consistent with human reviewing generated code, running builds, making minimal fixups.

---

## 4. Counter-Evidence

1. **Real `go.mod`** with resolvable imports (`gonum`, `urfave/cli/v2`).
2. **Some tests compile and run**: `sindy_test.go`, `rkf45_test.go`, `integrator_test.go` contain real assertions.
3. **ONNX files exist** (`tier2_full.onnx` 50MB, `tier1_pruned.onnx` 8MB) — binary artifacts, possibly real exports. But **no training logs, checkpoints, or reproducible scripts** produce them.
4. **PDF references** (`UK_RTO.pdf`, `fm6_02.pdf`) appear genuine human uploads.
5. **Makefile, `.golangci.yml`, GitHub Actions** — standard human tooling.

**Assessment:** Human-curated scaffold (module layout, PDFs, CI, Makefile) + ~90% LLM-generated source, with shallow human cleanup.

---

## 5. Conservative Remediation (Flag, Don't Delete)

Stop the bleeding while preserving architecture vocabulary.

### 5.1 Fix Broken Primitives
- Replace `randFloat()` constant with `math/rand` or `crypto/rand`. Add seed parameter.
- Replace `mathRand()` constant with `rand.Float64()`.

### 5.2 Remove Hallucinated Dependency
- Delete `try/except astrojax` block in `ai/v3/astrojax_world_model.py`.
- Rename `AstrojaxWorldModel` → `OrbitalWorldModel` or `MockWorldModel`.
- Delete `_init_astrojax_forces()` (dead code).
- Rename `_mock_propagate` → `propagate_orbit_rk4()` and document it as simplified 2-body+J2 only.

### 5.3 Make Stubs Explicit
- Prefix every `*_stub.go` function with `// STUB: returns error until ONNX runtime bound`.
- Replace `t.Skip("...not yet implemented")` with `t.Fatal("...not yet implemented — must pass before merge")` so CI surfaces gaps.

### 5.4 Add Implementation Status to README
Insert a table at top: **Working** / **Stub** / **Research Claim** for every capability.

### 5.5 Strip Marketing from Technical Docs
- Remove "Sovereign Handshake", "Quantum-immune Majorana", "Perpetual ownership", "Zero foreign kill-switch" unless backed by tested code.
- Retain physics/math content (Port-Hamiltonian, symplectic, SINDy) as legitimate domain concepts.

---

## 6. Aggressive Remediation (Strategic Prune)

Reduce repo to **honest-state**: every file compiles, runs, and is tested; every README claim has a code path.

### 6.1 Quarantine Stub Layer
- Move all `pkg/ml/onnx/*_stub.go` to `pkg/ml/onnx/disabled/` with explanatory README. Delete build-tag variants from main tree. If ONNX is never bound, delete the stub layer entirely.

### 6.2 Delete `astrojax_world_model.py`
- `@/Users/adi/idex/ai/v3/astrojax_world_model.py` (530 lines on fabricated dependency). If a differentiable world model is needed, rewrite using `diffrax` or `jax.numpy`.

### 6.3 Prune Unused `ai/v3/` Modules
Modules with **zero callers** in `pkg/` or `cmd/` (use `pyright`/`vulture` or grep for imports):
- `e3nn_hamiltonian.py`, `lie_phs.py`, `federated_learning.py`, `material_fingerprint.py`
- `maya_detector.py`, `mbpo_agent.py`, `mbpo_hardened.py`, `mbpo_se3_agent.py`
- `pol_patchtst_jepa.py`, `rns_relay.py`, `supcon_anchor_loop.py`, `supcon_trainer.py`
- `symbolic_transformer_sr.py`, `volumetric_hash_inr.py`, `blind_source_separation.py`

Move to `experiments/` or delete.

### 6.4 Audit ONNX Artifacts
- `@/Users/adi/idex/pkg/ml/pax/models/tier2_full.onnx` (50MB) and `tier1_pruned.onnx` (8MB): check SHA against any training log. If no reproducible script produces them, delete.
- `@/Users/adi/idex/ai/v3/axiom_se3_ghnn.onnx` (28KB): inspect with Netron. If untrained/random weights, delete.

### 6.5 Collapse Requirements
- Merge 9 `requirements.txt` files into one top-level `pyproject.toml` with a resolvable lockfile (`poetry.lock` or `uv.lock`). Remove duplicates and conflicts.

### 6.6 Rewrite README
- From 687 lines to ~100 lines. Describe only what compiles, runs, and has passing tests.
- Move marketing/strategy content to a separate `docs/STRATEGIC_VISION.md` labeled **not implemented**.

### 6.7 Delete or Move Marketing Docs
- `docs/COMPREHENSIVE_ARCHITECTURE_ANALYSIS.md`, `docs/HARD_CONSTRAINT_ARCHITECTURE.md`, `docs/INTERPRETABILITY_ARCHITECTURE.md` — if they describe unimplemented systems, move to `docs/research/` or delete.
- Keep `docs/RKF45_IMPLEMENTATION.md`, `docs/HYBRID_INTEGRATOR_IMPLEMENTATION.md` if they match working code.

---

## 7. Conclusion

The `idex` repository is a **sophisticated AI-generation artifact** — conceptually ambitious, mathematicically literate, and structurally dishonest. It contains:
- ~236k lines of mostly plausible Go and Python,
- 12 explicitly named stub files and 472 implicit stub points,
- A hallucinated dependency (`astrojax`) that anchors a 530-line module,
- Broken random-number primitives that would silently corrupt training data,
- Marketing prose that outpaces runnable code by an order of magnitude.

The **human contribution** appears to be: repository scaffolding, PDF references, CI/tooling config, and shallow lint cleanup. The **AI contribution** is the bulk of the source code, internal architecture docs, and commit messages.

**Recommended next action:** Choose either the **Conservative** track (minimal patches, preserve vocabulary) or the **Aggressive** track (prune to honest-state), then execute systematically. Do not present this repository to stakeholders or funders in its current form without the Implementation Status table (§5.4) prominently displayed.
