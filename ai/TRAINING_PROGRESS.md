# IDEX ML Training — Execution Progress

Tracks implementation progress against the phased plan at
`/Users/manishmaurya/.windsurf/plans/idex-ml-phased-training-plan-99f0d3.md`.

## Day-1 Upstream Fixes (plan §4, step 1) — COMPLETE

These are the highest-leverage single-line/upstream fixes that unblock every
subsequent phase. Each was verified before edit.

| # | Target | Status | Notes |
|---|---|---|---|
| 1 | `pkg/orbital/hpop.go:100` RK4 → RKF45 | **Done** | `Propagate` now calls existing `rkf45Integrate` (adaptive Cash-Karp, in-file at L331) with fallback to RK4 on underflow. Tolerances read from `HPOPConfig.Tolerance`. `gofmt` / `go vet ./pkg/orbital/` clean. |
| 2 | `ai/neural_ode/train.py` fake Yoshida-4 → real Forest-Ruth | **Done** | Replaced the deprecated stub at L205 that silently aliased `implicit_midpoint_step`. New implementation at L198-277 does true Forest-Ruth q/p alternation with c/d coefficients lifted from `q/core/jax_backend.py:461-469` (bit-identical to JAX backend). Explicit, no fixed-point loop, gradient-safe. |
| 3 | `ai/v3/port_hamiltonian.py` `relu` → `softplus` on R diagonal | **Already upstream** | JAX side: `R_net` MLP activation at L236, diagonal softplus at L358. PyTorch twin `ai/neural_ode/port_hamiltonian_pytorch.py`: MLP L87/L89, diagonal L185. Both use `softplus + 1e-6` for strict dissipativity. No edit needed. |
| 4 | `pkg/physics/sindy.go` hardcoded R²=0.95 removal | **Already upstream** | `computeRSquared` at L562-626 does real SS_res/SS_tot with [0,1] clamp. The 0.95 occurrences in the codebase are legitimate thresholds (`certificate.go:96 MinSINDyConfidenceC`, `sliding_sindy.go:120 DefaultConvergenceRatio`), not faked fit metrics. |

## Training Script Fixes — COMPLETE

| # | Target | Status | Notes |
|---|---|---|---|
| 5 | `scripts/train_all_sequential.sh:94-98` `weight_exists_in_gcs` exit code | **Done** | Added `return $?` to properly propagate `gsutil ls` exit status (0 if file exists, 1 if not). Skip guards now work correctly. |
| 6 | `scripts/train_all_sequential.sh:287-367` collision phase | **Done** | Restored full training flow (was hardcoded skip). Added consistent skip guard (weights or completion marker), check for `.collision_skipped` marker to respect prior intent, full dataset download/preprocessing/training pipeline. Bash syntax validated. |

## Phase Status

- [x] **Day-1 fixes** (integrators, activations, R²)
- [x] **Training script fixes** (skip guards, collision phase)
- [ ] **P0** Neural Atmosphere FNO — blocked on NRLMSISE-00 CGo data generator @ 1e7 scale
- [ ] **P1** SE(3)-GNN + DPHNO orbital foundation
- [ ] **P2** Hypersonic DPHNO + neural SINDy residual-inverse control
- [ ] **P3** 10 task heads on frozen P0–P2 backbone (retrofit `scripts/train_all_sequential.sh`)
- [ ] **P4** Policies / RL / adversarial hardening
- [ ] **P5** Conformal UQ + VSC certificate pipeline
- [ ] **P6** INT8/ONNX + PAX tier deploy

## Outstanding Follow-ups

- `go build ./...` currently fails globally on missing `libzmq` system dep
  (transitive via `pebbe/zmq4`). Not caused by Day-1 edits but will block CI;
  add `brew install zmq` / `apt install libzmq3-dev` to onboarding.
- PyTorch training pipeline (`ai/neural_ode/train.py`) is flagged deprecated in
  its module docstring (JAX `ai/v3/train_phs_jax.py` is canonical). Forest-Ruth
  fix applied to both paths so downstream deploy/ONNX export stays symplectic.

## Next Step

Begin **P0 Neural Atmosphere FNO** — the prerequisite for every drag-consuming
phase. Code already exists at `q/core/hno_pytorch.py`; missing pieces are
(a) NRLMSISE-00 Zarr dataset generator, (b) hydrostatic projection training
loop wiring, (c) GCP Spot launcher entry in `gcp_spot_trainer_full.py`.
