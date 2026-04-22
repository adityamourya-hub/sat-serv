# M1 Implementation Plan
## Scaffolding, Data Generators, Drag Physics, ZMQ Topics, Baseline Pc Calibrator

**Goal:** Close the five capability gaps identified against IAF SDA requirements by adding trainable pipelines for Volumetric Scanning, Neighbourhood Watch Screening, Characterisation/Cataloguing, Re-entry Analysis, and Conjunction Analysis—without breaking existing Collision, Maneuver, or Coverage flows.

**Non-Goals:** Full model training runs (M2+), GUI/dashboard changes, hardware procurement.

**Constraint:** Every new model must export to ONNX, register with the existing `SDA_Trainer` orchestrator, and remain compatible with the Plasma/ZMQ IPC already present in `sda-train/train_init.py`.

---

## 1. Directory & File Scaffolding

Create the following structure under `sda-train/` and `ai/`.

```
sda-train/
  data/
    synthetic/
      volumetric/           # BEV grids + 3D labels
      neighbourhood/        # encounter trajectories + intent labels
      reentry/              # decay trajectories + TTD labels
      conjunction/          # MC encounter scenarios + empirical Pc labels
      catalog/              # cropped chips + kinematics + attribute labels
    processed/
      manifests/            # per-shard JSON manifests (reuse generate_manifest.py)
  pkg/
    physics/
      drag_models.py        # atmospheric density + drag acceleration
    volumetric/
      bev_encoder.py        # CenterHead-style BEV model (PyTorch)
    neighbourhood/
      intent_encoder.py     # 1D CNN/LSTM over 120 s windows
    reentry/
      decay_forecaster.py     # seq2event GRU + corridor head
    conjunction/
      pc_calibrator.py      # pairwise MLP / small GNN calibrator
    catalog/
      dual_encoder.py       # image encoder + kinematic encoder
  scripts/
    gen_volumetric.py       # volumetric scenario generator
    gen_neighbourhood.py    # encounter scenario generator
    gen_reentry.py          # re-entry decay generator
    gen_conjunction.py      # MC conjunction scenario generator
    gen_catalog.py          # synthetic chip generator from render_sim.py

ai/
  training/
    models/                 # ONNX exports (existing, reuse)
```

**No files are deleted.** Existing `sda-train/scripts/init_data.py`, `render_sim.py`, and `train_init.py` are extended, not replaced.

---

## 2. Physics Loss: Drag Term (`sda-train/pkg/physics/loss.py`)

### 2.1 Why
Re-entry and low-altitude conjunction analysis require drag. The existing `PhysicsInformedLoss` only models Keplerian gravity. We add a drag residual that can be toggled per training stage.

### 2.2 Changes

**Add `drag_models.py` as a new file** (do not bloat `loss.py`):

```python
# sda-train/pkg/physics/drag_models.py
import torch
import numpy as np

class USStandardAtmosphere1976:
    """Exponential atmosphere with reference scale heights."""
    def __init__(self, h0: float = 0.0, rho0: float = 1.225,
                 H: float = 8500.0):
        self.h0 = h0      # m
        self.rho0 = rho0  # kg/m^3
        self.H = H        # m (scale height)

    def density(self, altitude_m: torch.Tensor) -> torch.Tensor:
        return self.rho0 * torch.exp(-(altitude_m - self.h0) / self.H)


def drag_acceleration(state: torch.Tensor,
                       cd: float = 2.2,
                       area_mass_ratio: float = 0.01) -> torch.Tensor:
    """
    Compute drag acceleration vector [ax, ay, az].
    state: [..., 6] = [x, y, z, vx, vy, vz] in meters, m/s
    """
    r_vec = state[..., :3]
    v_vec = state[..., 3:]
    r_mag = torch.norm(r_vec, dim=-1, keepdim=True)
    v_mag = torch.norm(v_vec, dim=-1, keepdim=True)

    # Altitude above Earth surface (assumes spherical Earth)
    altitude = r_mag - 6371000.0

    # Simple exponential model; extend with tables later
    atm = USStandardAtmosphere1976()
    rho = atm.density(altitude)

    # Drag = -0.5 * Cd * (A/m) * rho * |v| * v
    drag = -0.5 * cd * area_mass_ratio * rho * v_mag * v_vec
    return drag
```

**Modify `PhysicsInformedLoss.__init__`** to accept a `drag_weight`:

```python
def __init__(self, mu=3.986004418e14, w_data=0.7, w_physics=0.3,
             w_drag=0.0, cd=2.2, area_mass_ratio=0.01):
    ...
    self.w_drag = w_drag
    self.cd = cd
    self.area_mass_ratio = area_mass_ratio
```

**Add `drag_residual` method to `PhysicsInformedLoss`**:

```python
def drag_residual(self, pred_state: torch.Tensor) -> torch.Tensor:
    """
    ODE residual including drag.
    Expected acceleration = -mu * r / |r|^3 + a_drag
    """
    from .drag_models import drag_acceleration
    r = pred_state[..., :3]
    v = pred_state[..., 3:]
    r_mag = torch.norm(r, dim=-1, keepdim=True)
    expected = -self.mu * r / (r_mag ** 3)
    expected += drag_acceleration(pred_state, self.cd, self.area_mass_ratio)

    if pred_state.dim() == 3 and pred_state.shape[1] > 1:
        actual = torch.diff(v, dim=1)
        actual = torch.nn.functional.pad(actual, (0, 0, 0, 1), mode='replicate')
    else:
        actual = torch.zeros_like(expected)

    return torch.mean((actual - expected) ** 2)
```

**Update `__call__` in `PhysicsInformedLoss`**:

```python
total_loss = (
    self.w_data * mse_loss +
    self.w_physics * physics_loss +
    self.w_drag * self.drag_residual(pred_state) +
    w_conservation * (energy_loss + angular_momentum_loss)
)
```

**Update `get_loss_components`** to return `drag_loss` when `w_drag > 0`.

### 2.3 Acceptance Criteria
- `python -m pkg.physics.loss` still passes (backward compatible with `w_drag=0`).
- A synthetic decay trajectory at 120 km altitude produces non-zero drag loss.

---

## 3. ZMQ Topic Router (`sda-train/train_init.py`)

### 3.1 Why
The current `SDA_Trainer` uses a single PULL socket and a pipe-delimited string. We need multiplexed topics so the Go engine can route volumetric frames, neighbourhood kinematics, re-entry states, and conjunction pairs to the correct Python trainer without spawning five separate processes.

### 3.2 Changes

**Replace the single `receiver` with a topic-aware router.** Keep the existing socket for backward compatibility, but add a secondary PULL socket for typed messages:

```python
# In SDA_Trainer.__init__
self.router = self.context.socket(zmq.PULL)
self.router.bind("ipc:///tmp/train_topic.ipc")

self.topic_handlers = {
    "collision": self._handle_collision,
    "maneuver":  self._handle_maneuver,
    "coverage":  self._handle_coverage,
    # New topics
    "volumetric.scan": self._handle_volumetric,
    "watch.intent":    self._handle_neighbourhood,
    "catalog.query":   self._handle_catalog,
    "reentry.forecast":self._handle_reentry,
    "conj.analyze":    self._handle_conjunction,
}
```

**Message format:**
```
TOPIC|<json_payload>
```

Where `<json_payload>` is a compact JSON string (no newlines). Example:
```
volumetric.scan|{"frame_id":"abc123","pose":[...],"roi_km":500}
```

**Add `_route_message`**:

```python
def _route_message(self, raw: str):
    if "|" not in raw:
        # Legacy pipe-delimited orbital state
        return self.receive_training_data(raw)
    topic, payload = raw.split("|", 1)
    handler = self.topic_handlers.get(topic)
    if handler:
        return handler(json.loads(payload))
    return None
```

**Stub each new handler** (concrete implementations live in dedicated trainer classes):

```python
def _handle_volumetric(self, payload: dict) -> dict:
    """Queue a volumetric scan frame for BEV training."""
    self.volumetric_buffer.append(payload)
    return {"status": "queued", "topic": "volumetric.scan"}

def _handle_neighbourhood(self, payload: dict) -> dict:
    """Queue a watch-list encounter for intent scoring."""
    self.neighbourhood_buffer.append(payload)
    return {"status": "queued", "topic": "watch.intent"}

def _handle_reentry(self, payload: dict) -> dict:
    """Queue a decay trajectory for TTD training."""
    self.reentry_buffer.append(payload)
    return {"status": "queued", "topic": "reentry.forecast"}

def _handle_conjunction(self, payload: dict) -> dict:
    """Queue a conjunction pair for Pc calibration."""
    self.conjunction_buffer.append(payload)
    return {"status": "queued", "topic": "conj.analyze"}

def _handle_catalog(self, payload: dict) -> dict:
    """Queue a catalog query/update for dual-encoder training."""
    self.catalog_buffer.append(payload)
    return {"status": "queued", "topic": "catalog.query"}
```

**Add buffer caps** to prevent unbounded memory growth:

```python
self.buffer_caps = {
    "volumetric": 256,
    "neighbourhood": 1024,
    "reentry": 512,
    "conjunction": 2048,
    "catalog": 512,
}
```

When a buffer exceeds its cap, drop the oldest sample (circular).

### 3.3 Acceptance Criteria
- Go can send `volumetric.scan|{"frame_id":"f1"}` and Python logs `[ROUTER] queued volumetric.scan f1`.
- Legacy `99999|6800000|50000|50000|0|7500|0|1234567890` still routes to the original regressor path.
- No ZMQ socket collisions when both old and new sockets are bound.

---

## 4. Data Generators

### 4.1 Volumetric Scan (`sda-train/scripts/gen_volumetric.py`)

**What it produces:**
- Synthetic 2D BEV occupancy grids (256 x 256 x 4 channels: density, vx, vy, vz).
- 3D ground-truth boxes: `[x, y, z, vx, vy, vz, class]` where class = {satellite, debris, unknown}.

**How:**
1. Re-use `PixelRenderer` from `render_sim.py` to render star-field + object frames.
2. Project truth states into a 500 km x 500 km x 200 km volume centered on the sensor.
3. Voxelise to 2 km resolution (256 x 256 x 100).
4. For training efficiency, collapse the vertical axis to a 256 x 256 BEV with four channels.
5. Save shards as NPZ: `bev_grid`, `labels`, `camera_pose`, `timestamp`.

**Volume:** 10,000 scenes x 4 curriculum levels (clean → dense debris + radiation).

**Output path:** `sda-train/data/synthetic/volumetric/vol_<shard_id>.npz`

### 4.2 Neighbourhood Watch (`sda-train/scripts/gen_neighbourhood.py`)

**What it produces:**
- Time-series encounter windows: 120 seconds at 1 Hz = 120 timesteps per sample.
- Features per timestep: relative position (3), relative velocity (3), approach rate, miss distance, solar phase angle, illumination flag.
- Label: `intent` (0=benign flyby, 1=intercept), `tti` (time-to-intercept in seconds, -1 if benign).

**How:**
1. Sample two orbital states from a LEO shell (200–2000 km).
2. Propagate both for 120 s with 2-body + J2 (analytic for speed).
3. Compute relative geometry. If min range < 50 km and approach rate > 0, label as intercept with `tti` at CPA.
4. If min range > 100 km, label benign.
5. Add Gaussian noise (σ_pos = 50 m, σ_vel = 0.5 m/s) to simulate sensor uncertainty.

**Volume:** 50,000 windows (10% intercept, 90% benign).

**Output path:** `sda-train/data/synthetic/neighbourhood/nbh_<shard_id>.npz`

### 4.3 Re-entry Decay (`sda-train/scripts/gen_reentry.py`)

**What it produces:**
- State sequences from 140 km down to 30 km (or breakup).
- Label: `time_to_demise` (seconds from first state), `corridor_lat`, `corridor_lon` (impact lat/lon).

**How:**
1. Start at a random state inside 140 km altitude, v ≈ 7.5 km/s.
2. Propagate with Runge-Kutta 4, force model = gravity + J2 + exponential drag.
3. Ballistic coefficient `β = Cd * A / m` sampled from [0.005, 0.1] m²/kg.
4. Stop when altitude < 30 km or vertical velocity > 0 (skip-out).
5. If breakup, record impact coordinates.

**Volume:** 20,000 trajectories.

**Output path:** `sda-train/data/synthetic/reentry/re_<shard_id>.npz`

### 4.4 Conjunction MC (`sda-train/scripts/gen_conjunction.py`)

**What it produces:**
- Pairwise encounter geometry + empirical Pc label.
- Features: relative position, relative velocity, combined covariance (if available), miss distance, time-to-CPA, object sizes, solar exposure, sensor geometry.
- Label: `empirical_pc` from 10,000 Monte Carlo samples per encounter.

**How:**
1. Sample primary and secondary states near a close approach.
2. Add navigation uncertainty (covariance diagonal σ_pos = 100 m, σ_vel = 1 m/s).
3. Run 10,000 miss-distance samples. `empirical_pc = hits_inside_combined_radius / 10000`.
4. Compute analytic Pc (Alfriend 1999) as an auxiliary feature (not the label).

**Volume:** 100,000 encounters (1% high-Pc, 10% medium, 89% low for balance).

**Output path:** `sda-train/data/synthetic/conjunction/conj_<shard_id>.jsonl`

### 4.5 Catalog Chips (`sda-train/scripts/gen_catalog.py`)

**What it produces:**
- 64 x 64 pixel chips cropped from `PixelRenderer` frames.
- Kinematic vector: `[a, e, i, RAAN, arg_perigee, mean_anomaly, period, rcs_estimate]`.
- Attribute labels: bus type (one-hot), mission class, size class, attitude stability flag.

**How:**
1. Use existing `render_sim.py` to render a frame with 1–5 objects.
2. For each object, project to pixel coordinates and crop 64 x 64.
3. Vary brightness, PSF, orientation, and glint to simulate different bus types.
4. Kinematics drawn from `assets/indian_satellite_catalogue.json` distributions.

**Volume:** 200,000 chips.

**Output path:** `sda-train/data/synthetic/catalog/cat_<shard_id>.npz`

---

## 5. Baseline Pc Calibrator (`sda-train/pkg/conjunction/pc_calibrator.py`)

### 5.1 Architecture
A tiny MLP (≤ 10k parameters) to keep CPU inference < 5 ms per pair.

```python
import torch.nn as nn

class PcCalibrator(nn.Module):
    def __init__(self, in_dim=12, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, features):
        # features: [B, 12]
        # 12 dims: [log10_miss_dist, log10_analytic_pc, rel_speed,
        #          size_primary, size_secondary, aspect_ratio,
        #          pos_uncertainty, vel_uncertainty, solar_angle,
        #          time_to_cpa, sin_inclination, cos_inclination]
        return self.net(features)
```

### 5.2 Training Regimen
- **Loss:** Brier score + Expected Calibration Error (ECE) regularizer.
- **ECE:** Bin predictions into 10 bins, compute |accuracy – mean_prediction| per bin, sum weighted by bin count.
- **Monotonicity loss:** `relu(dPcalib/d(log_miss_dist))` must be ≤ 0 (calibrated Pc must not increase with increasing miss distance). Approximate with finite-difference penalty on a held-out grid.
- **Optimizer:** AdamW, lr=1e-3, batch=512, 50 epochs.
- **Data:** `gen_conjunction.py` output, 80/20 split.

### 5.3 Export
```python
dummy = torch.zeros(1, 12)
torch.onnx.export(model, dummy, "ai/training/models/pc_calibrator_v1.onnx",
                  input_names=["features"], output_names=["pc_calibrated"],
                  dynamic_axes={"features": {0: "batch"}})
```

### 5.4 Integration
- Go receives raw encounter geometry, computes analytic Pc and extracts the 12 features.
- ONNX runtime loads `pc_calibrator_v1.onnx`.
- Output: calibrated Pc + Brier uncertainty band.

---

## 6. Orchestrator Wiring

**Modify `SDA_Trainer` (`sda-train/train_init.py`)** to instantiate trainers only when data arrives (lazy init) to save RAM:

```python
class SDA_Trainer:
    ...
    def _get_volumetric_trainer(self):
        if not hasattr(self, '_vol_trainer'):
            from pkg.volumetric.bev_encoder import BEVTrainer
            self._vol_trainer = BEVTrainer()
        return self._vol_trainer

    def _get_pc_calibrator(self):
        if not hasattr(self, '_pc_trainer'):
            from pkg.conjunction.pc_calibrator import PcCalibratorTrainer
            self._pc_trainer = PcCalibratorTrainer()
        return self._pc_trainer

    # Analogous stubs for neighbourhood, reentry, catalog
```

**Add `process_batch_for_topic(self, topic: str)`**:
- Checks buffer length.
- When `len(buffer) >= batch_size`, calls the corresponding trainer's `train_step()`.
- Logs loss components to stdout (existing pattern) and optionally to `sda-train/data/processed/<topic>_metrics.jsonl`.

---

## 7. Test & Verification Plan

| Component | Test | Command |
|-----------|------|---------|
| Drag loss | `python -m pkg.physics.loss` passes with `w_drag=0.1` | `cd sda-train && python pkg/physics/loss.py` |
| ZMQ router | Send synthetic topic messages, verify routing | `python scripts/test_zmq_router.py` (new) |
| Volumetric generator | Generate 100 scenes, validate NPZ shape | `python scripts/gen_volumetric.py --validate` |
| Neighbourhood generator | Generate 1k windows, check class balance | `python scripts/gen_neighbourhood.py --validate` |
| Re-entry generator | Check energy monotonic decrease (drag) | `python scripts/gen_reentry.py --validate` |
| Conjunction generator | MC Pc vs analytic Pc sanity check | `python scripts/gen_conjunction.py --validate` |
| Pc calibrator | Train 5 epochs, export ONNX, run inference | `python pkg/conjunction/pc_calibrator.py` |
| End-to-end | Go sends topic msg → Python routes → ONNX export exists | `make test-m1` (new Makefile target) |

---

## 8. Task Breakdown & Owners (1 Developer)

**Day 1–2: Scaffolding + Drag**
- Create directory tree.
- Implement `drag_models.py`.
- Wire drag into `PhysicsInformedLoss`.
- Write drag unit test.

**Day 3–4: ZMQ Topic Router**
- Refactor `SDA_Trainer` with `_route_message`.
- Add buffer management.
- Write `scripts/test_zmq_router.py`.

**Day 5–7: Data Generators (all five)**
- `gen_volumetric.py`, `gen_neighbourhood.py`, `gen_reentry.py`, `gen_conjunction.py`, `gen_catalog.py`.
- Each includes `--validate` flag.
- Generate a small golden shard (≤ 100 MB total) committed to LFS or documented for CI.

**Day 8–9: Pc Calibrator**
- Implement `pc_calibrator.py` (model + trainer).
- Train on synthetic conjunction data.
- Export ONNX.
- Verify monotonicity penalty works.

**Day 10: Integration + Hardening**
- Wire lazy trainer init into `SDA_Trainer`.
- Add `make test-m1` target.
- Run full end-to-end: Go stub sender → Python router → training step → ONNX checkpoint.
- Update `generate_manifest.py` metadata schema to include new model types.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Synthetic data does not match real sensor PSF/noise | Add domain-randomisation in `PixelRenderer` (vary PSF σ, background levels, readout noise). |
| Pc calibrator overfits to analytic Pc | Use empirical MC Pc as label, not analytic Pc. Drop analytic Pc from inputs during ablation. |
| ZMQ IPC buffer overflow on M2 Air | Cap buffers (§3.2), drop oldest, add backpressure signal to Go. |
| Drag model too simple for HGV skip-glide | Keep exponential model for M1; swap to NRLMSISE-00 or neural density in M3. |
| Curriculum warming conflicts with new generators | Each generator accepts `level` arg (0–3) and maps to its own difficulty knobs. |

---

## 10. Definition of Done

- [ ] `sda-train/pkg/physics/drag_models.py` exists and drag loss is non-zero below 150 km.
- [ ] `SDA_Trainer` routes all six topics without crashing on legacy messages.
- [ ] At least 1,000 samples exist per new capability in `sda-train/data/synthetic/`.
- [ ] `pc_calibrator_v1.onnx` exports and infers in < 5 ms per pair on CPU.
- [ ] `make test-m1` passes in under 60 seconds.
- [ ] No regression in existing collision GNN or coverage Prophet pipelines.
