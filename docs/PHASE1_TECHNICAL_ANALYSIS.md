# Phase 1 Technical Analysis: Foundation (Months 1-4)
## Critical Path Implementation — Engine Runtime, Neural ODE, ATeena-IQNet

**Date**: April 16, 2026  
**Purpose**: Deep technical research on implementing the three critical workstreams that form the MVP foundation

---

## Table of Contents

1. [Workstream 1.1: Engine Runtime](#workstream-11)
2. [Workstream 1.2: Neural ODE Propagator](#workstream-12)
3. [Workstream 1.3: ATeena-IQNet Foundation](#workstream-13)
4. [Integration Architecture](#integration)
5. [Risk Mitigation & Fallbacks](#risks)
6. [Success Metrics & Validation](#metrics)

---

<a name="workstream-11"></a>
## Workstream 1.1: Engine Runtime — Production gRPC Server

### 1.1 What's Missing

The `SITUATION_AUDIT.md` shows the **Engine Runtime as ❌ PARTIAL**. The existing code has:
- `pkg/common/auditlog.go` — audit logging infrastructure
- `pkg/catalog/` — catalog management with BVH spatial index
- `pkg/collision/` — conjunction screening algorithms
- **Missing**: gRPC server, event loops, streaming APIs, production integration

Current state: The code has components but no cohesive runtime that can ingest sensor data, run UKF updates, screen for conjunctions, and serve alerts in real-time.

### 1.2 The Architecture: Event-Driven Streaming System

**Key Pattern**: Go channels + gRPC streaming + worker pools

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSIS Engine Runtime                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  gRPC Ingest │───▶│ Observation │───▶│  Priority Queue     │  │
│  │  Service     │    │  Validator  │    │  (buffered channels) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         ▲                                        │              │
│         │                                   ┌────┴────┐         │
│  ┌──────┴──────┐                          │         │         │
│  │ mTLS Auth   │                    ┌─────▼───┐ ┌───▼─────┐    │
│  │ (client     │                    │ UKF     │ │ Collision│   │
│  │  certs)     │                    │ Worker  │ │ Worker   │   │
│  └─────────────┘                    │  Pool   │ │  Pool    │   │
│                                     └────┬────┘ └────┬────┘    │
│                                          │           │         │
│  ┌───────────────────────────────────────┴───────────┴────┐    │
│  │                    ONNX Inference Pool                  │    │
│  │         (collision, maneuver, sensor policy)           │    │
│  └───────────────────────────────┬─────────────────────────┘    │
│                                  │                             │
│  ┌───────────────────────────────▼─────────────────────────┐    │
│  │                   Alert Dispatcher                       │    │
│  │         (priority queue + subscription routing)          │    │
│  └───────────────────────────────┬─────────────────────────┘    │
│                                  │                             │
│  ┌───────────────────────────────▼─────────────────────────┐    │
│  │              gRPC Alert Streaming                        │    │
│  │         (SubscribeAlerts server streaming)               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           BFTL Consensus (Month 6+ placeholder)         │    │
│  │              (catalog replication slot)                   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 The Math: Queueing Theory for Real-Time Constraints

**Problem**: Ensure <100ms p99 latency for observation ingestion → alert generation

**M/M/c Queue Model** for worker pools:

```
Service rate: μ = 1/T_service (observations/second)
Arrival rate: λ (observations/second from all sensors)
Number of workers: c

Traffic intensity: ρ = λ / (c·μ)

For stability: ρ < 1

Expected waiting time in queue (Erlang C formula):
W_q = (P_q · T_service) / (c · (1 - ρ))

Where P_q is probability all servers busy
```

**Design Parameters**:
- T_service = 50ms (UKF update with ONNX inference)
- Target W_q < 10ms
- λ_max = 1000 obs/sec (100 radars × 10 Hz)
- Required workers: c = 75

**Go Channel Buffer Sizing**:

```go
obsQueue := make(chan Observation, 1000)     // 1 second of burst
alertQueue := make(chan Alert, 100)          // alerts are rare
ukfJobs := make(chan UKFJob, 75)             // 75 workers
collisionJobs := make(chan CollisionJob, 30) // 30 workers
```

### 1.4 Protobuf Schema Design

```protobuf
// proto/ssis/v1/engine.proto
syntax = "proto3";
package ssis.v1;

service IngestionService {
    rpc StreamObservations(stream Observation) returns (stream IngestAck);
    rpc IngestObservation(Observation) returns (IngestAck);
}

service AlertService {
    rpc SubscribeAlerts(AlertSubscription) returns (stream Alert);
}

message Observation {
    string sensor_id = 1;
    google.protobuf.Timestamp timestamp = 2;
    oneof measurement {
        RadarMeasurement radar = 3;
        OpticalMeasurement optical = 4;
    }
}

message RadarMeasurement {
    string target_id = 1;
    double range_km = 2;
    double range_rate_kms = 3;
    double azimuth_deg = 4;
    double elevation_deg = 5;
    double rcs_dbm = 6;
    double snr_db = 7;
}

message Alert {
    string alert_id = 1;
    AlertType type = 2;
    AlertPriority priority = 3;
    string object_id = 5;
    SHAPAttribution explanation = 100;
}

message SHAPAttribution {
    repeated FeatureImportance features = 1;
}
```

### 1.5 Event Loop Implementation

```go
// pkg/engine/runtime.go
type Runtime struct {
    obsQueue     chan *pb.Observation
    alertQueue   chan *pb.Alert
    ukfPool      *UKFWorkerPool
    catalog      *catalog.BVHIndex
    ctx          context.Context
}

func (r *Runtime) Run() {
    r.ukfPool.Start(75)
    
    for {
        select {
        case obs := <-r.obsQueue:
            r.processObservation(obs)
        case <-r.ctx.Done():
            return
        }
    }
}

func (r *Runtime) processObservation(obs *pb.Observation) {
    if !validateObservation(obs) {
        return
    }
    
    r.ukfPool.Submit(UKFJob{
        Type: "radar",
        Obs:  obs,
        State: r.catalog.GetOrCreate(obs.TargetId),
    })
}
```

### 1.6 ONNX Inference Pool

```go
// pkg/engine/onnx_pool.go
type ONNXInferencePool struct {
    gpuQueue   chan ONNXJob
    batchSize  int
    timeoutMs  int
}

func (p *ONNXInferencePool) batchingWorker() {
    batch := make([]ONNXJob, 0, p.batchSize)
    timer := time.NewTimer(time.Duration(p.timeoutMs) * time.Millisecond)
    
    for {
        select {
        case job := <-p.gpuQueue:
            batch = append(batch, job)
            if len(batch) >= p.batchSize {
                p.runBatch(batch)
                batch = batch[:0]
            }
        case <-timer.C:
            if len(batch) > 0 {
                p.runBatch(batch)
                batch = batch[:0]
            }
        }
    }
}
```

### 1.7 Implementation Timeline

**Month 1**: Protobuf schema, mTLS, server bootstrap, observation ingestion  
**Month 2**: Event loop, UKF worker pool, catalog integration, metrics  
**Month 3**: ONNX Runtime, dynamic batching, collision inference, fallback  
**Month 4**: Alert dispatcher, subscriptions, REST fallback, deployment

---

<a name="workstream-12"></a>
## Workstream 1.2: Neural ODE Propagator

### 2.1 What's Missing

Existing code has:
- `ai/v3/sfmn_kernel.py` — rKAN + Yoshida-4 integrator
- `ai/training/pinn_refinement/train_pinn.py` — PINN training
- **Missing**: Neural ODE with TorchDiffEq, hybrid architecture, ONNX export

### 2.2 The Math: Neural ODE for Orbital Mechanics

**Classical ODE**: `dx/dt = f(t, x)` where f is physics  
**Neural ODE**: `dx/dt = f_physics(t, x) + f_neural(t, x; θ)`

The neural component learns residuals: atmospheric density variations, unmodeled perturbations, thrust.

**State Vector** (6D): `x = [r_x, r_y, r_z, v_x, v_y, v_z]^T`

**Physics Component**:
```python
mu = 398600.4418  # km^3/s^2
r = np.linalg.norm(x[:3])
accel_kepler = -mu / r**3 * x[:3]  # Kepler
accel_j2 = compute_j2(x)            # J2 perturbation
accel_drag = compute_drag(x)        # Atmospheric drag
```

**Neural Component**:
```python
class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),  # [r(3), v(3), t(1)]
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # residual acceleration
        )
```

**Hybrid Dynamics**:
```python
class HybridOrbitalDynamics(nn.Module):
    def forward(self, t, x):
        accel_physics = compute_physics(x)
        accel_neural = self.residual(t, x)
        accel_total = accel_physics + accel_neural
        return torch.cat([x[..., 3:], accel_total], dim=-1)
```

### 2.3 Physics-Informed Loss

```python
def physics_loss(pred, true, times):
    # Observation loss
    loss_obs = F.mse_loss(pred, true)
    
    # Energy conservation
    r, v = pred[..., :3], pred[..., 3:]
    specific_energy = 0.5 * torch.sum(v**2, dim=-1) - mu / torch.norm(r, dim=-1)
    loss_energy = torch.var(specific_energy)
    
    # Angular momentum conservation
    h = torch.cross(r, v, dim=-1)
    loss_momentum = torch.var(torch.norm(h, dim=-1))
    
    return loss_obs + 0.1 * loss_energy + 0.1 * loss_momentum
```

### 2.4 Training with TorchDiffEq

```python
from torchdiffeq import odeint_adjoint

class OrbitalNeuralODE(nn.Module):
    def forward(self, x0, t_span):
        dynamics = HybridDynamics(self.residual_net)
        
        if self.training:
            trajectory = odeint_adjoint(
                dynamics, x0, t_span,
                method='dopri5', rtol=1e-7, atol=1e-9
            )
        else:
            trajectory = odeint(
                dynamics, x0, t_span,
                method='dopri5'
            )
        
        return trajectory.permute(1, 0, 2)
```

### 2.5 Training Data from TLE History

```python
def prepare_training_data(tle_history_file):
    data = []
    
    for sat in load_satellites(tle_history_file):
        tles = sat.tle_history
        
        for i in range(len(tles) - 1):
            tle_current = tles[i]
            tle_future = tles[i+1:]
            
            x0 = sgp4_propagate(tle_current, tle_current.epoch)
            t_obs = [t.epoch for t in tle_future]
            x_obs = [sgp4_propagate(tle_current, t.epoch) for t in tle_future]
            
            data.append({
                'initial_state': x0,
                'obs_times': t_obs,
                'obs_states': x_obs
            })
    
    return data
```

### 2.6 ONNX Export

```python
def export_to_onnx(model, output_path, fixed_dt=60.0):
    class TraceablePropagator(nn.Module):
        def forward(self, x0, num_steps):
            trajectory = [x0]
            x = x0
            
            for _ in range(num_steps):
                # Fixed-step RK4 (traceable)
                k1 = self._compute_derivative(x)
                k2 = self._compute_derivative(x + 0.5 * self.dt * k1)
                k3 = self._compute_derivative(x + 0.5 * self.dt * k2)
                k4 = self._compute_derivative(x + self.dt * k3)
                x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(x)
            
            return torch.stack(trajectory, dim=1)
    
    traceable = TraceablePropagator(model.residual_net, fixed_dt)
    traced = torch.jit.trace(traceable, (x0, num_steps))
    
    torch.onnx.export(traced, (x0, num_steps), output_path,
        input_names=['initial_state', 'num_steps'],
        output_names=['trajectory'],
        dynamic_axes={'initial_state': {0: 'batch'}}
    )
```

### 2.7 Implementation Timeline

**Month 1-2**: TorchDiffEq integration, hybrid dynamics, residual MLP, data pipeline, training loop  
**Month 3**: Fixed-step traceable version, ONNX export, A/B testing, latency optimization  
**Month 4**: Go ONNX integration, fallback to SGP4, production deployment, monitoring

### 2.8 Expected Performance

| Metric | SGP4 | Neural ODE | Improvement |
|:---|:---|:---|:---|
| 1-day position RMSE | ~100m | ~10m | **10×** |
| 7-day position RMSE | ~1km | ~50m | **20×** |
| Inference time | ~1ms | ~50ms | Acceptable |

---

<a name="workstream-13"></a>
## Workstream 1.3: ATeena-IQNet Foundation

### 3.1 What's Missing

- Raw IQ sample processing from ATeena radar
- Learned matched filter
- Range-Doppler map generation
- Target detection with P_d > 0.9 at 0 dB SNR

### 3.2 The Math: Radar Signal Processing

**Range Compression** (Learned 1D-CNN):
```python
# Classical: s_filtered = s(t) ⊗ s*(-t)
# Learned: s_filtered = Conv1D(s; w)
self.range_conv = nn.Sequential(
    nn.Conv1d(2, 64, kernel_size=16),  # learns matched filter
    nn.BatchNorm1d(64),
    nn.ReLU(),
)
```

**Doppler Processing**:
```python
# Classical: FFT along slow-time
# Learned: LSTM
self.doppler_lstm = nn.LSTM(
    input_size=n_range,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)
```

### 3.3 Neural Architecture

```python
class ATeenaIQNet(nn.Module):
    def __init__(self, n_pulses=128, n_range=1024, n_elements=64):
        # Stage 1: Learned range compression
        self.range_conv = nn.Conv1d(2, 64, kernel_size=16)
        
        # Stage 2: Beamforming
        self.beamformer = nn.Linear(n_elements, 64)
        
        # Stage 3: Doppler processing
        self.doppler_lstm = nn.LSTM(n_range, 256, 2, bidirectional=True)
        
        # Stage 4: 2D-CNN on range-Doppler
        self.rd_cnn = nn.Sequential(
            nn.Conv2d(2, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Stage 5: Temporal LSTM
        self.temporal_lstm = nn.LSTM(256*8*8, 256, 2, bidirectional=True)
        
        # Stage 6: Detection head
        self.detection_head = nn.Linear(512, 6)  # [p_det, r, v, az, el, rcs]
```

### 3.4 Synthetic Data Simulator

```python
def simulate_ateena_cpi(targets, n_pulses=128, n_range=1024, n_elements=64):
    """
    Generate raw IQ for one Coherent Processing Interval
    """
    c = 3e8
    lambda_ = c / (3e9)  # S-band
    
    iq = np.zeros((n_pulses, n_range, n_elements), dtype=np.complex128)
    
    for tgt in targets:
        tau = 2 * tgt['range_km'] * 1000 / c
        range_bin = int(tau * 100e6)  # 100 MHz bandwidth
        fd = 2 * tgt['velocity_kms'] * 1000 / lambda_
        
        # Radar equation
        amplitude = np.sqrt(tgt['rcs_m2']) * radar_equation_gain(tgt['range_km'])
        
        # Steering vector
        theta = np.radians(tgt['azimuth_deg'])
        steering = np.exp(-1j * 2 * np.pi * (lambda_/2) * np.arange(n_elements) * np.sin(theta) / lambda_)
        
        for pulse in range(n_pulses):
            doppler_phase = np.exp(1j * 2 * np.pi * fd * pulse * 1e-3)  # PRI = 1ms
            iq[pulse, range_bin, :] += amplitude * doppler_phase * steering
    
    # Add thermal noise
    noise_power = k_B * T_sys * 100e6
    noise = np.sqrt(noise_power/2) * (np.random.randn(*iq.shape) + 1j * np.random.randn(*iq.shape))
    iq += noise
    
    return iq
```

### 3.5 Domain Randomization

```python
def randomize_scenario():
    n_targets = np.random.randint(1, 50)
    targets = []
    
    for _ in range(n_targets):
        targets.append({
            'range_km': np.random.uniform(200, 2000),
            'velocity_kms': np.random.uniform(7.0, 8.0),
            'azimuth_deg': np.random.uniform(0, 360),
            'elev_deg': np.random.uniform(10, 90),
            'rcs_m2': 10 ** np.random.uniform(-2, 2)
        })
    
    noise_figure = np.random.uniform(1.0, 6.0)  # 1-6 dB
    return targets, noise_figure
```

### 3.6 Multi-Task Loss

```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_det = nn.Parameter(torch.zeros(1))
        self.log_sigma_range = nn.Parameter(torch.zeros(1))
        self.log_sigma_doppler = nn.Parameter(torch.zeros(1))
        
    def forward(self, pred, target):
        loss_det = F.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
        
        mask = target[:, 0] > 0.5
        loss_range = F.mse_loss(pred[mask, 1], target[mask, 1])
        loss_doppler = F.mse_loss(pred[mask, 2], target[mask, 2])
        
        # Uncertainty weighting (Kendall & Gal 2018)
        total_loss = (
            torch.exp(-self.log_sigma_det) * loss_det + self.log_sigma_det +
            torch.exp(-self.log_sigma_range) * loss_range + self.log_sigma_range +
            torch.exp(-self.log_sigma_doppler) * loss_doppler + self.log_sigma_doppler
        )
        
        return total_loss
```

### 3.7 Implementation Timeline

**Month 1-2**: ATeena radar model, IQ generation, RCS models, 100k CPI dataset, domain randomization  
**Month 3**: Learned matched filter, Range-Doppler CNN, temporal LSTM, detection head  
**Month 4**: Full model integration, training (synthetic), ONNX export, simulator API

### 3.8 Expected Performance

| Metric | Target | Validation |
|:---|:---|:---|
| Detection probability | > 0.90 at 0 dB SNR | ROC curve |
| False alarm rate | < 1e-6 per CPI | Confusion matrix |
| Range accuracy | < 50m RMSE | Regression error |
| Inference latency | < 100ms | ONNX Runtime |

---

<a name="integration"></a>
## Integration Architecture

### Phase 1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Phase 1 MVP Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS:                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ ATeena Radar │  │ CelesTrak    │  │ Operator     │             │
│  │ (or sim)     │  │ TLEs         │  │ Queries      │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              gRPC Ingestion Service                      │       │
│  │         (mTLS, streaming, validation)                  │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │                Engine Runtime                          │       │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐    │       │
│  │  │ UKF Pool    │ │ Collision   │ │ ONNX Pool     │    │       │
│  │  │ (75 workers)│ │ (30 workers)│ │ (20 workers)  │    │       │
│  │  └──────┬──────┘ └──────┬──────┘ └───────┬───────┘    │       │
│  │         │               │                │            │       │
│  │         └───────────────┴────────────────┘            │       │
│  │                         │                             │       │
│  │                         ▼                             │       │
│  │  ┌─────────────────────────────────────────────────┐  │       │
│  │  │         Neural ODE Propagator                    │  │       │
│  │  │    (ONNX: 50ms, 10× better than SGP4)          │  │       │
│  │  └─────────────────────────────────────────────────┘  │       │
│  │                         │                             │       │
│  │                         ▼                             │       │
│  │  ┌─────────────────────────────────────────────────┐  │       │
│  │  │           Alert Dispatcher                     │  │       │
│  │  │    (priority routing, subscriptions)           │  │       │
│  │  └────────────────────────┬────────────────────────┘  │       │
│  └───────────────────────────┼───────────────────────────┘       │
│                              │                                     │
│  OUTPUTS:                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              gRPC Alert Streaming                        │       │
│  │    (SubscribeAlerts with SHAP attribution)               │       │
│  └─────────────────────────┬───────────────────────────────┘       │
│                            │                                       │
│                            ▼                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ IAF Command  │  │ ISRO Control │  │ Operator UI  │             │
│  │ Center       │  │ Center       │  │ (React)      │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration Points

| Integration | Interface | Latency Requirement |
|:---|:---|:---|
| ATeena Radar → Engine | gRPC streaming | <10ms ingestion |
| Engine → Neural ODE | ONNX Runtime | <50ms inference |
| UKF → Catalog | Go channels (in-memory) | <1ms update |
| Collision → Alert | Priority queue | <100ms end-to-end |
| Alert → Subscribers | gRPC server streaming | <50ms delivery |

---

<a name="risks"></a>
## Risk Mitigation & Fallbacks

### Critical Risks

| Risk | Impact | Mitigation |
|:---|:---|:---|
| **Neural ODE training unstable** | High | Gradient clipping (max_norm=1.0); adaptive LR; fallback to RK4 |
| **ONNX export fails** | High | TorchScript trace; fixed-step RK4; fallback to SGP4 |
| **Synthetic radar data insufficient** | High | Domain randomization; classical matched filter baseline; plan for transfer learning |
| **gRPC streaming backpressure** | Medium | Buffered channels; flow control; graceful degradation |

### Fallback Strategies

**Neural ODE → SGP4 Fallback**:
```go
func (p *HybridPropagator) Propagate(state State, dt float64) (State, error) {
    // Try Neural ODE first
    if p.neuralODEAvailable {
        result, err := p.neuralODE.Infer(state, dt)
        if err == nil && p.validateResult(result) {
            return result, nil
        }
        p.metrics.IncNeuralODEFallback()
    }
    
    // Fallback to SGP4
    return p.sgp4.Propagate(state, dt)
}
```

**ONNX GPU → CPU Fallback**:
```go
func (p *ONNXPool) runInference(inputs []float32) ([]float32, error) {
    // Try GPU
    result, err := p.gpuSession.Run(inputs)
    if err != nil {
        p.metrics.IncGPUFallback()
        // Fallback to CPU
        return p.cpuSession.Run(inputs)
    }
    return result, nil
}
```

---

<a name="metrics"></a>
## Success Metrics & Validation

### Technical KPIs

| Metric | Target | Measurement |
|:---|:---|:---|
| Observation ingestion | <10ms p99 | Prometheus histogram |
| Neural ODE inference | <50ms p99 | ONNX Runtime profiling |
| UKF update rate | 1000/sec | Load testing |
| Alert generation | <100ms end-to-end | Distributed tracing |
| System availability | 99.0% | Uptime monitoring |

### Phase 1 Validation Tests

**Test 1: Load Test** (Month 4)
```bash
# Simulate 100 radars at 10 Hz = 1000 obs/sec
go test -v -run TestLoad1000 ./tests/load/ \
  -duration=1h \
  -ingestion-rate=1000 \
  -assert-p99-latency=100ms
```

**Test 2: Neural ODE Accuracy** (Month 4)
```bash
# Compare against 7-day propagated TLEs
go test -v -run TestNeuralODEAccuracy ./tests/orbital/ \
  -horizon=7days \
  -threshold-rmse=50m \
  -baseline=sgp4
```

**Test 3: ATeena Detection** (Month 4)
```bash
# Validate on held-out synthetic test set
python -m ai.radar.validate \
  --model models/radar/ateena_iqnet.onnx \
  --test-data data/radar/test_10k.h5 \
  --target-pd 0.90 \
  --target-snr 0db
```

### Go/No-Go Decision Gate (Month 4)

| Criteria | Threshold | Owner |
|:---|:---|:---|
| Engine handles 1000 obs/sec | <100ms p99 latency | Go Lead |
| Neural ODE vs SGP4 | 10× accuracy improvement | ML Lead |
| ATeena P_d at 0 dB SNR | > 0.90 | ML Engineer |
| System availability | > 99.0% over 1 week | SRE |
| Security audit | No critical CVEs | Security |

**If all criteria met**: Proceed to Phase 2 (Multi-Sensor Fusion)  
**If criteria missed**: Extend Phase 1, re-scope Phase 2

---

## Conclusion

This Phase 1 technical analysis provides:

1. **Detailed architecture** for Engine Runtime with queueing theory-based sizing
2. **Neural ODE mathematics** with TorchDiffEq integration and physics-informed loss
3. **ATeena-IQNet simulator** for synthetic training data generation
4. **Integration points** between all three workstreams
5. **Risk mitigation** with concrete fallback strategies
6. **Validation criteria** with specific tests and thresholds

The critical path is **Engine Runtime → Neural ODE → ATeena-IQNet** in that order, with each component having a production-grade fallback to existing (simpler) implementations.

**Total Phase 1 Budget**: ₹1.2 Cr (personnel ₹1.0 Cr, infrastructure ₹0.2 Cr)  
**Team**: 2 Go engineers, 3 ML engineers, 1 ML Lead (6 FTE)  
**Duration**: 4 months  
**Output**: Production MVP with neural-enhanced propagation serving real-time alerts
