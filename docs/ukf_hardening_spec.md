# UKF Hardening Specification for Indian SSA

## Scope
This specification hardens the existing UKF core in `pkg/orbital/fusion.go` so that it remains the sovereign physics-first estimator for Indian SSA workloads under ADITI 4.0.

It covers:
- rigorous justification of UKF over EKF,
- India-specific force-model extensions,
- covariance and innovation telemetry for AIML features,
- per-track keyed filter isolation,
- persistence and replay contracts,
- benchmark and acceptance criteria.

## 1. Why UKF, Not EKF

### State and dynamics
$$
\mathbf{x} = \begin{bmatrix} r_x & r_y & r_z & v_x & v_y & v_z \end{bmatrix}^T \in \mathbb{R}^6
$$

$$
\dot{\mathbf{x}} = f(\mathbf{x}, t) = \begin{bmatrix} \mathbf{v} \\ -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{\text{pert}} \end{bmatrix}
$$

where $\mathbf{a}_{\text{pert}}$ includes:
- zonal harmonics ($J_2, J_3, J_4$),
- atmospheric drag driven by $\mathrm{F10.7}$ and $A_p$,
- Sun/Moon third-body acceleration,
- optional SRP.

### Measurement model
For MOTR/NETRA/DRDO radar observables:
$$
\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(0, R_k)
$$

with
$$
\mathbf{z}_k = \begin{bmatrix} \rho \\ \dot{\rho} \\ \theta \\ \phi \end{bmatrix}
$$

### EKF linearization error
EKF uses first-order linearization:
$$
\mathbf{x}_{k+1} \approx f(\hat{\mathbf{x}}_k, t_k) + F_k(\mathbf{x}_k - \hat{\mathbf{x}}_k)
$$
$$
\mathbf{z}_k \approx h(\hat{\mathbf{x}}_{k|k-1}) + H_k(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})
$$
where $F_k = \partial f / \partial \mathbf{x}$ and $H_k = \partial h / \partial \mathbf{x}$.

The truncation remainder is second-order in the local perturbation, so the covariance update is biased by neglected curvature terms:
$$
\mathrm{Cov}[f(\mathbf{x})] = FPF^T + O(\|\Delta x\|^3)
$$
$$
\mathrm{Cov}[h(\mathbf{x})] = HPH^T + O(\|\Delta x\|^3)
$$
During Indian LEO solar-maximum drag spikes, low-elevation geometry, and close approaches, the neglected higher-order terms become operationally dominant, producing optimistic covariance and inconsistent gates.

### UKF unscented transform
For $n=6$ and parameters $\alpha=0.1$, $\beta=2$, $\kappa=0$:
$$
\lambda = \alpha^2(n+\kappa) - n
$$
$$
\mathcal{X}_0 = \hat{\mathbf{x}}, \quad
\mathcal{X}_i = \hat{\mathbf{x}} \pm \sqrt{(n+\lambda)P}_i, \quad i=1\dots 2n
$$

Weights:
$$
W_0^{(m)} = \frac{\lambda}{n+\lambda}, \quad
W_0^{(c)} = \frac{\lambda}{n+\lambda} + (1-\alpha^2+\beta), \quad
W_i^{(m,c)} = \frac{1}{2(n+\lambda)}
$$

Time update:
$$
\hat{\mathbf{x}}_{k|k-1} = \sum_{i=0}^{2n} W_i^{(m)} \mathcal{X}_i^{(k|k-1)}
$$
$$
P_{k|k-1} = \sum_{i=0}^{2n} W_i^{(c)}(\mathcal{X}_i - \hat{\mathbf{x}})(\mathcal{X}_i - \hat{\mathbf{x}})^T + Q
$$

Measurement update:
$$
\hat{\mathbf{z}} = \sum W_i^{(m)} \mathcal{Z}_i
$$
$$
S = \sum W_i^{(c)}(\mathcal{Z}_i - \hat{\mathbf{z}})(\mathcal{Z}_i - \hat{\mathbf{z}})^T + R
$$
$$
P_{xz} = \sum W_i^{(c)}(\mathcal{X}_i - \hat{\mathbf{x}})(\mathcal{Z}_i - \hat{\mathbf{z}})^T
$$
$$
K = P_{xz}S^{-1}, \quad \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + K\boldsymbol{\nu}_k, \quad P_k = P_{k|k-1} - KSK^T
$$

UKF preserves nonlinear observability without Jacobian linearization and is therefore the correct sovereign estimator for Indian SSA close-approach, re-entry, and drag-dominated regimes.

## 2. India-Specific Force Model Hardening

The current `PropagateCartesianState` path should be hardened into an explicit force-model stack:

$$
\mathbf{a}_{\text{total}} = \mathbf{a}_{2B} + \mathbf{a}_{J2} + \mathbf{a}_{J3} + \mathbf{a}_{J4} + \mathbf{a}_{\text{drag}} + \mathbf{a}_{\text{3rd}} + \mathbf{a}_{\text{SRP}}
$$

### J2/J3/J4
$$
\mathbf{a}_{J2} = -\frac{3}{2} \frac{\mu J_2 R_E^2}{r^5}
\left[\left(1-5\frac{z^2}{r^2}\right)\mathbf{r} + 2z\mathbf{e}_z\right]
$$
Higher zonals should be included using the same Earth-fixed Earth-centered formulation already used in SSP/SSA tools.

### Atmospheric drag
$$
\mathbf{a}_{\text{drag}} = -\frac{1}{2}\rho v_{\text{rel}}\mathbf{v}_{\text{rel}}\frac{C_DA}{m}
$$
where:
- $\rho$ is resolved by NRLMSISE-00,
- $C_D$ is tuned per Indian spacecraft class,
- $A/m$ comes from geometry catalog metadata,
- $\mathrm{F10.7}$ and $A_p$ are sourced with provenance.

### Third-body perturbations
$$
\mathbf{a}_{\text{3rd}} = \mu_3 \left( \frac{\mathbf{r}_3 - \mathbf{r}}{|\mathbf{r}_3 - \mathbf{r}|^3} - \frac{\mathbf{r}_3}{|\mathbf{r}_3|^3} \right)
$$
for Sun and Moon.

## 3. Required Go Code Deltas

### `pkg/dsp/measurement_filter.go`
Add the cleaned-measurement telemetry contract:
- innovation vector,
- UKF pre/post state,
- UKF pre/post covariance,
- trace and max-eigenvalue metrics,
- drag coefficient estimate,
- solar flux provenance,
- perturbation model version.

### `pkg/dsp/filter_registry.go`
Use a string-based key:
```go
type FilterKey struct {
    ObjectID   string
    SiteID     string
    SensorMode string
}
```
This eliminates cross-object contamination and is compatible with Indian multi-sensor track management.

### `pkg/orbital/fusion.go`
Fusion must remain UKF-only. Do not add EKF fallback. The measurement path should be:
1. resolve keyed filter,
2. clean raw radar measurement,
3. propagate sigma points,
4. update with exact nonlinear radar model,
5. emit cleaned record with pre/post covariance telemetry.

### `pkg/persistence/store.go`
Persist cleaned-measurement telemetry into the measurement projection and signed ontology-event record.

## 4. measurement_records Projection (PostgreSQL)

### Columns
- `id`
- `object_ref`
- `site_id`
- `sensor_mode`
- `ts`
- `payload`
- `signature`
- `payload_digest`
- `ukf_cov_pre` JSONB
- `ukf_cov_post` JSONB
- `trace_p` DOUBLE PRECISION
- `max_eigen_p` DOUBLE PRECISION
- `innovation_mahalanobis` DOUBLE PRECISION
- `innovation_vector` JSONB
- `drag_coeff_estimate` DOUBLE PRECISION
- `solar_flux_f107` DOUBLE PRECISION
- `perturbation_model_version` TEXT

### Migration
See [migrations/001_measurement_records.sql](../migrations/001_measurement_records.sql).

## 5. Keyed Filter Registry

The per-track registry is now the correct isolation boundary.

### Semantics
- key = `object_id + site_id + sensor_mode`
- lazy initialization of one filter per active track
- inactivity cleanup policy should be added as a later enhancement using LRU/TTL semantics

### Integration target
- `measurement_orchestrator.go` should always call `registry.GetOrCreate(key)` before UKF fusion.

## 6. Test and Benchmark Plan

### Required benchmark cases
- LEO solar-maximum drag spike over Indian LEO asset
- low-elevation NETRA-style track near horizon
- MOTR conjunction screening run
- re-entry trajectory over Indian landmass
- filter-registry hot path lookup

### Acceptance criteria
- UKF covariance remains positive definite after update
- Mahalanobis gate pass rate stays above 95% in simulated close approach
- per-update runtime stays below 10 ms on target hardware

### Suggested command
```bash
go test -bench . -benchmem ./pkg/orbital ./pkg/dsp
```

## 7. Implementation Order
1. Keep UKF as the only state estimator.
2. Complete force-model enrichment and provenance telemetry.
3. Keep keyed filter registry as the track isolation boundary.
4. Persist covariance telemetry to measurement_records.
5. Feed the resulting stream into AIML only after calibration and replay validation.

## 8. Repository-Specific Notes
- Current code already uses UKF sigma points and nonlinear measurement update in `pkg/orbital/fusion.go`.
- The current registry and orchestrator layers are the right insertion points for the hardening changes.
- The current persistence layer already signs events; the measurement projection just needs the richer UKF telemetry fields to support downstream AIML.
# UKF Hardening Specification for Indian SSA

## Scope
This specification hardens the existing UKF core in `pkg/orbital/fusion.go` so that it remains the sovereign physics-first estimator for Indian SSA workloads under ADITI 4.0.

It covers:
- rigorous justification of UKF over EKF,
- India-specific force-model extensions,
- covariance and innovation telemetry for AIML features,
- per-track keyed filter isolation,
- persistence and replay contracts,
- benchmark and acceptance criteria.

## 1. Why UKF, Not EKF

### State and dynamics
$$
\mathbf{x} = \begin{bmatrix} r_x & r_y & r_z & v_x & v_y & v_z \end{bmatrix}^T \in \mathbb{R}^6
$$

$$
\dot{\mathbf{x}} = f(\mathbf{x}, t) = \begin{bmatrix} \mathbf{v} \\ -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{\text{pert}} \end{bmatrix}
$$

where $\mathbf{a}_{\text{pert}}$ includes:
- zonal harmonics ($J_2, J_3, J_4$),
- atmospheric drag driven by $\mathrm{F10.7}$ and $A_p$,
- Sun/Moon third-body acceleration,
- optional SRP.

### Measurement model
For MOTR/NETRA/DRDO radar observables:
$$
\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(0, R_k)
$$

with
$$
\mathbf{z}_k = \begin{bmatrix} \rho \\ \dot{\rho} \\ \theta \\ \phi \end{bmatrix}
$$

### EKF linearization error
EKF uses first-order linearization:
$$
\mathbf{x}_{k+1} \approx f(\hat{\mathbf{x}}_k, t_k) + F_k(\mathbf{x}_k - \hat{\mathbf{x}}_k)
$$
$$
\mathbf{z}_k \approx h(\hat{\mathbf{x}}_{k|k-1}) + H_k(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})
$$
where $F_k = \partial f / \partial \mathbf{x}$ and $H_k = \partial h / \partial \mathbf{x}$.

The truncation remainder is second-order in the local perturbation, so the covariance update is biased by neglected curvature terms:
$$
\mathrm{Cov}[f(\mathbf{x})] = FPF^T + O(\|\Delta x\|^3)
$$
$$
\mathrm{Cov}[h(\mathbf{x})] = HPH^T + O(\|\Delta x\|^3)
$$
During Indian LEO solar-maximum drag spikes, low-elevation geometry, and close approaches, the neglected higher-order terms become operationally dominant, producing optimistic covariance and inconsistent gates.

### UKF unscented transform
For $n=6$ and parameters $\alpha=0.1$, $\beta=2$, $\kappa=0$:
$$
\lambda = \alpha^2(n+\kappa) - n
$$
$$
\mathcal{X}_0 = \hat{\mathbf{x}}, \quad
\mathcal{X}_i = \hat{\mathbf{x}} \pm \sqrt{(n+\lambda)P}_i, \quad i=1\dots 2n
$$

Weights:
$$
W_0^{(m)} = \frac{\lambda}{n+\lambda}, \quad
W_0^{(c)} = \frac{\lambda}{n+\lambda} + (1-\alpha^2+\beta), \quad
W_i^{(m,c)} = \frac{1}{2(n+\lambda)}
$$

Time update:
$$
\hat{\mathbf{x}}_{k|k-1} = \sum_{i=0}^{2n} W_i^{(m)} \mathcal{X}_i^{(k|k-1)}
$$
$$
P_{k|k-1} = \sum_{i=0}^{2n} W_i^{(c)}(\mathcal{X}_i - \hat{\mathbf{x}})(\mathcal{X}_i - \hat{\mathbf{x}})^T + Q
$$

Measurement update:
$$
\hat{\mathbf{z}} = \sum W_i^{(m)} \mathcal{Z}_i
$$
$$
S = \sum W_i^{(c)}(\mathcal{Z}_i - \hat{\mathbf{z}})(\mathcal{Z}_i - \hat{\mathbf{z}})^T + R
$$
$$
P_{xz} = \sum W_i^{(c)}(\mathcal{X}_i - \hat{\mathbf{x}})(\mathcal{Z}_i - \hat{\mathbf{z}})^T
$$
$$
K = P_{xz}S^{-1}, \quad \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + K\boldsymbol{\nu}_k, \quad P_k = P_{k|k-1} - KSK^T
$$

UKF preserves nonlinear observability without Jacobian linearization and is therefore the correct sovereign estimator for Indian SSA close-approach, re-entry, and drag-dominated regimes.

## 2. India-Specific Force Model Hardening

The current `PropagateCartesianState` path should be hardened into an explicit force-model stack:

$$
\mathbf{a}_{\text{total}} = \mathbf{a}_{2B} + \mathbf{a}_{J2} + \mathbf{a}_{J3} + \mathbf{a}_{J4} + \mathbf{a}_{\text{drag}} + \mathbf{a}_{\text{3rd}} + \mathbf{a}_{\text{SRP}}
$$

### J2/J3/J4
$$
\mathbf{a}_{J2} = -\frac{3}{2} \frac{\mu J_2 R_E^2}{r^5}
\left[\left(1-5\frac{z^2}{r^2}\right)\mathbf{r} + 2z\mathbf{e}_z\right]
$$
Higher zonals should be included using the same Earth-fixed Earth-centered formulation already used in SSP/SSA tools.

### Atmospheric drag
$$
\mathbf{a}_{\text{drag}} = -\frac{1}{2}\rho v_{\text{rel}}\mathbf{v}_{\text{rel}}\frac{C_DA}{m}
$$
where:
- $\rho$ is resolved by NRLMSISE-00,
- $C_D$ is tuned per Indian spacecraft class,
- $A/m$ comes from geometry catalog metadata,
- $\mathrm{F10.7}$ and $A_p$ are sourced with provenance.

### Third-body perturbations
$$
\mathbf{a}_{\text{3rd}} = \mu_3 \left( \frac{\mathbf{r}_3 - \mathbf{r}}{|\mathbf{r}_3 - \mathbf{r}|^3} - \frac{\mathbf{r}_3}{|\mathbf{r}_3|^3} \right)
$$
for Sun and Moon.

## 3. Required Go Code Deltas

### `pkg/dsp/measurement_filter.go`
Add the cleaned-measurement telemetry contract:
- innovation vector,
- UKF pre/post state,
- UKF pre/post covariance,
- trace and max-eigenvalue metrics,
- drag coefficient estimate,
- solar flux provenance,
- perturbation model version.

### `pkg/dsp/filter_registry.go`
Use a string-based key:
```go
type FilterKey struct {
    ObjectID   string
    SiteID     string
    SensorMode string
}
```
This eliminates cross-object contamination and is compatible with Indian multi-sensor track management.

### `pkg/orbital/fusion.go`
Fusion must remain UKF-only. Do not add EKF fallback. The measurement path should be:
1. resolve keyed filter,
2. clean raw radar measurement,
3. propagate sigma points,
4. update with exact nonlinear radar model,
5. emit cleaned record with pre/post covariance telemetry.

### `pkg/persistence/store.go`
Persist cleaned-measurement telemetry into the measurement projection and signed ontology-event record.

## 4. measurement_records Projection (PostgreSQL)

### Columns
- `id`
- `object_ref`
- `site_id`
- `sensor_mode`
- `ts`
- `payload`
- `signature`
- `payload_digest`
- `ukf_cov_pre` JSONB
- `ukf_cov_post` JSONB
- `trace_p` DOUBLE PRECISION
- `max_eigen_p` DOUBLE PRECISION
- `innovation_mahalanobis` DOUBLE PRECISION
- `innovation_vector` JSONB
- `drag_coeff_estimate` DOUBLE PRECISION
- `solar_flux_f107` DOUBLE PRECISION
- `perturbation_model_version` TEXT

### Migration
See [migrations/001_measurement_records.sql](../migrations/001_measurement_records.sql).

## 5. Keyed Filter Registry

The per-track registry is now the correct isolation boundary.

### Semantics
- key = `object_id + site_id + sensor_mode`
- lazy initialization of one filter per active track
- inactivity cleanup policy should be added as a later enhancement using LRU/TTL semantics

### Integration target
- `measurement_orchestrator.go` should always call `registry.GetOrCreate(key)` before UKF fusion.

## 6. Test and Benchmark Plan

### Required benchmark cases
- LEO solar-maximum drag spike over Indian LEO asset
- low-elevation NETRA-style track near horizon
- MOTR conjunction screening run
- re-entry trajectory over Indian landmass
- filter-registry hot path lookup

### Acceptance criteria
- UKF covariance remains positive definite after update
- Mahalanobis gate pass rate stays above 95% in simulated close approach
- per-update runtime stays below 10 ms on target hardware

### Suggested command
```bash
go test -bench . -benchmem ./pkg/orbital ./pkg/dsp
```

## 7. Implementation Order
1. Keep UKF as the only state estimator.
2. Complete force-model enrichment and provenance telemetry.
3. Keep keyed filter registry as the track isolation boundary.
4. Persist covariance telemetry to measurement_records.
5. Feed the resulting stream into AIML only after calibration and replay validation.

## 8. Repository-Specific Notes
- Current code already uses UKF sigma points and nonlinear measurement update in `pkg/orbital/fusion.go`.
- The current registry and orchestrator layers are the right insertion points for the hardening changes.
- The current persistence layer already signs events; the measurement projection just needs the richer UKF telemetry fields to support downstream AIML.
