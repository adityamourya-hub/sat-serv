# Astra-Shakti Development Roadmap
## SSIS Production System — 16-Month Implementation Plan

**Version**: 1.0  
**Date**: April 16, 2026  
**Based on**: `TECHNICAL_RESEARCH_BRIDGE_ANALYSIS.md`  

---

## Executive Summary

This roadmap transforms the **SITUATION_AUDIT → US_DEFENSE_TECH_BRAINSTORM** gap analysis into an actionable 16-month development plan. The goal is to field a production-grade SSA system for the Indian Air Force and ISRO that exceeds all existing competitors (Orekit, Basilisk, Tudatpy, Nyx).

### Key Strategic Decisions

| Decision | Rationale |
|:---|:---|
| **MVP at Month 6** | Engine Runtime + Neural ODE + ATeena-IQNet provides immediate operational value |
| **NavIC reframed** | Bistatic radar not feasible from ground; focus on ionospheric correction |
| **SmartBFT for consensus** | Only production BFT library in Go; Hyperledger Fabric v3 proven |
| **Synthetic data first** | ATeena-IQNet trained on simulator until real data available |
| **Parallel track execution** | 4 workstreams with defined integration points |

### Success Criteria

- **Month 6 (MVP)**: Production gRPC server serving neural-enhanced propagation to 100 concurrent users
- **Month 12 (Beta)**: Multi-sensor fusion (radar + optical + SAR) with <100ms inference latency
- **Month 16 (Production)**: Full Astra-Shakti capability with BFT consensus, ionospheric correction, and explainable AI

---

## Phase 1: Foundation (Months 1-4) — CRITICAL PATH

### Objective
Establish production runtime infrastructure and neural-enhanced propagation. Without this foundation, no AI model can be operationalized.

### Workstream 1.1: Engine Runtime (Lead: Go Team)

**Month 1-2: gRPC Infrastructure**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 1.1 | Protobuf schema definition | `proto/ssis/v1/engine.proto` | Go Lead |
| 1.2 | Observation ingestion service | `IngestObservation` streaming RPC | Go Dev 1 |
| 1.3 | Alert subscription service | `SubscribeAlerts` server streaming | Go Dev 2 |
| 1.4 | Catalog query API | `QueryCatalog` with filtering | Go Dev 1 |
| 2.1 | mTLS authentication layer | Certificate-based auth | Go Lead |
| 2.2 | Audit logging integration | Append-only signed logs | Go Dev 2 |
| 2.3 | Health check & metrics | `/healthz`, Prometheus metrics | Go Dev 1 |
| 2.4 | Integration tests | 90% coverage on gRPC layer | QA |

**Month 3-4: Event Loop & Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 3.1 | Observation queue (channels) | Buffered channel-based queue | Go Dev 1 |
| 3.2 | UKF update loop | Sequential OD pipeline | Go Lead |
| 3.3 | Conjunction screening loop | BVH-based 30s periodic scan | Go Dev 2 |
| 3.4 | ONNX inference loop | Async model serving | Go Dev 1 |
| 4.1 | Alert dispatcher | Priority-based routing | Go Dev 2 |
| 4.2 | Operator UI API (REST fallback) | HTTP JSON endpoints | Go Dev 1 |
| 4.3 | Load testing | 1000 concurrent connections | QA |
| 4.4 | Documentation & deployment guide | `docs/OPERATOR_GUIDE.md` | Tech Writer |

**Milestone 1.1**: Production gRPC server handling 100 concurrent sensor streams

### Workstream 1.2: Neural ODE Propagator (Lead: ML Team)

**Month 1-2: Architecture & Training**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 1.1 | TorchDiffEq integration | `ai/v3/neural_ode_kernel.py` | ML Lead |
| 1.2 | Hybrid model: rKAN + MLP | Physics + neural correction | ML Engineer 1 |
| 1.3 | TLE history dataset | 5 years of public TLEs | Data Engineer |
| 1.4 | Training pipeline | 1000 epochs, validation | ML Engineer 1 |
| 2.1 | Physics-informed loss | Kepler + energy constraints | ML Lead |
| 2.2 | Hyperparameter sweep | Optuna optimization | ML Engineer 1 |
| 2.3 | Extrapolation validation | 7-day prediction accuracy | ML Engineer 2 |
| 2.4 | Model selection | Best checkpoint | ML Lead |

**Month 3-4: Export & Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 3.1 | ONNX export | `models/orbital/neural_ode.onnx` | ML Engineer 1 |
| 3.2 | ONNX Runtime integration | Go bindings test | Go Dev 1 |
| 3.3 | A/B testing vs SGP4 | Accuracy comparison report | ML Lead |
| 3.4 | Latency optimization | <50ms inference | ML Engineer 2 |
| 4.1 | Fallback logic (SGP4 if NN fails) | `PropagatorAxiomONNX` | Go Lead |
| 4.2 | Production deployment | Live on staging server | DevOps |
| 4.3 | Performance monitoring | Dashboard & alerts | ML Engineer 2 |
| 4.4 | Research paper draft | arXiv submission | ML Lead |

**Milestone 1.2**: Neural ODE propagator achieving 2 orders of magnitude better extrapolation than SGP4

### Workstream 1.3: ATeena-IQNet Foundation (Lead: ML Team)

**Month 1-2: Radar Simulator**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 1.1 | ATeena radar model | Antenna pattern, beam steering | Radar Expert |
| 1.2 | Orbital mechanics integration | SGP4 target trajectories | ML Engineer 2 |
| 1.3 | RCS models | Swerling I/III distributions | Radar Expert |
| 1.4 | Clutter & noise models | Ground, ionospheric returns | ML Engineer 2 |
| 2.1 | IQ generation pipeline | `ai/simulator/ateena_iq.py` | ML Engineer 2 |
| 2.2 | Synthetic dataset (100k CPIs) | Labeled range-Doppler maps | Data Engineer |
| 2.3 | Domain randomization | SNR, target count variation | ML Engineer 2 |
| 2.4 | Validation visualization | Range-Doppler plot examples | ML Lead |

**Month 3-4: Neural Architecture**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 3.1 | Learned matched filter | Conv1D replacing classical | ML Engineer 1 |
| 3.2 | Range-Doppler CNN | 2D feature extraction | ML Engineer 1 |
| 3.3 | Temporal LSTM | BiLSTM + attention | ML Engineer 2 |
| 3.4 | Multi-task detection head | [p_det, range, doppler, angle] | ML Lead |
| 4.1 | Training (synthetic data) | P_d > 0.9 @ SNR=0dB | ML Engineer 1 |
| 4.2 | Uncertainty weighting | Kendall & Gal implementation | ML Engineer 2 |
| 4.3 | ONNX export | `models/radar/ateena_iqnet.onnx` | ML Engineer 1 |
| 4.4 | Simulator API for future fine-tuning | gRPC data generation | Go Dev 2 |

**Milestone 1.3**: Synthetic-data-trained ATeena-IQNet with P_d > 0.9 at 0 dB SNR

### Phase 1 Dependencies & Risks

```
Critical Path:
  Engine Runtime (1.1) ────────────────────────┐
       ↑                                       ↓
  Neural ODE (1.2) → ONNX → Integration ───→ MVP
       ↑                                       ↑
  ATeena-IQNet (1.3) → ONNX → Integration ───┘
```

| Risk | Probability | Mitigation |
|:---|:---|:---|
| ONNX Runtime performance issues | Medium | Fallback to SGP4; profile with pprof |
| TorchDiffEq numerical instability | Low | Adaptive step size; Dopri5 → RK4 fallback |
| Synthetic data distribution shift | Medium | Extensive domain randomization; plan for transfer learning |
| gRPC streaming backpressure | Medium | Buffered channels; flow control in protobuf |

---

## Phase 2: Multi-Sensor Fusion (Months 5-8)

### Objective
Integrate heterogeneous sensor inputs (radar, optical, SAR, NavIC) via GNN fusion and deploy production sensor tasking.

### Workstream 2.1: Fusion GNN (Lead: ML Team)

**Month 5-6: Architecture Extension**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 5.1 | Heterogeneous sensor encoders | Radar/Optical/SAR encoders | ML Engineer 1 |
| 5.2 | Graph construction | Proximity + orbit-family edges | ML Engineer 2 |
| 5.3 | GATv2 layer enhancement | Edge-type aware attention | ML Lead |
| 5.4 | Multi-task heads | Conjunction + maneuver + catalog | ML Engineer 1 |
| 6.1 | Training data pipeline | Synthetic multi-sensor scenarios | Data Engineer |
| 6.2 | Model training | 500 epochs, multi-task loss | ML Engineer 2 |
| 6.3 | Explainability integration | SHAP attribution for edges | ML Lead |
| 6.4 | ONNX export & latency test | <100ms inference | ML Engineer 1 |

**Month 7-8: Production Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 7.1 | Go integration | `pkg/ml/inference.go` updates | Go Dev 1 |
| 7.2 | Real-time graph construction | Streaming node/edge updates | Go Dev 2 |
| 7.3 | A/B testing vs baseline | Fusion accuracy report | ML Lead |
| 7.4 | Performance optimization | Batch inference, caching | Go Dev 1 |
| 8.1 | Failure mode testing | Sensor dropout scenarios | QA |
| 8.2 | Operator dashboard | Explainable fusion visualization | Frontend Dev |
| 8.3 | Documentation | `docs/FUSION_ARCHITECTURE.md` | Tech Writer |
| 8.4 | Production deployment | Live on production server | DevOps |

**Milestone 2.1**: Multi-sensor fusion achieving <100ms latency with explainable attribution

### Workstream 2.2: NavIC-SigInt (Reframed) (Lead: GNSS Team)

**Month 5-6: Ionospheric Correction**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 5.1 | NavIC RINEX parser | Dual-frequency observation reader | GNSS Engineer |
| 5.2 | STEC calculation | L5 + S-band TEC computation | GNSS Engineer |
| 5.3 | Chapman layer model | Physics-based regularization | Ionosphere Expert |
| 5.4 | LSTM anomaly detector | Pseudorange residual analysis | ML Engineer 2 |

**Month 7-8: Integration & Validation**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 7.1 | UKF integration | TEC as bias correction | Go Dev 2 |
| 7.2 | Real-time validation | Comparison with IGS GIM | GNSS Engineer |
| 7.3 | Timing anomaly detection | Spoofing/jamming alerts | ML Engineer 2 |
| 7.4 | Production deployment | NavIC correction live | DevOps |
| 8.1 | ISRO coordination | Data sharing agreement | PM |
| 8.2 | Documentation | `docs/NAVIC_INTEGRATION.md` | Tech Writer |

**Milestone 2.2**: NavIC ionospheric correction improving orbit determination accuracy by 30%

### Workstream 2.3: SAR-OrbitNet Kickoff (Lead: SAR Team)

**Month 5-6: ISRO Engagement**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 5.1 | ISRO MoU negotiation | Data access agreement | PM + Legal |
| 5.2 | RISAT technical requirements | ISAR imaging specs | SAR Expert |
| 5.3 | Synthetic ISAR simulator | `ai/simulator/sar_isar.py` | SAR Expert |
| 5.4 | Initial dataset (simulated) | 10k ISAR chips with labels | Data Engineer |
| 6.1 | ResNet-18 backbone | Modified for complex input | ML Engineer 1 |
| 6.2 | Physics-constrained head | Orbital element prediction | ML Lead |
| 6.3 | Training pipeline | Kepler + energy loss | ML Engineer 1 |
| 6.4 | Preliminary results | Validation on simulated data | SAR Expert |

**Milestone 2.3**: ISRO MoU signed; SAR-OrbitNet trained on simulated ISAR data

### Phase 2 Dependencies & Risks

| Risk | Probability | Mitigation |
|:---|:---|:---|
| ISRO data access delays | High | Parallel synthetic data track; no blocker for MVP |
| GNN overfitting to synthetic | Medium | Extensive domain randomization; real data fine-tuning |
| NavIC receiver availability | Medium | Use IGS network as backup; dual-frequency GPS fallback |
| Fusion latency >100ms | Low | Model pruning; quantization to INT8 |

---

## Phase 3: Distributed & Advanced (Months 9-12)

### Objective
Deploy distributed consensus for multi-site operations and complete SAR integration. Achieve Beta readiness.

### Workstream 3.1: BFTL Consensus (Lead: Distributed Systems)

**Month 9-10: SmartBFT Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 9.1 | SmartBFT library integration | `pkg/consensus/smartbft.go` | Distributed Lead |
| 9.2 | Consensus configuration | 4-node test cluster | Distributed Dev |
| 9.3 | State machine replication | Catalog + audit log replication | Distributed Dev |
| 9.4 | Byzantine fault injection | Malicious node simulation | QA |
| 10.1 | Network partition handling | Split-brain prevention | Distributed Lead |
| 10.2 | Performance testing | <100ms consensus latency | QA |
| 10.3 | Air-gap deployment test | Offline cluster operation | DevOps |
| 10.4 | Security audit | Third-party review | Security Lead |

**Month 11-12: Production Deployment**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 11.1 | Multi-site cluster | Bangalore + Delhi nodes | DevOps |
| 11.2 | Disaster recovery | Backup/restore procedures | DevOps |
| 11.3 | Monitoring & alerting | Consensus health dashboard | SRE |
| 11.4 | Documentation | `docs/BFTL_OPERATIONS.md` | Tech Writer |
| 12.1 | Production rollout | 4-node production cluster | DevOps |
| 12.2 | Failover testing | Node failure scenarios | QA |
| 12.3 | Performance validation | <50ms consensus under load | QA |

**Milestone 3.1**: Production BFT consensus across 4 IAF/ISRO sites

### Workstream 3.2: SAR-OrbitNet Completion (Lead: SAR Team)

**Month 9-10: Real Data Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 9.1 | ISRO data pipeline | Secure RISAT data ingestion | SAR Expert |
| 9.2 | Transfer learning | Fine-tune on real ISAR | ML Engineer 1 |
| 9.3 | Domain adaptation | Synthetic → real alignment | ML Lead |
| 9.4 | Accuracy validation | Orbit determination error | SAR Expert |
| 10.1 | ONNX export | `models/sar/sar_orbitnet.onnx` | ML Engineer 1 |
| 10.2 | Go integration | ISAR measurement type | Go Dev 2 |
| 10.3 | UKF fusion | SAR as observation source | Go Dev 1 |
| 10.4 | End-to-end testing | Real SAR → orbit update | QA |

**Month 11-12: Production Hardening**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 11.1 | Characterization head | Object type classification | ML Engineer 2 |
| 11.2 | Production monitoring | ISAR processing metrics | SRE |
| 11.3 | ISRO feedback integration | Accuracy improvements | SAR Expert |
| 11.4 | Documentation | `docs/SAR_INTEGRATION.md` | Tech Writer |
| 12.1 | Full production deployment | SAR-OrbitNet live | DevOps |
| 12.2 | Performance validation | <200ms SAR processing | QA |

**Milestone 3.2**: SAR-OrbitNet processing real RISAT data with <500m orbit accuracy

### Workstream 3.3: Ionospheric Tomography (Lead: Ionosphere Team)

**Month 9-12: Research Track**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 9.1 | 3D voxel grid definition | India-region discretization | Ionosphere Expert |
| 9.2 | GNSS ray geometry | Tomography matrix construction | GNSS Engineer |
| 9.3 | 3D U-Net architecture | Encoder-decoder with skip | ML Engineer 2 |
| 10.1 | Chapman prior integration | Physics-informed loss | Ionosphere Expert |
| 10.2 | Training on IGS data | Global ionosphere maps | Data Engineer |
| 11.1 | Validation vs ionosondes | Ground truth comparison | Ionosphere Expert |
| 11.2 | NavIC-specific refinement | Regional accuracy focus | GNSS Engineer |
| 12.1 | Production integration | 3D TEC for propagation | Go Dev 2 |
| 12.2 | Documentation | `docs/IONOSPHERIC_TOMOGRAPHY.md` | Tech Writer |

**Milestone 3.3**: 3D ionospheric tomography improving LEO orbit accuracy by 20%

### Phase 3 Dependencies & Risks

| Risk | Probability | Mitigation |
|:---|:---|:---|
| SmartBFT performance issues | Low | Fallback to Raft (CFT); profile consensus path |
| ISRO data quality issues | Medium | Synthetic backup; incremental integration |
| Ionospheric model divergence | Medium | Chapman prior regularization; ensemble methods |
| Multi-site network latency | Medium | Asynchronous consensus; eventual consistency |

---

## Phase 4: Production Hardening (Months 13-16)

### Objective
Achieve full Astra-Shakti capability with hardened security, comprehensive documentation, and operational readiness certification.

### Workstream 4.1: System Hardening (Lead: Security & SRE)

**Month 13-14: Security Certification**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 13.1 | Penetration testing | External security audit | Security Lead |
| 13.2 | Fuzz testing | Go fuzzing on all parsers | QA |
| 13.3 | Formal verification | Gobra on collision module | Verification Lead |
| 13.4 | SBOM finalization | Complete dependency inventory | Security Lead |
| 14.1 | CVE remediation | All critical/high patched | SRE |
| 14.2 | Air-gap deployment | Fully offline operation | DevOps |
| 14.3 | Crypto validation | Algorithm compliance check | Security Lead |
| 14.4 | Certification documentation | `docs/SECURITY_CERTIFICATION.md` | Tech Writer |

**Month 15-16: Operational Readiness**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 15.1 | Disaster recovery drills | Failover testing | SRE |
| 15.2 | Runbook completion | All procedures documented | Tech Writer |
| 15.3 | Operator training | IAF/ISRO personnel training | Training Lead |
| 15.4 | Load testing | 10,000 concurrent objects | QA |
| 16.1 | Final acceptance test | End-to-end validation | PM |
| 16.2 | Production handover | Official deployment | DevOps |
| 16.3 | Warranty period start | 12-month support | Support Lead |
| 16.4 | Post-deployment review | Lessons learned | PM |

**Milestone 4.1**: Security certification and operational acceptance by IAF/ISRO

### Workstream 4.2: Advanced Features (Lead: Research Team)

**Month 13-16: Research Integration**

| Week | Task | Deliverable | Owner |
|:---|:---|:---|:---|
| 13.1 | Hyperbolic trajectory detection | Boost-phase tracking | ML Lead |
| 13.2 | Advanced maneuver detection | Impulsive + low-thrust | ML Engineer 2 |
| 14.1 | Deep space extension | Lunar/GEO+ propagation | Orbital Expert |
| 14.2 | Quantum-safe crypto | Post-quantum signatures | Crypto Lead |
| 15.1 | Edge deployment | ARM64 binary for field units | DevOps |
| 15.2 | Satellite onboard algorithms | Embedded inference | ML Lead |
| 16.1 | Research roadmap | Future capability plan | CTO |

**Milestone 4.2**: Advanced research features ready for future integration

---

## Team Structure & Resource Allocation

### Core Team (6 FTE)

| Role | Count | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|:---|:---|:---|:---|:---|:---|
| **Go Engineers** | 2 | 2.0 | 1.5 | 1.0 | 0.5 |
| **ML Engineers** | 3 | 2.5 | 2.5 | 2.0 | 1.0 |
| **ML Lead** | 1 | 0.5 | 0.5 | 0.5 | 0.5 |
| **Domain Experts** (contract) | 2 | 0.5 | 1.0 | 1.0 | 0.5 |

### Extended Team (Consultants)

| Role | Engagement | Duration |
|:---|:---|:---|
| Radar Signal Processing Expert | 20% | Months 1-4 |
| SAR Processing Expert | 30% | Months 5-12 |
| Ionospheric Physicist | 20% | Months 9-12 |
| Distributed Systems Expert | 15% | Months 9-12 |
| Security Auditor | 10% | Months 13-14 |

### Infrastructure Requirements

| Resource | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|:---|:---|:---|:---|:---|
| **GPU Compute** (A100-hours) | 150 | 200 | 100 | 50 |
| **Storage** (TB) | 5 | 20 | 50 | 100 |
| **Network** (Mbps) | 100 | 500 | 1000 | 1000 |
| **VMs/Containers** | 10 | 20 | 30 | 40 |

---

## Budget Estimate (₹ Crores)

### Personnel (16 months)

| Category | Cost |
|:---|:---|
| Core team (6 FTE × ₹30L/year × 1.33 years) | ₹2.40 Cr |
| Consultants (various rates) | ₹0.40 Cr |
| **Subtotal Personnel** | **₹2.80 Cr** |

### Infrastructure (16 months)

| Category | Cost |
|:---|:---|
| GPU compute (500 A100-hours × ₹500/hour) | ₹0.25 Cr |
| Cloud infrastructure | ₹0.15 Cr |
| Development hardware | ₹0.10 Cr |
| **Subtotal Infrastructure** | **₹0.50 Cr** |

### Other Costs

| Category | Cost |
|:---|:---|
| ISRO data agreement (estimated) | ₹0.20 Cr |
| Security audit & certification | ₹0.15 Cr |
| Travel & coordination | ₹0.10 Cr |
| Contingency (10%) | ₹0.38 Cr |
| **Subtotal Other** | **₹0.83 Cr** |

### **Total Budget: ₹4.13 Cr**

---

## Key Performance Indicators (KPIs)

### Technical KPIs

| Metric | MVP (M6) | Beta (M12) | Production (M16) |
|:---|:---|:---|:---|
| **Propagation accuracy** (7-day RMSE) | <100m | <50m | <10m |
| **Inference latency** (p99) | <100ms | <50ms | <20ms |
| **Conjunction screening** (objects/second) | 1,000 | 10,000 | 50,000 |
| **System availability** | 99.0% | 99.5% | 99.9% |
| **Detection probability** (radar) | N/A | P_d > 0.85 | P_d > 0.95 |
| **False positive rate** | <5% | <2% | <1% |

### Operational KPIs

| Metric | Target |
|:---|:---|
| **Catalog objects tracked** | >100,000 |
| **IAF assets monitored** | All (100%) |
| **Alert generation time** | <30 seconds |
| **Operator UI response** | <100ms |
| **Multi-site consensus** | <50ms latency |

---

## Risk Management Matrix

| Risk | Impact | Probability | Mitigation | Owner |
|:---|:---|:---|:---|:---|
| **Critical** |
| Engine runtime fails to scale | High | Low | Load testing; horizontal scaling; caching | Go Lead |
| Neural ODE training divergence | High | Low | Gradient clipping; adaptive LR; ensemble | ML Lead |
| ISRO data access denied | High | Medium | Synthetic-first strategy; alternative data | PM |
| **High** |
| Team attrition | Medium | Medium | Knowledge documentation; cross-training | PM |
| GPU compute shortage | Medium | Low | Cloud fallback; model compression | ML Lead |
| Regulatory delays | Medium | Medium | Early engagement; parallel tracks | PM |
| **Medium** |
| Open-source dependency issues | Low | Medium | Vendoring; SBOM; alternative libs | Security Lead |
| Integration complexity | Medium | Medium | API contracts; integration tests | Go Lead |
| **Low** |
| Documentation delays | Low | Low | Tech writer allocation; templates | Tech Writer |
| Minor scope creep | Low | High | Change control; MVP focus | PM |

---

## Integration Points & Deliverables

### Month 6: MVP Demo

**Capabilities**:
- gRPC server with real-time propagation
- Neural ODE (2 orders better than SGP4)
- ATeena-IQNet (synthetic-trained)
- Existing models (collision, maneuver) serving

**Demo Script**:
1. Ingest live CelesTrak TLEs (834 satellites)
2. Propagate with Neural ODE (show energy conservation)
3. Screen for conjunctions (<50km threshold)
4. Show explainable alert with SHAP attribution

### Month 12: Beta Review

**Capabilities**:
- Multi-sensor fusion (radar + optical + SAR sim)
- NavIC ionospheric correction
- 4-node BFT consensus test cluster
- Operator dashboard v1

**Review Criteria**:
- <100ms inference latency validated
- 99.5% uptime over 30 days
- IAF operator training completed

### Month 16: Production Handover

**Deliverables**:
- Hardened system with security certification
- Complete documentation suite
- Trained IAF/ISRO operators
- 12-month warranty & support

**Acceptance Criteria**:
- All KPIs met
- Security audit passed
- Disaster recovery validated
- Formal acceptance sign-off

---

## Documentation Deliverables

| Document | Owner | Due |
|:---|:---|:---|
| `OPERATOR_GUIDE.md` | Tech Writer | Month 6 |
| `API_REFERENCE.md` | Go Lead | Month 6 |
| `ML_MODEL_CARDS.md` | ML Lead | Month 6 |
| `FUSION_ARCHITECTURE.md` | ML Lead | Month 8 |
| `NAVIC_INTEGRATION.md` | GNSS Engineer | Month 8 |
| `BFTL_OPERATIONS.md` | Distributed Lead | Month 12 |
| `SAR_INTEGRATION.md` | SAR Expert | Month 12 |
| `IONOSPHERIC_TOMOGRAPHY.md` | Ionosphere Expert | Month 12 |
| `SECURITY_CERTIFICATION.md` | Security Lead | Month 16 |
| `RUNBOOK.md` | SRE | Month 16 |

---

## Conclusion

This roadmap transforms the **SITUATION_AUDIT** gap analysis into a concrete 16-month execution plan. The critical insight is that **Engine Runtime + Neural ODE + ATeena-IQNet** (Months 1-6) creates an MVP that already exceeds all existing SSA competitors.

Key success factors:
1. **Parallel execution**: 4 workstreams with clear integration points
2. **Synthetic-first strategy**: No dependency on external data for core capabilities
3. **Honest feasibility assessment**: NavIC bistatic detection correctly reframed
4. **Production focus**: Every milestone includes deployment, not just research

**Go/No-Go Decision Gates**:
- **Month 3**: Engine runtime stress test (10k concurrent objects)
- **Month 6**: MVP demo with IAF stakeholders
- **Month 9**: ISRO data integration decision
- **Month 12**: Beta acceptance test
- **Month 16**: Final production handover

The **₹4.13 Cr budget** and **6-person core team** provide a realistic resource envelope for delivering Astra-Shakti as a sovereign Indian SSA capability.
