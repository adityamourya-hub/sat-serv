# IAF Acceptance Roadmap — SSIS Operational Deployment

**Document ID**: IAF-SSIS-ROADMAP-001  
**Version**: 1.0  
**Date**: April 19, 2026  
**Classification**: RESTRICTED — IAF ADITI 4.0 Challenge Response  
**Stakeholders**: Indian Air Force, DRDO, ISRO, Developer Agency

---

## Executive Summary

This roadmap transforms SSIS from a **research-grade prototype** (current state) to an **IAF-operational sovereign SSA system** in 6 months. It addresses all 9 ADITI 4.0 capabilities (a–i) plus compliance requirements (§2–§6).

**Current State**: Architecture complete, ML models untrained, certification not started  
**Target State**: IAF-certified, trained models, operational deployment ready  
**Critical Path**: ML training pipeline (Month 1–2) → Standards compliance (Month 2–3) → Certification (Month 4–6)

---

## Month-by-Month Roadmap

### 📅 MONTH 1: Foundation & Data Pipeline
**Theme**: "From Prototype to Data-Driven"

#### Week 1–2: Training Infrastructure
| Task | Owner | Deliverable | ADITI Clause |
|------|-------|-------------|--------------|
| Set up H100/A100 cluster (on-prem or PARAM) | Infrastructure | 8× GPU compute node | §2 (scalability) |
| Deploy MLflow + PostgreSQL (air-gapped) | MLOps | Experiment tracking online | §5 (maintainability) |
| Implement Space-Track ingestion pipeline | Data Eng | Daily TLE sync to local DB | (a) Volumetric Scan |
| Build automated maneuver labeling | ML Eng | Auto-label script from TLE gaps | (f) Maneuver Detection |

#### Week 3–4: First Real Models
| Task | Owner | Deliverable | Verification |
|------|-------|-------------|--------------|
| Train Maneuver Detector (Maya) v0.1 | ML Team | First checkpoint with real weights | `pytest ai/v3/test_maya_detector.py` |
| Generate 10M collision scenarios for surrogate | Physics | MC dataset for Pc training | `ls data/collision_mc/ | wc -l` |
| Pretrain Hamiltonian on 2-body problem | Physics | Energy conservation < 1e-12 | `go test ./pkg/physics/...` |
| Implement CDM export (CCSDS 508.0-B-2) | Integration | CDM XML generator | Sample CDM in `/artifacts/cdm/` |

**Month 1 Exit Criteria**:
```bash
# Verification Gate M1
python scripts/verify_training_infrastructure.py  # MLflow + GPUs healthy
ls models/maya_detector/v0.1/  # checkpoint exists
head -100 artifacts/sample_aditi_cdm.xml  # valid CCSDS CDM
```

---

### 📅 MONTH 2: Core Capabilities & Standards
**Theme**: "Industry Standards Integration"

#### Week 5–6: Physics & Propagation
| Task | Owner | Deliverable | Standard |
|------|-------|-------------|----------|
| Integrate JPL DE440 ephemeris (SPICE toolkit) | Physics | Replace Meeus with JPL | SPICE CSPICE |
| Add NRLMSISE-00 atmospheric model | Physics | Drag model upgrade | NRLMSISE-00 |
| Implement SP3 ingestion/export | Integration | SP3-c, SP3-d support | IGS SP3 |
| Add ITU filing database support | Catalog | Orbital slot management | ITU-R S.1645 |

#### Week 7–8: Operational Features
| Task | Owner | Deliverable | ADITI Clause |
|------|-------|-------------|--------------|
| Build UCT (Uncorrelated Track) association | OD | UCT-to-catalog correlation | (g) Characterization |
| Implement batch Orbit Determination (BLS) | OD | Epoch determination capability | (c) Conjunction Analysis |
| Sensor network topology configuration | Sensors | IAF radar grid JSON schema | (a) Volumetric Scan |
| Sovereign TLE generation pipeline | OD | Produce TLEs from observations | (g) Cataloguing |

**Month 2 Exit Criteria**:
```bash
# Verification Gate M2
go test ./pkg/orbital/... -run TestDE440  # JPL ephemeris passes
./scripts/generate_sovereign_tle.py --radar-data netra_2026.tle  # produces valid TLE
head artifacts/sample_sp3.sp3  # SP3 format validated against GFZ
```

---

### 📅 MONTH 3: Advanced Analytics & Security
**Theme**: "Intelligence & Hardening"

#### Week 9–10: ML Model Completion
| Task | Owner | Deliverable | Target Metric |
|------|-------|-------------|---------------|
| Complete Collision Surrogate training | ML | Production-ready ONNX | 10× speedup vs MC, <1% error |
| Train Sensor Tasking RL policy | RL | PPO checkpoint beating heuristic | 20% coverage improvement |
| Pretrain POL Transformer (JEPA) | ML | 1M light curve pretraining | Rotation period accuracy ±5% |
| Train Re-entry Predictor | ML | Decay time <5% error | Validated on 50 historic re-entries |

#### Week 11–12: Security Hardening
| Task | Owner | Deliverable | Standard |
|------|-------|-------------|----------|
| FIPS 140-3 Mode (P-384 alternative) | Crypto | Dual-mode Ed25519 + P-384 | FIPS 140-3 |
| Post-Quantum Crypto layer (Dilithium3) | Crypto | PQ-hybrid signatures | NIST PQC |
| Formal verification (Gobra completion) | Verification | All core proofs | Gobra/Why3 |
| STIG-compliant Kubernetes manifests | DevOps | Hardened deployment | DISA STIG |

**Month 3 Exit Criteria**:
```bash
# Verification Gate M3
python scripts/evaluate_collision_surrogate.py  # accuracy report
go test ./pkg/ml/onnx/... -run TestRealModels  # NO stub fallbacks
cat security/fips_mode_report.md  # FIPS validation passed
make govulncheck  # zero critical CVEs
```

---

### 📅 MONTH 4: Certification Preparation
**Theme**: "Audit & Documentation"

#### Week 13–14: Certification Artifacts
| Task | Owner | Deliverable | Standard |
|------|-------|-------------|----------|
| Common Criteria EAL4+ documentation | Security | ST, SAR, ADV, AGD docs | ISO/IEC 15408 |
| MIL-STD-882E hazard analysis | Safety | SHA, SSHA, SSHA traceability | MIL-STD-882E |
| ISO 27001 audit preparation | Security | ISMS documentation | ISO/IEC 27001:2022 |
| RM F (Risk Management Framework) | Security | Threat model + POA&M | NIST SP 800-37 |

#### Week 15–16: Validation & Integration
| Task | Owner | Deliverable | Partner |
|------|-------|-------------|---------|
| Cross-validation with TUDAT/GMAT | Physics | Accuracy comparison report | ISRO/IIT |
| Integration with NETRA radar | Integration | Live data ingestion | ISRO |
| Interop test with ISRO SSA systems | Integration | Data exchange validated | ISRO |
| Performance benchmark (100k objects) | Performance | Sub-second screening sustained | IAF |

**Month 4 Exit Criteria**:
```bash
# Verification Gate M4
ls docs/certification/eal4/  # all CC docs present
./scripts/validate_vs_tudat.py  # <0.1% position error
cat docs/iso27001/readiness_report.md  # audit-ready
```

---

### 📅 MONTH 5: Operational Integration
**Theme**: "Live Fire Testing"

#### Week 17–18: IAF Network Integration
| Task | Owner | Deliverable | Environment |
|------|-------|-------------|-------------|
| Deploy to IAF network segment | DevOps | Air-gapped operational instance | IAF Data Centre |
| Integrate IAF threat library | Intel | Curated adversary RSO database | Classification |
| Connect to IAF sensor grid (MOTR, NETRA) | Sensors | Real-time radar feed | MOTR Station |
| NOTAM/airspace integration | Integration | Suborbital threat → air ops | IAF ATC |

#### Week 19–20: Live Fire Exercise
| Task | Owner | Deliverable | Scenario |
|------|-------|-------------|----------|
| Red-team penetration testing | Security | Pen-test report | White-hat hackers |
| Conjunction drill with ISRO | Ops | Simulated collision avoidance | LEO satellite |
| HGV tracking exercise | Ops | Live hypersonic test (if available) | Test range |
| Operator training program | Training | 20 IAF operators certified | IAF School |

**Month 5 Exit Criteria**:
```bash
# Verification Gate M5
curl -k https://ssis.iaf.mil:50051/health  # IAF instance responding
kubectl get pods -n ssis  # all services green
cat reports/live_fire_exercise_2026.md  # passed all scenarios
```

---

### 📅 MONTH 6: Final Acceptance
**Theme**: "IAF Certification"

#### Week 21–22: Final Certification
| Task | Owner | Deliverable | Authority |
|------|-------|-------------|-----------|
| Common Criteria EAL4+ evaluation | Security | CC certificate | Indian ITSEC |
| IAF acceptance testing | IAF | UAT sign-off | Air Force |
| DRDO technical audit | DRDO | Audit clearance | DRDO |
| Sovereignty verification | Legal | Indigenous component cert | Ministry |

#### Week 23–24: Handover
| Task | Owner | Deliverable | Status |
|------|-------|-------------|--------|
| Knowledge transfer documentation | Dev | Complete technical docs | Archived |
| Source code escrow | Legal | Source + build system | National archive |
| Maintenance agreement | Legal | 5-year support contract | Signed |
| Operational handover ceremony | All | IAF operational declaration | **COMPLETE** |

**Month 6 Exit Criteria**:
```bash
# Final Verification Gate M6
cat docs/certification/common_criteria_certificate.pdf  # PHYSICAL CERT
./scripts/final_acceptance_check.sh  # all ADITI (a–i) pass
kubectl get deployment ssis-engine -n ssis-production  # RUNNING
```

---

## ADITI 4.0 Requirement Mapping

| Clause | Capability | Implementation | Month | Verification Test |
|--------|------------|----------------|-------|-------------------|
| **(a)** | Volumetric Scan | `pkg/proximity/volumetric.go` + BVH | 1–2 | `go test ./pkg/spatial/...` + 100k catalog load |
| **(b)** | Neighbourhood Watch | `pkg/proximity/neighbourhood.go` + intent inference | 3 | `scripts/scenario_redteam.sh` (4 scenarios) |
| **(c)** | Conjunction Analysis | `pkg/collision/foster.go` + CDM export | 1–2 | CDM validation vs Space-Track |
| **(d)** | Collision Detection | `pkg/proximity/collision.go` + Monte Carlo | 1 | Monte Carlo Pc accuracy test |
| **(e)** | Blind Spot Detection | `pkg/sensors/blindspot.go` | 2 | PoD < 0.05 coverage analysis |
| **(f)** | Maneuver Detection | `ai/v3/maya_detector.py` + `pkg/ml/pinn_maneuver.go` | 1–3 | Kelecy AMOS benchmark |
| **(g)** | Characterization | `pkg/fusion/*.go` + UCT association | 2 | UCT correlation rate > 95% |
| **(h)** | Re-entry Analysis | `pkg/hypersonic/reentry.go` | 2–3 | 50 historic re-entry validation |
| **(i)** | Rapid Processing | `pkg/hypersonic/rapid.go` + IMM | 1 | Sub-second 30k propagation |
| **§2** | Network compatibility | Kubernetes + IAF network | 5 | IAF network deployment |
| **§3** | Open source data | Space-Track, CDDIS integration | 1 | Data pipeline operational |
| **§4** | SDLC + certification | Full documentation + CC EAL4+ | 4–6 | CC certificate |
| **§5** | Scalability | Axiom-Zero + BVH | 1 | 100k object sustained |
| **§6** | Perpetual licensing | Apache 2.0 + sovereign terms | 6 | Legal sign-off |

---

## Resource Requirements

### Personnel (Full-Time Equivalents)

| Role | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
|------|---------|---------|---------|---------|---------|---------|
| **Project Manager** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **ML Engineers** | 3.0 | 3.0 | 3.0 | 2.0 | 1.0 | 0.5 |
| **Physics/Astrodynamics** | 2.0 | 2.0 | 2.0 | 1.0 | 0.5 | 0.5 |
| **Security Engineers** | 0.5 | 1.0 | 2.0 | 2.0 | 1.0 | 0.5 |
| **DevOps/Infra** | 1.0 | 1.0 | 1.5 | 2.0 | 2.0 | 1.0 |
| **Integration Engineers** | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 2.0 |
| **QA/Testing** | 0.5 | 1.0 | 1.5 | 2.0 | 2.0 | 2.0 |
| **Technical Writers** | 0.5 | 1.0 | 1.5 | 2.0 | 1.0 | 1.0 |
| **TOTAL FTE** | **9.0** | **11.0** | **14.0** | **14.0** | **11.5** | **9.0** |

### Hardware & Infrastructure

| Component | Specification | Cost (₹) | Month |
|-------------|---------------|----------|-------|
| **GPU Cluster** | 8× H100 80GB + InfiniBand | 4,00,00,000 | 1 |
| **Storage Server** | 100 TB NVMe + 500 TB HDD | 25,00,000 | 1 |
| **Network Equipment** | 100 Gbps switches, firewalls | 15,00,000 | 1 |
| **Air-Gapped MLOps** | MLflow, PostgreSQL, DVC | 5,00,000 | 1 |
| **IAF Deployment** | 3× production nodes (redundant) | 30,00,000 | 5 |
| **TOTAL HARDWARE** | | **4,75,00,000** | |

### Software & Licenses

| Item | Cost (₹) | Purpose |
|------|----------|---------|
| SPICE Toolkit (NASA) | Free | JPL ephemeris |
| TUDAT (TU Delft) | Free | Validation baseline |
| GMAT (NASA) | Free | Mission design validation |
| Orekit (CS) | Free | Java validation (optional) |
| Common Criteria Evaluation | 50,00,000 | EAL4+ certification |
| ISO 27001 Audit | 10,00,000 | ISMS certification |
| **TOTAL CERTIFICATION** | **60,00,000** | |

### Total Budget Estimate

| Category | Amount (₹) |
|----------|------------|
| Personnel (6 months) | 6,00,00,000 |
| Hardware & Infrastructure | 4,75,00,000 |
| Certification & Audit | 60,00,000 |
| Contingency (15%) | 1,70,25,000 |
| **TOTAL** | **13,05,25,000** |
| | **~$15.6M USD** |

---

## Risk Register

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| **GPU procurement delays** | Medium | High | Use PARAM/IITM cluster interim | PM |
| **ML training convergence issues** | Medium | High | Ensemble + fallback heuristics | ML Lead |
| **IAF network integration blocks** | High | High | Early engagement (Month 1) | Integration Lead |
| **Common Criteria timeline slip** | Medium | Critical | Start documentation Month 1 | Security Lead |
| **ISRO data sharing delays** | Medium | High | MoU signed Month 1 | PM |
| **Key personnel departure** | Low | High | Cross-training + documentation | PM |
| **Adversarial ML vulnerabilities** | Medium | High | Red-team testing Month 5 | Security |
| **Performance degradation at scale** | Low | High | Stress test Month 2 | Performance |

---

## IAF Stakeholder Engagement Schedule

| Month | Activity | Stakeholders | Deliverable |
|-------|----------|--------------|-------------|
| 1 | Kickoff meeting | IAF Project Director | Project charter |
| 1 | Data sharing MoU | ISRO/DRDO | Signed agreement |
| 2 | Architecture review | IAF Technical Committee | Design approval |
| 3 | Security audit kickoff | IAF Cyber Security | Audit plan |
| 4 | Integration test planning | IAF Ops + NETRA | Test plan |
| 5 | Live fire exercise | IAF Command | Exercise report |
| 6 | Final acceptance | IAF Chief + DRDO | Operational declaration |

---

## Success Metrics (KPIs)

### Technical KPIs

| Metric | Target | Current | Month 6 Target |
|--------|--------|---------|----------------|
| Catalog objects tracked | 100,000 | Architecture ready | 100,000 sustained |
| Propagation time (30k objects) | <1 second | Not tested | <1 second verified |
| Conjunction screening | n log n | O(n log n) | 100k in <5 seconds |
| Maneuver detection accuracy | >95% | 0% (untrained) | >95% on test set |
| Re-entry prediction error | <5% | N/A | <5% on 50 events |
| Collision Pc accuracy | <1% error | N/A | <1% vs Monte Carlo |
| System uptime | >99.9% | N/A | >99.9% over 30 days |

### Operational KPIs

| Metric | Target |
|--------|--------|
| CDM generation latency | <60 seconds from observation |
| Alert generation | <30 seconds for critical conjunctions |
| UCT correlation rate | >95% within 24 hours |
| Operator training completion | 20 IAF personnel certified |
| Mean time to resolution (MTTR) | <4 hours for critical bugs |

### Compliance KPIs

| Metric | Target |
|--------|--------|
| Common Criteria | EAL4+ certificate achieved |
| ISO 27001 | Full certification |
| MIL-STD-882E | All hazards mitigated |
| FIPS 140-3 | Mode operational |
| Source escrow | Deposited with National Archive |

---

## Dependencies & Prerequisites

### External Dependencies (Critical Path)

| Dependency | Lead Time | Mitigation |
|------------|-----------|------------|
| H100/A100 procurement | 8–12 weeks | Order Month 1, use cloud interim |
| ISRO data sharing MoU | 4–6 weeks | Legal engagement immediately |
| Common Criteria lab booking | 8–12 weeks | Book Month 1, docs ready Month 4 |
| IAF network access | 6–8 weeks | Early engagement, VPN interim |
| Security clearance for team | 4–6 weeks | Start background checks Month 0 |

### Internal Dependencies

1. **Month 2 needs Month 1 data pipeline** — sequential
2. **Month 3 training needs Month 1–2 datasets** — sequential
3. **Month 4 certification needs Month 3 hardening** — sequential
4. **Month 5 integration needs Month 4 validated system** — sequential

---

## Contingency Plans

### Plan B: Delayed GPU Access
If H100s delayed beyond Month 2:
1. Use Google Cloud/ AWS GPU instances (controlled egress)
2. Partner with IIT Madras PARAM cluster
3. Reduce training scope (focus on Maneuver + Collision only)

### Plan B: ML Training Fails
If models don't converge:
1. Fall back to physics-based heuristics (already implemented)
2. Use ensemble of smaller models
3. Reduce problem scope (e.g., GEO-only for maneuver detection)
4. Extend Month 2 by 2 weeks

### Plan B: IAF Network Blocks
If network integration fails:
1. Deploy to DMZ (demilitarized zone) with manual data sync
2. Reduce real-time requirements to batch processing
3. Use portable hardware (ruggedized servers)

---

## Appendices

### A. Detailed Training Schedule (Month 1–3)

See `docs/training/TRAINING_EXECUTION_GUIDE.md` (existing document to be updated)

### B. Certification Document Templates

Located in `docs/certification/templates/`
- CC_ST.adoc — Security Target template
- CC_SAR.adoc — Security Assurance Requirements
- ISO27001_gap_analysis.xlsx
- STIG_checklist.json

### C. IAF Contact Directory

(Restricted — separate document)

### D. Hardware Vendor Quotes

(Restricted — procurement document)

---

## Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Project Manager** | | | |
| **Technical Lead** | | | |
| **Security Lead** | | | |
| **IAF Project Director** | | | |
| **DRDO Representative** | | | |

---

**Document Control**

- **Version History**: 1.0 — Initial roadmap (April 19, 2026)
- **Next Review**: Weekly during Month 1, bi-weekly thereafter
- **Change Control**: All changes require PM + IAF approval
- **Distribution**: Core team, IAF Project Office, DRDO liaison

---

*This roadmap aligns with the verification gates in `i-needed-depper-technical-goofy-sundae.md` and addresses all ADITI 4.0 requirements for IAF acceptance.*
