# Patent Landscape Analysis — DRQ / SSIS
**Internal Strategic Planning Document** | UBERMASCHINE Pvt Ltd | May 2026
Classification: INTERNAL

---

## Executive Summary

The DRQ/SSIS technology stack has been analyzed against global patent databases (WIPO, USPTO, Google Patents, EPO, CNIPA) to identify potential IP conflicts and filing opportunities.

**Key Findings:**
- **3 of 5 core innovations are CLEAR** — no prior art patents found
- **1 innovation has PARTIAL OVERLAP** — Chinese patent on general IMM for hypersonic tracking
- **1 innovation is LARGELY CLEAR** — generic GNSS spoofing patents exist but NavIC+SSA context is novel

**Critical Urgency:**
- India has **NO grace period** for patent filing after public disclosure
- IAF ADITI submission will become public → must file provisional applications **before** submission
- Consider Section 39 secrecy direction for defence-related inventions

---

## 1. Core Innovation Analysis

### 1.1 DPHNO — Dissipative Port-Hamiltonian Neural Operator for Orbital Propagation

**Status: CLEAR ✅ — No competing patent found**

**Search Scope:**
- Keywords: "port-hamiltonian neural network", "dissipative", "satellite orbit", "orbital propagation", "space situational awareness"
- Databases: WIPO PatentScope, Google Patents, USPTO, EPO, CNIPA
- Timeframe: 2000–2026

**Findings:**
- Zero patents combining Port-Hamiltonian neural networks with satellite orbit prediction
- Research papers exist (arxiv 2023–2025) but **none filed as patents** in SSA context
- The Cholesky-enforced $R = LL^\top$ dissipation applied to satellite dynamics is **genuinely novel patent territory**

**Closest Prior Art:**
- `US20140074766A1` (Numerica Corp, 2013) — uses implicit Runge-Kutta for orbit state propagation
  - **Overlap:** None — pure numerical methods, zero neural network component
  - **Relevance:** Low — different technical approach

**Patentability Assessment:**
- **Novelty:** High — no prior art found
- **Non-obviousness:** Strong — combining Port-Hamiltonian theory with orbital dynamics is non-obvious to those skilled in the art
- **Industrial applicability:** Clear — direct application to space domain awareness systems

**Recommended Claims:**
1. A method for orbital propagation using a dissipative Port-Hamiltonian neural operator
2. A system comprising: Cholesky-enforced dissipation matrix $R = LL^\top$; scalar Hamiltonian network; control coupling network
3. Computer program product implementing the above

---

### 1.2 Residual-Targeting SINDy for Satellite Manoeuvre Detection

**Status: CLEAR ✅ — No competing patent found**

**Search Scope:**
- Keywords: "SINDy", "sparse identification nonlinear dynamics", "satellite maneuver", "maneuver detection", "equation discovery"
- Databases: WIPO, Google Patents, USPTO, EPO

**Findings:**
- No patents found for SINDy applied to satellite manoeuvre detection/discovery
- SINDy base algorithm is published academic work (Brunton et al., 2016 PNAS) — **prior art in literature, not patents**
- This means the base algorithm is freely implementable; your specific application is patentable

**Novel Contribution:**
- **Residual targeting:** $r(t) = \ddot{x}_\text{obs} - \ddot{x}_\text{Keplerian}$ rather than full dynamics $\dot{x} = f(x)$
- **Bootstrap confidence intervals** with 1000-resample validation
- **Hamiltonian invariant hash** for certificate issuance

**Patentability Assessment:**
- **Novelty:** High — residual-targeting approach for SSA is unpatented
- **Non-obviousness:** Moderate — SINDy is known but residual application to Keplerian baseline is non-obvious
- **Industrial applicability:** Clear — satellite anomaly detection for defence

**Recommended Claims:**
1. A method for satellite manoeuvre detection using residual-targeted sparse identification of nonlinear dynamics
2. A system comprising: residual computation module; SINDy regression engine; bootstrap CI generator; Hamiltonian invariant hash generator
3. A computer program product for the above

---

### 1.3 IMM Filter for Hypersonic + Satellite Tracking

**Status: PARTIALLY OVERLAPPED ⚠️ — Prior art exists, variant is novel**

**Search Scope:**
- Keywords: "interacting multiple model", "IMM", "hypersonic", "satellite tracking", "space domain awareness"
- Databases: CNIPA, Google Patents, USPTO

**Prior Art Found:**
- **`CN104199022B`** (Chinese, National University of Defense Technology, 2014)
  - Title: "Target modal estimation based near-space hypersonic velocity target tracking method"
  - Content: Uses IMM for near-space hypersonic tracking with dynamic mode estimation
  - Jurisdiction: China only
  - Status: Granted

**Overlap Analysis:**
- **Overlapped:** General IMM framework for hypersonic tracking
- **NOT covered by CN104199022B:**
  - Bayesian online transition-matrix adaptation (your Dirichlet-Markov approach)
  - Free-energy-based adversarial intent scoring (Active Inference Expected Free Energy)
  - **Unified multi-threat stack** — satellite manoeuvre + HGV + GNSS spoofing + debris sharing one filter architecture

**Jurisdictional Impact:**
- Chinese patents have limited PCT/India enforceability
- Prior art exists but is jurisdictionally and technically narrow

**Patentability Assessment:**
- **Novelty:** Moderate — base IMM is prior art, but your extensions are novel
- **Non-obviousness:** Strong — unified multi-threat architecture is non-obvious
- **Industrial applicability:** Clear

**Recommended Claims:**
1. **Avoid claiming** general IMM framework (prior art)
2. **Claim the novel extensions:**
   - A method for unified multi-threat filtering across satellite, hypersonic, GNSS, and debris domains
   - A system comprising: Bayesian online transition-matrix adaptation; free-energy-based adversarial intent scoring
3. **Carve-around strategy:** "wherein the filter architecture is configured for simultaneous tracking across multiple threat domains"

---

### 1.4 Conformal Prediction for Space Domain Awareness Alerts

**Status: CLEAR ✅ — No competing patent found**

**Search Scope:**
- Keywords: "conformal prediction", "uncertainty quantification", "space domain awareness", "conjunction alert", "coverage certificate"
- Databases: WIPO, Google Patents, USPTO, EPO

**Findings:**
- No patents found anywhere applying conformal prediction to SSA/conjunction alerts
- Conformal prediction is published academic method (Vovk et al., 2005) — freely implementable

**Closest Prior Art:**
- `US6085128A` (1998, OCEAN technique) — orbital covariance estimation
  - **Overlap:** None — classical Kalman-based, no ML, no conformal coverage guarantees

**Novel Contribution:**
- **Distribution-free coverage certificates** on every gRPC alert
- **Mathematically proven marginal coverage:** $P(\text{truth} \in C_\alpha) \geq 1 - \alpha$
- Integration with sovereign air-gapped deployment

**Patentability Assessment:**
- **Novelty:** High — conformal prediction for SSA is unpatented
- **Non-obviousness:** Strong — combining conformal ML with sovereign deployment is non-obvious
- **Industrial applicability:** Clear

**Recommended Claims:**
1. A method for generating conformal prediction intervals for space domain awareness alerts
2. A system comprising: conformal quantile regression engine; coverage certificate generator; gRPC alert integration
3. A method for sovereign deployment with mathematically proven marginal coverage

---

### 1.5 NavIC Dual-Frequency ML Spoofing Detection for SSA

**Status: LARGELY CLEAR ✅ — Generic GNSS patents exist, NavIC+SSA is novel**

**Search Scope:**
- Keywords: "GNSS spoofing detection", "NavIC", "space situational awareness", "machine learning", "neural network"
- Databases: EPO, USPTO, Google Patents

**Prior Art Found:**
- **`EP4024087A2`** — GNSS spoofing detection via bearing/range measurements
  - **Overlap:** None — geometric/analytical approach, not ML
- **`US20200049837A1`** — ML-assisted GNSS location estimation
  - **Overlap:** None — end-user device positioning, not SSA context

**Novel Contribution:**
- **BiLSTM-Attention** on dual-frequency L5/S pseudorange residuals
- **SSA operational context** — protecting a navigation constellation, not end-user positioning
- **Dual-constellation residual architecture** specific to NavIC

**Patentability Assessment:**
- **Novelty:** High — NavIC+SSA context with ML is unpatented
- **Non-obviousness:** Moderate — GNSS spoofing is known, but NavIC constellation protection is non-obvious
- **Industrial applicability:** Clear

**Recommended Claims:**
1. A method for GNSS spoofing detection in a space domain awareness system
2. A system comprising: BiLSTM-Attention network; dual-frequency pseudorange residual processor; constellation-level integrity monitoring
3. A method specifically for NavIC constellation protection using the above

---

## 2. Competitor IP Landscape

### 2.1 Commercial SSA Competitors

| Competitor | Patents Found | Overlap Risk | Notes |
|---|---|---|---|
| **LeoLabs** | No ML/physics patents found | None | Radar-based SSA, no neural network patents |
| **Slingshot Aerospace** | No SSA ML patents; merged with Numerica SDA division (2022) | None | Focus on collision avoidance services |
| **ExoAnalytic** (now Anduril) | No specific ML patents; acquired by Anduril (2026) | None | Global telescope network, no ML IP found |
| **Numerica Corp** | `US20140074766A1` — numerical state propagation (classical) | Low | Implicit Runge-Kutta, no ML component |
| **Kayhan Space** | No patents found in public search | None | Collision avoidance SaaS |

### 2.2 Academic/State Research

| Entity | Patents Found | Overlap Risk | Notes |
|---|---|---|---|
| **Chinese NUDT** | `CN104199022B` — IMM hypersonic tracking | Partial on IMM | See Section 1.3 for carve-around strategy |
| **US DoD/AFRL** | General SSA patents, no ML-physics combination | Low | Classical methods predominately |

### 2.3 Geographic Jurisdiction Analysis

- **United States:** Strong prior art in classical SSA methods, weak in ML-physics
- **Europe:** Limited SSA-specific patents, strong in GNSS
- **China:** Active in hypersonic tracking (NUDT), limited in SSA ML
- **India:** Minimal prior art in SSA — **opportunity for first-to-file**

---

## 3. Filing Recommendations

### 3.1 Priority Tiering

**🟢 GREEN — File Immediately (Highest Priority)**
1. **DPHNO for orbital propagation**
   - No prior art
   - Core innovation
   - PCT application recommended
   - IPC: G06N3/00, B64G1/10

2. **Residual-targeting SINDy + Hamiltonian certificate**
   - Unique combination
   - No prior art
   - PCT application recommended
   - IPC: G06N5/00, G05B17/02

3. **Conformal coverage certificates for conjunction alerts**
   - Completely clear
   - Novel application
   - PCT application recommended
   - IPC: G06N20/00, G06Q10/04

**🟡 YELLOW — File with Careful Claim Scoping**
4. **NavIC ML spoofing detection in SSA context**
   - Carve around generic GNSS spoofing patents
   - Specifically claim SSA operational context
   - India + PCT filing
   - IPC: G01S19/21, G06N3/00

5. **Unified multi-threat IMM stack**
   - Claim unified architecture, NOT base IMM
   - India + PCT filing
   - IPC: G01S13/66, G06N5/00

### 3.2 Filing Strategy

**PCT vs India:**
- **PCT Application:** File within 12 months of provisional, gives 30-month window for national phase entry
- **India First:** File provisional in India first (cheaper, establishes priority date)
- **Recommended:** File provisional in India → PCT within 12 months → national phase in US/EU/China

**Timeline (Critical):**
```
T+0:   File provisional applications in India (Section 39 secrecy direction)
T+30:  File PCT applications
T+365: Complete national phase filings (US, EP, CN)
T-7:   BEFORE IAF submission becomes public
```

**Urgency:**
- India has **NO grace period** — public disclosure immediately bars filing
- IAF ADITI submission will become public → file **before** submission
- Target: File by **May 15, 2026** (10 days from now)

### 3.3 India-Specific Filing Guidance

**Section 39 Secrecy Direction:**
- Applicable to inventions "relating to atomic energy or defence"
- DRQ/SSIS qualifies as defence-related (space domain awareness for IAF)
- **Benefits:**
  - Patent application kept secret
  - No publication until secrecy direction lifted
  - Allows disclosure to IAF without losing patent rights
- **Process:**
  1. File provisional application with IP India
  2. Request Section 39 secrecy direction
  3. Patent Office refers to Ministry of Defence
  4. MoD grants secrecy direction (typically 6–12 months)
  5. Application remains secret until direction lifted

**Recommended Action:**
- File all 5 provisional applications under Section 39
- This allows IAF submission without losing patent rights
- Coordinate with patent attorney experienced in defence filings

---

## 4. Claim Drafting Guidance

### 4.1 DPHNO Claims

**Independent Claim 1 (Method):**
> A computer-implemented method for orbital propagation of a space object, the method comprising:
> obtaining a state vector of the space object;
> computing a time derivative of the state vector using a dissipative Port-Hamiltonian neural operator, wherein the neural operator implements:
>   a canonical symplectic matrix J;
>   a learned dissipation matrix R parameterized as R = LL^T, where L is a Cholesky factor;
>   a scalar Hamiltonian network H parameterized by neural network weights;
>   and a control coupling network g;
> propagating the state vector forward in time using the computed time derivative;
> wherein the Cholesky parameterization of R guarantees positive semi-definiteness, ensuring energy dissipation in the absence of external forcing.

**Independent Claim 2 (System):**
> A system for orbital propagation, comprising:
> one or more processors;
> memory storing instructions that, when executed by the processors, cause the system to:
>   obtain a state vector of a space object;
>   compute a time derivative of the state vector using a dissipative Port-Hamiltonian neural operator;
>   wherein the neural operator comprises:
>     a canonical symplectic matrix J;
>     a learned dissipation matrix R = LL^T, where L is a Cholesky factor;
>     a scalar Hamiltonian network H; and
>     a control coupling network g;
>   propagate the state vector forward in time using the computed time derivative.

**IPC Classification:**
- Primary: G06N3/00 (Arrangements based on computational models)
- Secondary: B64G1/10 (Spacecraft guidance)

### 4.2 Residual-Targeting SINDy Claims

**Independent Claim 1 (Method):**
> A computer-implemented method for detecting satellite manoeuvres, the method comprising:
> obtaining observed trajectory data of a satellite;
> computing Keplerian-predicted trajectory data for the satellite;
> computing residual dynamics as a difference between observed and Keplerian-predicted accelerations;
> applying sparse identification of nonlinear dynamics (SINDy) to the residual dynamics to discover a sparse set of non-Keplerian terms;
> wherein the discovered terms represent thrust signatures indicative of a satellite manoeuvre.

**Independent Claim 2 (Certificate Generation):**
> A method for generating a physics-verified manoeuvre certificate, comprising:
> discovering a sparse dynamical equation for satellite manoeuvre using residual-targeted SINDy;
> computing a Hamiltonian invariant hash from the discovered equation;
> generating a falsifiable prediction for a future time step;
> wherein the certificate is self-invalidating if Hamiltonian drift exceeds a threshold.

**IPC Classification:**
- Primary: G06N5/00 (Computing arrangements using knowledge-based models)
- Secondary: G05B17/02 (Automatic control systems)

### 4.3 Conformal Prediction Claims

**Independent Claim 1 (Method):**
> A computer-implemented method for generating uncertainty-quantified alerts in a space domain awareness system, the method comprising:
> obtaining a prediction for a space object from a machine learning model;
> computing a conformal prediction interval for the prediction using a calibration set;
> wherein the conformal prediction interval satisfies a mathematically proven marginal coverage property: P(truth ∈ C_α) ≥ 1 - α;
> generating an alert comprising the prediction and the conformal prediction interval;
> transmitting the alert via a gRPC interface with a coverage certificate.

**IPC Classification:**
- Primary: G06N20/00 (Machine learning)
- Secondary: G06Q10/04 (Forecasting or optimisation)

### 4.4 NavIC Spoofing Detection Claims

**Independent Claim 1 (Method):**
> A computer-implemented method for detecting GNSS spoofing in a space domain awareness system, the method comprising:
> obtaining dual-frequency pseudorange measurements from a GNSS constellation;
> computing residuals between the dual-frequency measurements;
> processing the residuals using a BiLSTM-Attention neural network to classify a signal condition as normal, jamming, or spoofing;
> wherein the system is configured for monitoring a navigation constellation at the constellation level, not at the end-user device level.

**IPC Classification:**
- Primary: G01S19/21 (GNSS signal integrity)
- Secondary: G06N3/00 (Neural networks)

### 4.5 Unified IMM Stack Claims

**Independent Claim 1 (Method):**
> A computer-implemented method for unified multi-threat tracking, the method comprising:
> maintaining a single Interacting Multiple Model (IMM) filter architecture;
> wherein the IMM filter is configured to simultaneously track objects across multiple threat domains including: satellite manoeuvre detection; hypersonic glide vehicle tracking; GNSS spoofing detection; and debris breakup detection;
> applying Bayesian online transition-matrix adaptation to estimate mode transitions;
> computing a free-energy-based adversarial intent score for each tracked object;
> wherein the unified architecture enables cross-domain threat correlation.

**IPC Classification:**
- Primary: G01S13/66 (Radar tracking)
- Secondary: G06N5/00 (Knowledge-based models)

---

## 5. Risk Assessment

### 5.1 Patent Infringement Risk

| Innovation | Infringement Risk | Mitigation |
|---|---|---|
| DPHNO | None | No prior art |
| Residual-Targeting SINDy | None | No prior art |
| IMM Filter | Low (Chinese patent) | Carve-around strategy, claim novel extensions |
| Conformal Prediction | None | No prior art |
| NavIC Spoofing | Low | Carve around generic GNSS patents, claim SSA context |

### 5.2 Freedom to Operate

**Overall Assessment:** High — DRQ/SSIS can be deployed without significant infringement risk.

**Caveats:**
- Chinese IMM patent has limited enforceability outside China
- Generic GNSS spoofing patents exist but are jurisdictionally narrow
- No competitor holds blocking patents on ML-physics SSA approaches

### 5.3 Defensive Publication Strategy

If filing is delayed, consider defensive publication to prevent competitors from patenting your innovations:
- Publish technical whitepaper on arxiv
- File with IP India as defensive publication
- Establish prior art date

**Not recommended** if filing is possible — patents provide stronger protection.

---

## 6. Recommended Next Steps

### Immediate (This Week)
1. **Engage patent attorney** with experience in defence/Section 39 filings
2. **Prepare provisional applications** for all 5 innovations
3. **File provisional applications** with IP India under Section 39 secrecy direction

### Short-Term (Next 30 Days)
1. **File PCT applications** within 12 months of provisional
2. **Coordinate with IAF** on submission timeline — ensure filing before public disclosure
3. **Prepare technical drawings** and claim sets for national phase filings

### Medium-Term (Next 12 Months)
1. **Complete national phase filings** in US, EP, China
2. **Monitor competitor filings** — watch for blocking patents
3. **File continuation applications** for additional claim sets

---

## Appendix: Patent Citations

### Prior Art Patents Referenced

| Patent Number | Title | Jurisdiction | Year | Relevance |
|---|---|---|---|---|
| CN104199022B | Target modal estimation based near-space hypersonic velocity target tracking method | China | 2014 | Partial overlap on IMM |
| US20140074766A1 | Method and system for propagating the state of an object and its uncertainty | US | 2013 | Classical methods, no ML |
| US20230143771A1 | Space traffic management system | US (Japanese assignee) | 2023 | Data sharing architecture |
| EP4024087A2 | GNSS signal spoofing detection via bearing and/or range measurements | Europe | 2022 | Geometric approach, not ML |
| US20200049837A1 | Machine learning assisted satellite based geolocation | US | 2020 | End-user positioning, not SSA |
| US6085128A | Orbit/covariance estimation and analysis (OCEAN) | US | 1998 | Classical Kalman methods |

---

**Document Control**
- Version: 1.0
- Date: May 5, 2026
- Author: UBERMASCHINE Pvt Ltd
- Classification: INTERNAL
- Distribution: Strategic Planning Team, Patent Counsel
