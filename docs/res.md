# Strategic SOTA Audit: Global Astrodynamics Substrates

This document contains a comprehensive audit of the global state-of-the-art (SOTA) in orbital mechanics and astrodynamics libraries as of 2024-2026. This research serves as the technical baseline for the Sovereign Space Intelligence Substrate (SSIS).

### 1–15: Core Astrodynamics & Orbital Propagation Libraries

| # | Repository | Link | Why Relevant to PS24 |
|---|------------|------|----------------------|
| 1 | CS-SI/Orekit | https://github.com/CS-SI/Orekit | Industry-standard low-level space dynamics (Java) – volumetric scan, propagation, events |
| 2 | tudat-team/tudatpy | https://github.com/tudat-team/tudatpy | TU Delft Astrodynamics Toolbox (Python/C++) – full mission analysis & estimation |
| 3 | nyx-space/nyx | https://github.com/nyx-space/nyx | High-fidelity Rust/Python astrodynamics toolkit |
| 4 | jacobwilliams/Fortran-Astrodynamics-Toolkit | https://github.com/jacobwilliams/Fortran-Astrodynamics-Toolkit | Comprehensive modern Fortran orbital mechanics library |
| 5 | open-space-collective/open-space-toolkit-astrodynamics | https://github.com/open-space-collective/open-space-toolkit-astrodynamics | Orbit, attitude, access & mission analysis |
| 6 | ssmichael1/satkit | https://github.com/ssmichael1/satkit | High-performance Rust + Python orbital dynamics |
| 7 | AVSLab/basilisk | https://github.com/AVSLab/basilisk | Full astrodynamics simulation framework |
| 8 | esa/pykep | https://github.com/esa/pykep | ESA astrodynamics & trajectory optimisation |
| 9 | duncaneddy/brahe | https://github.com/duncaneddy/brahe | Modern practical astrodynamics library |
| 10 | lox-space/lox | https://github.com/lox-space/lox | Oxidized (Rust) astrodynamics with Python bindings |
| 11 | ATTron/astroz | https://github.com/ATTron/astroz | Fast SGP4/SDP4 + orbital mechanics in Zig |
| 12 | IO-Aerospace-software-engineering/Astrodynamics | https://github.com/IO-Aerospace-software-engineering/Astrodynamics | .NET SPICE-powered astrodynamics |
| 13 | AlabamaASRL/asset_asrl | https://github.com/AlabamaASRL/asset_asrl | Trajectory design & optimal control |
| 14 | boinor/boinor | https://github.com/boinor/boinor | Pure Python interactive astrodynamics |
| 15 | strath-ace/smart-astro | https://github.com/strath-ace/smart-astro | Astrodynamics toolbox for reusable core routines |
| 25 | ssmichael1/astrokit | https://github.com/ssmichael1/astrokit | Astronomy & satellite SGP4 toolkit |
| 26 | ibm/spacetech-ssa | https://github.com/ibm/spacetech-ssa | End-to-end ML pipeline for LEO SSA & conjunctions |
| 27 | esa/cascade | https://github.com/esa/cascade | Large-scale conjunction & collision detection |
| 28 | ut-astria/caspy | https://github.com/ut-astria/caspy | Conjunction analysis & screening (CCSDS OEM → CDM) |
| 29 | david-rc-dayton/collision-check | https://github.com/david-rc-dayton/collision-check | Satellite collision probability estimator |
| 30 | arma1978/conjunction | https://github.com/arma1978/conjunction | Conjunction data & Alfano Pc calculation |
| 31 | RoboticExplorationLab/cvx-spacecraft-cola | https://github.com/RoboticExplorationLab/cvx-spacecraft-cola | Conjunction analysis & manoeuvre planning |
| 32 | rrovetto/space-situational-awareness-domain-ontology | https://github.com/rrovetto/space-situational-awareness-domain-ontology | SSA domain ontology (cataloguing & characterisation) |
| 33 | AshHarvey/ssa-gym | https://github.com/AshHarvey/ssa-gym | OpenAI Gym environment for SSA sensor tasking |
| 34 | NTU-ICG/SpaceDet | https://github.com/NTU-ICG/SpaceDet | Space-based image dataset for SSA object detection |
| 35 | cognitiveailab/ssa-corpus | https://github.com/cognitiveailab/ssa-corpus | SSA event extraction corpus |

---

### SSIS Competitive Edge vs. Global SOTA

While the libraries above provide massive foundational value, SSIS is uniquely architected for the **IAF ADITI 4.0** requirements in the following ways:

1. **Indigenous Sovereignty**: SSIS implements a Go-native kernel (Phase 0) that eliminates reliance on foreign-hosted JVM environments (Orekit) or complex C++ toolchains (Basilisk), ensuring ease of certification for air-gapped deployment.
2. **Deterministic Audit (HMAC)**: Unlike generic libraries, SSIS enforces cryptographic integrity at the kernel level. Every collision report and conjunction assessment is HMAC-SHA256 signed, ensuring bit-identical replay capability.
3. **Physics-Informed Neuro-Topological Analysis**: SSIS goes beyond standard propagation by integrating PINO (Physics-Informed Neural Operators) and topological void detection (homological algebra) for automated maneuver detection and blind-spot mapping.
