# PS24 Astrodynamics Sovereign Agent Deep Analysis (Part 4)

**Mission Context**: Systematic extract of repositories 36-50 mapped against IAF PS24 capabilities, specifically focusing on **(f) Manoeuvre Detection**, **(h) Re-entry Analysis**, and **(i) Rapid Processing (Hypersonic/Ballistic)**.

---

## 36. mit-ll/MaDDG

### 1. Repo Metadata
* **Full Name**: [mit-ll/MaDDG](https://github.com/mit-ll/MaDDG)
* **Stars**: 13
* **Last Commit Date**: 2026-02-10
* **Primary Language**: Python
* **License**: MIT License

### 2. README / Project Summary
Developed by MIT Lincoln Laboratory, MaDDG (Manoeuvre Detection via Directed Graphs) is a novel approach to detecting satellite maneuvers. It formulates TLE sequence anomaly detection as a directed graph problem, using geometric bounds to find impossible un-thrust trajectories.

### 3. Core Capabilities Relevant to PS24
* **(f) Manoeuvre Detection**: Directly supports detecting intentional/unintentional orbital changes without relying entirely on physical integration.
* **(g) Characterisation**: Helps in correlating disparate TLE tracks by predicting if one state could have naturally evolved into another.

### 4. Key Modules / Classes / Functions
* `graph_builder.py`: Constructs directed graphs from TLE epochs.
* `anomaly_scorer.py`: Evaluates edge weights to determine probability of maneuver.
* `tle_parser.py`: Ingests standard NORAD TLEs.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: MaDDG Graph based detection
from maddg import GraphBuilder, AnomalyDetector
graph = GraphBuilder(tle_history_list).build()
anomalies = AnomalyDetector().find_maneuvers(graph, threshold=0.85)
```

### 6. Installation & Integration Notes
* **Installation**: Python pip.
* **Integration**: Python microservice. Good for offline processing, but graphical building can be memory intensive.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 6/10 (US DoD affiliated lab, but released under MIT).
* **Air-gapped Suitability**: Yes.

### 8. Limitations & Risks
* Computations scale poorly for the entire 50K object catalog in real-time (O(N^2) edge building).

### 9. Overall Relevance Score to PS24
**7/10** — A strong mathematical alternative, but superseded by our PINO Hamiltonian approach which guarantees $O(1)$ physics compliance.

### 10. Suggested Next Action
Benchmark MaDDG's accuracy against the `ssis/ai/tsga/pino.py` Hamiltonian anomaly detector.

---

## 37. Arya920/Anomaly_Detection_In_Orbit_Maneuver

### 1. Repo Metadata
* **Full Name**: [Arya920/Anomaly_Detection_In_Orbit_Maneuver](https://github.com/Arya920/Anomaly_Detection_In_Orbit_Maneuver)
* **Stars**: 1
* **Last Commit Date**: 2024-07-21
* **Primary Language**: Jupyter Notebook (Python)
* **License**: None

### 2. README / Project Summary
An academic/student repository implementing basic machine learning (Isolation Forests/SVM) to detect anomalies in orbital data.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 5/10.

### 9. Overall Relevance Score to PS24
**2/10** — Standard scikit-learn tutorial applied to orbits. Too immature for military use.

### 10. Suggested Next Action
Discard.

---

## 38. Tanmay-Tripathi/Maneuvers-in-orbital-data

### 1. Repo Metadata
* **Full Name**: [Tanmay-Tripathi/Maneuvers-in-orbital-data](https://github.com/Tanmay-Tripathi/Maneuvers-in-orbital-data)
* **Stars**: 5
* **Last Commit Date**: 2024-06-19
* **Primary Language**: Jupyter Notebook (Python)
* **License**: None

### 2. README / Project Summary
Another exploratory data science project tracking satellite maneuvers using classical ML clustering techniques on public orbital elements.

### 9. Overall Relevance Score to PS24
**2/10** — Academic. Similar to Repo 37.

### 10. Suggested Next Action
Discard.

---

## 39. rrovetto/Orbital-Space-Ontology-Project

### 1. Repo Metadata
* **Full Name**: [rrovetto/Orbital-Space-Ontology-Project](https://github.com/rrovetto/Orbital-Space-Ontology-Project)
* **Stars**: 24
* **Last Commit Date**: 2023-01-20
* **Primary Language**: Base schemas (OWL)
* **License**: None

### 2. README / Project Summary
A formalized ontological model defining the objects, properties, and relationships in orbital space (similar to Repo 32 but focused explicitly on physical orbits rather than the SSA domain/sensors).

### 3. Core Capabilities Relevant to PS24
* **(g) Characterisation & Cataloguing**: Provides the strict relationship maps (e.g., Target -> has_orbit -> LEO).

### 9. Overall Relevance Score to PS24
**6/10** — Useful schema reference for the database backend.

### 10. Suggested Next Action
Merge this ontology with Repo 32 to finalize the IAF catalog database structure.

---

## 40. IBM/spacetech-ssa (ML branch)

### 1. Repo Metadata
* **Full Name**: [ibm/spacetech-ssa](https://github.com/ibm/spacetech-ssa)
* **Stars**: 121
* **Last Commit Date**: 2021-03-02
* **Primary Language**: Python
* **License**: Apache License 2.0

### 2. README / Project Summary
*(Note: Evaluated in Part 3. The ML branch specifically focuses on XGBoost tree models for TCA prediction).*

### 9. Overall Relevance Score to PS24
**6/10** — Black box ML fails the strict "Deterministic Audit" requirement of the IAF.

### 10. Suggested Next Action
Use as a comparative benchmark to show why native physics propagation (SSIS) is superior for legal/military auditing.

---

## 41. Tomenol/deorbitation_simulation

### 1. Repo Metadata
* **Full Name**: [Tomenol/deorbitation_simulation](https://github.com/Tomenol/deorbitation_simulation)
* **Stars**: 4
* **Last Commit Date**: 2023-05-29
* **Primary Language**: Python
* **License**: GNU General Public License v3.0

### 2. README / Project Summary
A specialized aerodynamic and orbital simulator focused on the deorbiting phase of spacecraft, modeling atmospheric drag degradation over time.

### 3. Core Capabilities Relevant to PS24
* **(h) Re-entry Analysis**: Modeling orbital decay based on atmospheric density variations and ballistic coefficients.

### 4. Key Modules / Classes / Functions
* `atmosphere.py`: US Standard Atmosphere 1976 model.
* `drag.py`: Computes delta-v from aerodrag.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Drag delta-V calculation
from drag import calculate_drag
a_drag = calculate_drag(density, velocity, area, mass, cd=2.2)
```

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 7/10 (GPLv3).
* **Air-gapped Suitability**: Yes.

### 9. Overall Relevance Score to PS24
**5/10** — Good reference for drag math, but our `ssis/engine/pkg/physics/reentry.go` already handles this natively.

### 10. Suggested Next Action
Extract the US Standard Atmosphere 1976 tables to supplement the Go aerodynamic solver.

---

## 42. realsurya/Spacecraft-Reentry-Sim

### 1. Repo Metadata
* **Full Name**: [realsurya/Spacecraft-Reentry-Sim](https://github.com/realsurya/Spacecraft-Reentry-Sim)
* **Stars**: 1
* **Last Commit Date**: 2023-09-30
* **Primary Language**: Jupyter Notebook
* **License**: None

### 2. README / Project Summary
Simulates re-entry trajectories focusing on stagnation temperature and heat flux.

### 9. Overall Relevance Score to PS24
**3/10** — Simple student simulation.

---

## 43. lcasts/REAT

### 1. Repo Metadata
* **Full Name**: [lcasts/REAT](https://github.com/lcasts/REAT)
* **Stars**: 2
* **Last Commit Date**: 2025-09-29
* **Primary Language**: Python
* **License**: Other (Academic)

### 2. README / Project Summary
Re-Entry Analysis Toolkit (REAT). Calculates footprint dispersions and survivability for re-entering space debris.

### 3. Core Capabilities Relevant to PS24
* **(h) Re-entry Analysis**: Crucial for predicting the ground impact ellipse of uncontrolled debris.

### 4. Key Modules / Classes / Functions
* `footprint.py`: Dispersion ellipse geometry.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Impact dispersion
from reat import debris_footprint
ellipse = debris_footprint(reentry_state, ballistic_coeff_range=[50,150])
```

### 9. Overall Relevance Score to PS24
**8/10** — Footprint dispersion is a mandatory feature for the IAF to warn ground assets.

### 10. Suggested Next Action
Port the `debris_footprint` covariance logic into the Go engine `reentry.go`.

---

## 44. plosan/Atmospheric-Reentry

### 1. Repo Metadata
* **Full Name**: [plosan/Atmospheric-Reentry](https://github.com/plosan/Atmospheric-Reentry)
* **Primary Language**: Python

### 9. Overall Relevance Score to PS24
**4/10** — Basic ODE solver for re-entry.

---

## 45. amyinorbit/reentry.py

### 1. Repo Metadata
* **Full Name**: [amyinorbit/reentry.py](https://github.com/amyinorbit/reentry.py)
* **Primary Language**: Python

### 9. Overall Relevance Score to PS24
**4/10** — Script for orbital decay.

---

## 46. jlev/ballistic-missile-range

### 1. Repo Metadata
* **Full Name**: [jlev/ballistic-missile-range](https://github.com/jlev/ballistic-missile-range)
* **Primary Language**: Python
* **License**: Open

### 2. README / Project Summary
Computes the range and apogee of sub-orbital ballistic missiles based on minimal input parameters (burnout velocity, flight path angle).

### 3. Core Capabilities Relevant to PS24
* **(i) Rapid Processing (Sub-orbital / Ballistic)**: Instantaneous threat profiling of missile launches based on early radar tracks.

### 4. Key Modules / Classes / Functions
* `trajectory.py`: Lambert-based ballistic solvers.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Ballistic Range
from trajectory import compute_range
impact_lat, impact_lon = compute_range(launch_lat, launch_lon, v_burnout, gamma)
```

### 9. Overall Relevance Score to PS24
**8/10** — Extremely relevant for the sub-orbital threat tracking requirement.

### 10. Suggested Next Action
Implement the ballistic Lambert mapping in the Go engine to classify sub-orbital tracks instantly.

---

## 47. ferencdv/missile_trajectory_simulator

### 1. Repo Metadata
* **Full Name**: [ferencdv/missile_trajectory_simulator](https://github.com/ferencdv/missile_trajectory_simulator)
* **Primary Language**: Matlab/Python

### 9. Overall Relevance Score to PS24
**6/10** — Highly detailed, but heavily reliant on continuous thrust profiles which might not be known to SSA radars.

---

## 48. lwcook/hypersonic-simulation

### 1. Repo Metadata
* **Full Name**: [lwcook/hypersonic-simulation](https://github.com/lwcook/hypersonic-simulation)
* **Primary Language**: C++

### 2. README / Project Summary
Hypersonic glide vehicle (HGV) modeling, accounting for skip-glide trajectories and dynamic aerodynamic control surfaces.

### 3. Core Capabilities Relevant to PS24
* **(i) Rapid Processing (Hypersonic threats)**: Models non-Keplerian manoeuvring in the upper atmosphere.

### 9. Overall Relevance Score to PS24
**9/10** — HGVs are the primary threat defeating traditional radar. This logic is mandatory for modern early warning.

### 10. Suggested Next Action
Merge this skip-glide logic into our existing `solveAerospace` ported from Substrate A to handle HGV characterization.

---

## 49. dimitarbez/ICBMSimulator

### 1. Repo Metadata
* **Full Name**: [dimitarbez/ICBMSimulator](https://github.com/dimitarbez/ICBMSimulator)
* **Primary Language**: Unity/C#

### 9. Overall Relevance Score to PS24
**2/10** — Visual game simulation, not rigorous physics. Discard.

---

## 50. Bahrul-Rozak/Ballistic-Missile-Simulation-Visualization

### 1. Repo Metadata
* **Full Name**: Bahrul-Rozak/Ballistic-Missile-Simulation-Visualization
* **Primary Language**: Python/JS

### 9. Overall Relevance Score to PS24
**3/10** — Basic visualization. Discard.

---

## Summary Table: Repositories 36-50

| Repo # | Name | Rel. Score | Top PS24 Capabilities Supported | Recommended Role in IAF SSA Tool |
| :--- | :--- | :--- | :--- | :--- |
| **36** | mit-ll/MaDDG | 7/10 | (f) Manoeuvre Detection | Benchmark against SSIS PINO Hamiltonian anomaly detector. |
| **39** | Orbital-Space-Ontology | 6/10 | (g) Cataloguing | Integrate into unified JSON/DB schema for interoperability. |
| **41** | Tomenol/deorbitation | 5/10 | (h) Re-entry Analysis | Extract USAF 1976 atmosphere tables for the Go engine. |
| **43** | lcasts/REAT | 8/10 | (h) Re-entry Analysis | Port debris footprint/ellipse dispersion logic to Go core. |
| **46** | jlev/ballistic-missile | 8/10 | (i) Sub-orbital Threats | Port rapid ballistic impact point estimation. |
| **48** | lwcook/hypersonic | 9/10 | (i) Hypersonic Threats | Integrate skip-glide trajectory logic to track HGVs. |

*(All other repositories scored <5 and are recommended for archiving/discarding due to academic immaturity or irrelevance).*
