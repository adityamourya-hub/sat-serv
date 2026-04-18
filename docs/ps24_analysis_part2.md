# PS24 Astrodynamics Sovereign Agent Deep Analysis (Part 2)

---

## 8. esa/pykep

### 1. Repo Metadata
* **Full Name**: [esa/pykep](https://github.com/esa/pykep)
* **Stars**: 403
* **Last Commit Date**: 2026-03-30
* **Primary Language**: C++ (Python bindings)
* **License**: Mozilla Public License 2.0

### 2. README / Project Summary
PyKep is a scientific library developed by the European Space Agency for basic astrodynamics and space mission trajectory optimization. It provides powerful algorithms for Lambert problems and multi-gravity assist planning.

### 3. Core Capabilities Relevant to PS24
* **(c) Conjunction Analysis**: Robust Lambert solvers enable predicting potential interception or collision paths.
* **(i) Rapid Processing**: Core algorithm speed implemented in C++.

### 4. Key Modules / Classes / Functions
* `pykep.lambert_problem`: Solves the 2-point boundary value problem.
* `pykep.planet`: Defines ephemeris objects.
* `pykep.sims_flanagan`: Trajectory optimization method.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Pykep Lambert problem (Interception math)
import pykep as pk
l = pk.lambert_problem(r1, r2, flight_time, mu)
v1, v2 = l.get_v1()[0], l.get_v2()[0]
```

### 6. Installation & Integration Notes
* **Installation**: Conda or Pip.
* **Integration**: Standard Python C-extension integration.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 5/10 (ESA maintained, implies deep European dependency).
* **Air-gapped Suitability**: Yes.
* **Indian-friendly**: MPL 2.0 is commercial/defense friendly but requires releasing modified source.

### 8. Limitations & Risks
* Focused mostly on interplanetary and deep space optimization rather than Earth LEO/MEO SSA tasks.

### 9. Overall Relevance Score to PS24
**4/10** — Excellent for ISRO interplanetary probes, irrelevant for LEO/Air Force surveillance.

### 10. Suggested Next Action
Ignore for PS24; focus instead on SGP4-specific implementations.

---

## 9. duncaneddy/brahe

### 1. Repo Metadata
* **Full Name**: [duncaneddy/brahe](https://github.com/duncaneddy/brahe)
* **Stars**: 70
* **Last Commit Date**: 2026-04-10
* **Primary Language**: Rust (Python bindings via pyO3)
* **License**: MIT License

### 2. README / Project Summary
Brahe is a modern, practical, and highly performant astrodynamics library written in Rust. It emphasizes modern software engineering (zero-cost abstractions, thread safety) while providing SGP4, attitude dynamics, and Earth observation modeling.

### 3. Core Capabilities Relevant to PS24
* **(b) Neighbourhood Watch**: Sensor footprint calculation modules.
* **(c) Conjunction Analysis**: Excellent orbital state propagation.
* **(i) Rapid Processing**: Rust memory-safe speed for processing 50K+ TLEs.

### 4. Key Modules / Classes / Functions
* `SGP4`: Orbit propagator for TLEs.
* `Spacecraft`: State tracking system.
* `Frames`: ECI/ECEF transformation logic.
* `Access`: Visibility logic.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Brahe SGP4 propagation
import brahe
tle = brahe.TLE("1 ...", "2 ...")
state = brahe.sgp4_propagate(tle, ephemeris_time)
```

### 6. Installation & Integration Notes
* **Installation**: Python pip wheels available, or native Cargo build.
* **Integration**: Outstanding integration profile due to Rust. Perfect for wrapping in Go/C.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 8/10 (Solo open-source dev, MIT license).
* **Air-gapped Suitability**: Yes, native compilation.
* **Indian-friendly**: Very high (MIT).

### 8. Limitations & Risks
* Lacks the deep mathematical verification layer of something like Orekit; maintained by a single developer.

### 9. Overall Relevance Score to PS24
**7/10** — A solid technical baseline, but we are replicating its core features with HMAC security in our native SSIS Go engine.

### 10. Suggested Next Action
Use as a benchmark for Rust performance against our Go/Python SSIS substrate.

---

## 10. lox-space/lox

### 1. Repo Metadata
* **Full Name**: [lox-space/lox](https://github.com/lox-space/lox)
* **Stars**: 369
* **Last Commit Date**: 2026-04-03
* **Primary Language**: Rust
* **License**: Mozilla Public License 2.0

### 2. README / Project Summary
Lox is an oxidized astrodynamics library intended to be a foundational infrastructure for space applications. It is heavily focused on safety, speed, and modern Rust paradigms.

### 3. Core Capabilities Relevant to PS24
* **(c) Conjunction Analysis**: Trajectory computation.
* **(i) Rapid Processing**: Highly concurrent state propagation.

### 4. Key Modules / Classes / Functions
* `lox::orbit`: Keplerian parameter conversion.
* `lox::propagator`: Integrators.

### 5. Ready-to-Use Example Code Snippets
```rust
// Rust: Lox state conversion
let orbit = KeplerianOrbit::new(a, e, i, raan, aop, ta);
let state = orbit.to_cartesian();
```

### 6. Installation & Integration Notes
* **Installation**: Cargo build.
* **Integration**: Native integration.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 7/10.
* **Air-gapped Suitability**: Yes.

### 8. Limitations & Risks
* Still in early development compared to Nyx or Orekit.

### 9. Overall Relevance Score to PS24
**5/10** — Redundant if we evaluate Nyx or Brahe.

### 10. Suggested Next Action
Skip.

---

## 11. ATTron/astroz

### 1. Repo Metadata
* **Full Name**: [ATTron/astroz](https://github.com/ATTron/astroz)
* **Stars**: 249
* **Last Commit Date**: 2026-03-31
* **Primary Language**: Zig
* **License**: GNU General Public License v3.0

### 2. README / Project Summary
Astroz is an incredibly fast astrodynamics and SGP4 library written in Zig. It utilizes Zig's comptime features and strict memory controls to outpace C/C++ implementations in batch propagation.

### 3. Core Capabilities Relevant to PS24
* **(c) Conjunction Analysis**: Validated SGP4/SDP4 core.
* **(i) Rapid Processing**: Absolute maximum throughput for volumetric LEO scans (capable of processing the entire catalog in milliseconds).

### 4. Key Modules / Classes / Functions
* `sgp4.zig`: TLE propagation logic.
* `frames.zig`: Reference frame transforms.

### 5. Ready-to-Use Example Code Snippets
```zig
// Zig: Astroz propagation
const sat = try sgp4.init(tle_line1, tle_line2, allocator);
const state = sat.propagate(minutes_since_epoch);
```

### 6. Installation & Integration Notes
* **Installation**: Zig build system.
* **Integration**: Compiles to a tiny static binary or shared object, easily called via C-ABI from our Go engine.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 8/10 (GPLv3).
* **Air-gapped Suitability**: Yes.
* **Indian-friendly**: Very high, but GPLv3 requires open-sourcing derived works.

### 8. Limitations & Risks
* GPLv3 license is a showstopper for classified IAF closed-source systems.
* Zig ecosystem is young.

### 9. Overall Relevance Score to PS24
**6/10** — Incredible performance, but killed by GPLv3 licensing strictures for defense.

### 10. Suggested Next Action
Analyze the Zig implementation of SGP4 for optimization ideas, but do not integrate directly.

---

## 12. IO-Aerospace-software-engineering/Astrodynamics

### 1. Repo Metadata
* **Full Name**: Astrodynamics (IO-Aerospace)
* **Stars**: 35
* **Last Commit Date**: 2026-04-08
* **Primary Language**: C# (.NET)
* **License**: LGPL v2.1

### 2. README / Project Summary
A .NET C# astrodynamics library backed by the JPL SPICE toolkit, targeting the Microsoft engineering ecosystem for orbit mechanics and trajectory visualization.

### 3. Core Capabilities Relevant to PS24
* **(g) Characterisation**: Integration with SPICE kernels for massive ephemeris catalogs.

### 4. Key Modules / Classes / Functions
* `OrbitPropagator`: Standard propagation.
* `SpiceWrapper`: C# bindings for NAIF SPICE.

### 5. Ready-to-Use Example Code Snippets
```csharp
// C#: SPICE based state retrieval
SpiceWrapper.Furnish("de430.bsp");
double[] state = SpiceWrapper.GetState("SATELLITE", et, "J2000", "EARTH");
```

### 6. Installation & Integration Notes
* **Installation**: NuGet package.
* **Integration**: Heavily tied to Windows/.NET ecosystem.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 2/10 (Relies entirely on Microsoft runtime and JPL/NASA SPICE binaries).
* **Air-gapped Suitability**: Poor (SPICE kernels must be manually updated).
* **Indian-friendly**: No.

### 8. Limitations & Risks
* C# / .NET requirement limits Linux-based air-gapped containerization strategies preferred by defense.

### 9. Overall Relevance Score to PS24
**2/10** — Incompatible with the Go/Python strategy and sovereign independence goals.

### 10. Suggested Next Action
Discard.

---

## 13. AlabamaASRL/asset_asrl

### 1. Repo Metadata
* **Full Name**: [AlabamaASRL/asset_asrl](https://github.com/AlabamaASRL/asset_asrl)
* **Stars**: 39
* **Last Commit Date**: 2026-04-07
* **Primary Language**: C++ (Python bindings)
* **License**: Apache License 2.0

### 2. README / Project Summary
ASSET (Astrodynamics Software and Systems Engineering Toolkit) is a library for massive parallel trajectory optimization and design, developed by the University of Alabama.

### 3. Core Capabilities Relevant to PS24
* **(f) Manoeuvre Detection**: Optimal control formulation can be inverted to determine if a thrust profile matches an anomaly.

### 4. Key Modules / Classes / Functions
* `OptimalControlProblem`: Core NLP solver interface.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: ASSET optimal control setup
import asset_asrl as ast
ocp = ast.OptimalControlProblem()
```

### 6. Installation & Integration Notes
* **Installation**: Conda forge.
* **Integration**: C++ backend requires Intel MKL / OpenBLAS.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 4/10.

### 8. Limitations & Risks
* Requires heavy compute libraries (BLAS/LAPACK) which bloat the deployment package.

### 9. Overall Relevance Score to PS24
**3/10** — Too focused on optimizing maneuvers rather than detecting them via SSA.

### 10. Suggested Next Action
Discard.

---

## 14. boinor/boinor

### 1. Repo Metadata
* **Full Name**: [boinor/boinor](https://github.com/boinor/boinor)
* **Stars**: 21
* **Primary Language**: Python
* **License**: MIT License

### 2. README / Project Summary
A pure-Python, interactive astrodynamics scripting toolkit aimed at education and rapid prototyping.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 6/10.

### 9. Overall Relevance Score to PS24
**1/10** — Pure Python is far too slow for catalog-wide (50,000+ object) conjunction analysis.

### 10. Suggested Next Action
Discard.

---

## 15. strath-ace/smart-astro

### 1. Repo Metadata
* **Full Name**: [strath-ace/smart-astro](https://github.com/strath-ace/smart-astro)
* **Stars**: 5
* **Primary Language**: C++
* **License**: Other

### 9. Overall Relevance Score to PS24
**1/10** — Underdeveloped academic repository.

### 10. Suggested Next Action
Discard.

---

## 16. brandon-rhodes/python-sgp4

### 1. Repo Metadata
* **Full Name**: [brandon-rhodes/python-sgp4](https://github.com/brandon-rhodes/python-sgp4)
* **Stars**: (High - estimated 500+)
* **Primary Language**: Python (C++ extension)
* **License**: MIT License

### 2. README / Project Summary
The gold-standard Python implementation of the SGP4 satellite propagation algorithm, directly matching the mathematical models from standard defense publications.

### 3. Core Capabilities Relevant to PS24
* **(c) Conjunction Analysis**: Exact NORAD SGP4 calculation.
* **(i) Rapid Processing**: The underlying C++ extension is highly optimized.

### 4. Key Modules / Classes / Functions
* `sgp4.api.Satrec`: High-performance satellite record structure.
* `sgp4.api.jday`: Julian day time conversion.

### 5. Ready-to-Use Example Code Snippets
```python
# Python: Fast SGP4
from sgp4.api import Satrec, WGS84
satellite = Satrec.twoline2rv(line1, line2)
jd, fr = 2458000.0, 0.5
e, r, v = satellite.sgp4(jd, fr)
```

### 6. Installation & Integration Notes
* **Installation**: `pip install sgp4`
* **Integration**: Essential wrapper around rigorous C++.

### 7. Indigenous / IAF Suitability Assessment
* **Sovereignty Score**: 7/10 (US defense math standard, openly available).
* **Air-gapped Suitability**: Excellent.

### 8. Limitations & Risks
* Does not include higher fidelity drag or non-spherical Earth models beyond what is in TLEs.

### 9. Overall Relevance Score to PS24
**9/10** — This is the absolute benchmark that our internal Go propagator must mathematically match perfectly.

### 10. Suggested Next Action
Use as the absolute unassailable ground truth for automated parity unit tests against our Go SSIS SGP4 kernel.

*(Continuing...)*
