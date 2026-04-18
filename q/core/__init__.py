"""Core physics simulation modules for Quantum Immune Digital Twin."""

from .bdg_hamiltonian import (
    BdGHamiltonian,
    CosmicRayParameters,
    DigitalTwinEngine,
    Geant4ToLindbladCalibration,
    IronBasedSuperconductorPlatform,
    KitaevChainParameters,
    NanowireGeometry,
    QPBurstEvent,
    QPBurstSimulator,
    ShadowTomographyOverhead,
)

from .active_evasion import (
    ActiveEvasionEngine,
    DetectionEvent,
    DiffusionPrediction,
    EvasionPlan,
    FastDiffusionSolver,
    BraidingPathPlanner,
)

# Generator Layer (Evasion Engine)
from .generator_layer import (
    HamiltonianGraphNode,
    ReroutingPath,
    UnitaryManifoldRerouting,
    CalorimeterEvent,
    SaturnRelay,
    EvasionController,
)

# Advanced Evasion (2026 Stack)
from .evasion_advanced import (
    WeightedHamiltonianNode,
    BraidingTrajectory,
    DynamicDijkstraPathfinder,
    NonAdiabaticPassage,
    AdaptiveUnitaryRerouting,
)

# Sensing Layer (2026 Stack)
from .sensing_layer import (
    CalorimeterPulse,
    StrikePrediction,
    EdgeTriggeredAnomalyDetector,
    SubNanosecondTrigger,
    CalorimeterArray,
)

# TDBdG Solver (2026 Stack)
from .tdbdg_solver import (
    SSFMConfig,
    SplitStepFourierSolver,
    HamiltonianNeuralNetwork,
    create_tdbdg_solver_for_nanowire,
)

# Auditor Layer (Physics Grounding)
from .auditor_layer import (
    PfaffianInvariant,
    RealTimePfaffianChecker,
    TernaryQuantizer,
    QPHNN,
    PhysicsAuditor,
)

# JAX 2026 Stack (optional, gracefully degrades)
try:
    from .jax_backend import (
        CSSTParameters,
        RDASolverConfig,
        CompressedSensingShadowTomography,
        RDASolver,
        FastHamiltonianSolver,
        SymplecticAuditor,
        HAS_JAX,
        HAS_DIFFRAX,
        HAS_LINEAX,
        HAS_DYNAMIQS,
    )
    JAX_BACKEND_AVAILABLE = True
except ImportError:
    JAX_BACKEND_AVAILABLE = False
    HAS_JAX = HAS_DIFFRAX = HAS_LINEAX = HAS_DYNAMIQS = False

__all__ = [
    "BdGHamiltonian",
    "CosmicRayParameters",
    "DigitalTwinEngine",
    "Geant4ToLindbladCalibration",
    "IronBasedSuperconductorPlatform",
    "KitaevChainParameters",
    "NanowireGeometry",
    "QPBurstEvent",
    "QPBurstSimulator",
    "ShadowTomographyOverhead",
    "ActiveEvasionEngine",
    "DetectionEvent",
    "DiffusionPrediction",
    "EvasionPlan",
    "FastDiffusionSolver",
    "BraidingPathPlanner",
    # Generator Layer
    "HamiltonianGraphNode",
    "ReroutingPath",
    "UnitaryManifoldRerouting",
    "CalorimeterEvent",
    "SaturnRelay",
    "EvasionController",
    # Advanced Evasion (2026 Stack)
    "WeightedHamiltonianNode",
    "BraidingTrajectory",
    "DynamicDijkstraPathfinder",
    "NonAdiabaticPassage",
    "AdaptiveUnitaryRerouting",
    # Sensing Layer (2026 Stack)
    "CalorimeterPulse",
    "StrikePrediction",
    "EdgeTriggeredAnomalyDetector",
    "SubNanosecondTrigger",
    "CalorimeterArray",
    # TDBdG Solver (2026 Stack)
    "SSFMConfig",
    "SplitStepFourierSolver",
    "HamiltonianNeuralNetwork",
    "create_tdbdg_solver_for_nanowire",
    # Auditor Layer
    "PfaffianInvariant",
    "RealTimePfaffianChecker",
    "TernaryQuantizer",
    "QPHNN",
    "PhysicsAuditor",
]

# Conditionally export JAX backend
if JAX_BACKEND_AVAILABLE:
    __all__.extend([
        "CSSTParameters",
        "RDASolverConfig",
        "CompressedSensingShadowTomography",
        "RDASolver",
        "FastHamiltonianSolver",
        "SymplecticAuditor",
        "HAS_JAX",
        "HAS_DIFFRAX",
        "HAS_LINEAX",
        "HAS_DYNAMIQS",
    ])
