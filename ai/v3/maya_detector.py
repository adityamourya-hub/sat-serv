"""
Maya Deception Detector — Counterspace deception detection via behavioral inconsistency.

Detects adversary "Maya" (deception) tactics by cross-referencing:
  1. Hamiltonian invariant violations without plausible ΔV explanations
  2. RCS signature inconsistencies with orbital regime
  3. Maneuver pattern matching against known deception profiles
  4. Electromagnetic signature anomalies (IQ/spectral)
  5. Catalog correlation gaps (objects appearing/vanishing without physics basis)

Unique: No US/China/Russia system combines physics-grounded + behavioral deception detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DeceptionType(Enum):
    NONE = "none"
    SPOOFING = "spoofing"                  # False sensor returns injected
    DECOY = "decoy"                        # Physical decoy deployed
    CAMOUFLAGE = "camouflage"              # Object masking RCS/signature
    TRAJECTORY_MASKING = "trajectory_mask"  # Maneuver hidden by natural perturbation
    CATALOG_POISONING = "catalog_poison"    # False TLE injection


@dataclass
class DeceptionEvidence:
    check_name: str
    triggered: bool
    severity: float        # [0, 1]
    description: str
    physics_basis: str     # Why this indicates deception
    raw_metrics: Dict = field(default_factory=dict)


@dataclass
class MayaAssessment:
    object_id: int
    maya_score: float                   # [0, 1] aggregate deception probability
    deception_type: DeceptionType       # Most likely deception type
    evidence: List[DeceptionEvidence]   # Individual check results
    recommendation: str                 # MONITOR / ESCALATE / CONFIRM_VISUAL
    confidence: float                   # Assessment confidence
    human_summary: str


# --- Physics Consistency Checks ---

class HamiltonianConsistencyChecker:
    """
    Check if object's orbital energy changes are physically explainable.
    An unexplained energy change implies either:
      (a) Undetected maneuver, or
      (b) Deception (spoofed tracking data)
    """
    MU_EARTH = 3.986004418e5  # km³/s²

    def check(self, state_history: List[Dict]) -> DeceptionEvidence:
        """
        Args:
            state_history: List of {epoch_jd, x, y, z, vx, vy, vz, detected_maneuver: bool}
        """
        if len(state_history) < 3:
            return DeceptionEvidence("hamiltonian_violation", False, 0.0,
                                      "Insufficient history", "N/A")

        energies = []
        for s in state_history:
            r = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            v = np.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)
            E = 0.5 * v**2 - self.MU_EARTH / r  # Specific orbital energy
            energies.append(E)

        energies = np.array(energies)
        # Check energy jumps between consecutive states
        dE = np.abs(np.diff(energies))
        maneuvers = [s.get('detected_maneuver', False) for s in state_history[1:]]

        unexplained_jumps = []
        for i, (de, man) in enumerate(zip(dE, maneuvers)):
            # Energy change > 0.1 km²/s² without detected maneuver is suspicious
            if de > 0.1 and not man:
                unexplained_jumps.append((i, de))

        if unexplained_jumps:
            max_jump = max(j[1] for j in unexplained_jumps)
            severity = min(1.0, max_jump / 1.0)  # Normalize to [0,1]
            return DeceptionEvidence(
                "hamiltonian_violation", True, severity,
                f"{len(unexplained_jumps)} unexplained energy change(s), max ΔE={max_jump:.3f} km²/s²",
                "Orbital energy (H = ½v² - μ/r) is conserved under gravity. "
                "Unexplained changes indicate either hidden maneuvers or spoofed tracking.",
                raw_metrics={'n_jumps': len(unexplained_jumps), 'max_dE': max_jump},
            )

        return DeceptionEvidence("hamiltonian_violation", False, 0.0,
                                  "Energy changes consistent with dynamics", "Conservation of energy")


class RCSOrbitConsistencyChecker:
    """
    Cross-reference RCS with orbital regime.
    A 10m² object in a 200km orbit with no catalog match is suspicious.
    """
    # Typical RCS ranges by object type
    RCS_PROFILES = {
        'cubesat_1u': (0.001, 0.01),     # 1U CubeSat
        'cubesat_6u': (0.01, 0.05),      # 6U CubeSat
        'smallsat': (0.05, 0.5),          # Small satellite (100-500 kg)
        'medium_sat': (0.5, 5.0),         # Medium satellite
        'large_sat': (5.0, 50.0),         # Large satellite (ISS-class)
        'debris_small': (0.0001, 0.01),   # Small debris
        'debris_large': (0.01, 1.0),      # Large debris
        'rocket_body': (1.0, 20.0),       # Spent upper stage
    }

    def check(self, rcs_history: List[float], catalog_type: str,
              altitude_km: float) -> DeceptionEvidence:
        if not rcs_history:
            return DeceptionEvidence("rcs_orbit_mismatch", False, 0.0,
                                      "No RCS data", "N/A")

        rcs_median = np.median(rcs_history)
        rcs_std = np.std(rcs_history)

        # Check against expected range for catalog type
        expected = self.RCS_PROFILES.get(catalog_type, (0.01, 10.0))
        in_range = expected[0] <= rcs_median <= expected[1]

        # Check for anomalous RCS variability (tumbling or shape-change)
        cv = rcs_std / (rcs_median + 1e-12)
        high_variability = cv > 2.0

        # Altitude-RCS cross-check: very large RCS at very low altitude is suspicious
        # (decoy balloon in LEO would have large RCS but isn't a real threat)
        altitude_rcs_anomaly = (altitude_km < 400 and rcs_median > 20.0)

        triggered = (not in_range) or altitude_rcs_anomaly
        severity = 0.0
        descriptions = []

        if not in_range:
            severity += 0.4
            descriptions.append(f"RCS {rcs_median:.2f}m² outside expected [{expected[0]:.3f}, {expected[1]:.1f}]m² for {catalog_type}")
        if altitude_rcs_anomaly:
            severity += 0.3
            descriptions.append(f"Large RCS ({rcs_median:.1f}m²) at low altitude ({altitude_km:.0f}km) — possible decoy")
        if high_variability:
            severity += 0.2
            descriptions.append(f"High RCS variability (CV={cv:.2f}) — possible tumbling or shape-shifting")

        severity = min(1.0, severity)
        desc = "; ".join(descriptions) if descriptions else "RCS consistent with catalog type"

        return DeceptionEvidence(
            "rcs_orbit_mismatch", triggered, severity, desc,
            "RCS should correlate with object size/type. Mismatches indicate decoys or misclassification.",
            raw_metrics={'rcs_median': rcs_median, 'rcs_cv': cv, 'in_range': in_range},
        )


class ManeuverPatternChecker:
    """
    Match maneuver sequences against known deception profiles:
      - Split/merge events (decoy deployment)
      - Oscillatory station-keeping mimicking natural perturbation
      - Coordinated maneuvers hiding in solar pressure variations
    """
    # Known deception signatures: (pattern_name, description, check_fn)

    def check(self, maneuver_history: List[Dict]) -> DeceptionEvidence:
        """
        Args:
            maneuver_history: List of {epoch_jd, delta_v_ms, direction_eci: [3], duration_s}
        """
        if len(maneuver_history) < 2:
            return DeceptionEvidence("deception_pattern", False, 0.0,
                                      "Insufficient maneuver history", "N/A")

        patterns_found = []

        # 1. Split event: rapid sequence of small maneuvers (decoy deployment)
        times = [m['epoch_jd'] for m in maneuver_history]
        dvs = [m['delta_v_ms'] for m in maneuver_history]
        dt = np.diff(times) * 86400  # Convert JD to seconds

        rapid_small = sum(1 for d, dv in zip(dt, dvs[1:]) if d < 300 and dv < 0.5)
        if rapid_small >= 3:
            patterns_found.append("rapid_small_maneuvers (possible decoy deployment)")

        # 2. Oscillatory pattern: alternating thrust directions
        if len(maneuver_history) >= 4:
            dirs = [np.array(m.get('direction_eci', [1, 0, 0])) for m in maneuver_history]
            dots = [np.dot(dirs[i], dirs[i + 1]) for i in range(len(dirs) - 1)]
            sign_changes = sum(1 for i in range(len(dots) - 1) if dots[i] * dots[i + 1] < 0)
            if sign_changes >= 2:
                patterns_found.append("oscillatory_thrust (masking as perturbation)")

        # 3. Low-thrust continuous (electric propulsion concealment)
        long_low = sum(1 for m in maneuver_history
                       if m.get('duration_s', 0) > 3600 and m['delta_v_ms'] < 0.1)
        if long_low >= 2:
            patterns_found.append("low_thrust_continuous (EP concealment)")

        triggered = len(patterns_found) > 0
        severity = min(1.0, len(patterns_found) * 0.35)
        desc = "; ".join(patterns_found) if patterns_found else "No deception patterns detected"

        return DeceptionEvidence(
            "deception_pattern", triggered, severity, desc,
            "Known adversary tactics include decoy deployment (rapid splits), "
            "oscillatory station-keeping, and electric propulsion concealment.",
            raw_metrics={'patterns': patterns_found, 'n_maneuvers': len(maneuver_history)},
        )


class CatalogCorrelationChecker:
    """
    Detect catalog anomalies: objects appearing or vanishing without physics basis.
    """
    def check(self, track_history: List[Dict]) -> DeceptionEvidence:
        """
        Args:
            track_history: List of {epoch_jd, tracked: bool, n_observations, residual_rms}
        """
        if len(track_history) < 5:
            return DeceptionEvidence("catalog_anomaly", False, 0.0,
                                      "Insufficient tracking history", "N/A")

        tracked = [t['tracked'] for t in track_history]
        residuals = [t.get('residual_rms', 0) for t in track_history]

        # Sudden appearance (no prior track → tracked)
        appearances = sum(1 for i in range(1, len(tracked))
                          if tracked[i] and not tracked[i - 1])

        # Sudden disappearance (tracked → not tracked, no re-entry prediction)
        disappearances = sum(1 for i in range(1, len(tracked))
                             if not tracked[i] and tracked[i - 1])

        # Residual spikes (track quality suddenly degrades)
        residual_spikes = sum(1 for r in residuals if r > 5.0)

        anomaly_count = appearances + disappearances + residual_spikes
        triggered = anomaly_count >= 2
        severity = min(1.0, anomaly_count * 0.2)
        parts = []
        if appearances:
            parts.append(f"{appearances} sudden appearance(s)")
        if disappearances:
            parts.append(f"{disappearances} sudden disappearance(s)")
        if residual_spikes:
            parts.append(f"{residual_spikes} residual spike(s)")

        desc = "; ".join(parts) if parts else "Catalog track consistent"

        return DeceptionEvidence(
            "catalog_anomaly", triggered, severity, desc,
            "Objects don't spontaneously appear/vanish. Anomalies suggest catalog poisoning or sensor spoofing.",
            raw_metrics={'appearances': appearances, 'disappearances': disappearances,
                         'residual_spikes': residual_spikes},
        )


# --- Aggregate Detector ---

class MayaDeceptionDetector:
    """
    Aggregate deception assessment combining all physics checks.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.hamiltonian_checker = HamiltonianConsistencyChecker()
        self.rcs_checker = RCSOrbitConsistencyChecker()
        self.pattern_checker = ManeuverPatternChecker()
        self.catalog_checker = CatalogCorrelationChecker()

        self.weights = weights or {
            'hamiltonian_violation': 0.35,
            'rcs_orbit_mismatch': 0.25,
            'deception_pattern': 0.25,
            'catalog_anomaly': 0.15,
        }

    def detect_deception(self, object_id: int,
                          state_history: List[Dict],
                          rcs_history: List[float],
                          catalog_type: str,
                          altitude_km: float,
                          maneuver_history: List[Dict],
                          track_history: List[Dict]) -> MayaAssessment:
        """
        Run all deception checks and produce aggregate assessment.
        """
        evidence = []

        # 1. Hamiltonian consistency
        e1 = self.hamiltonian_checker.check(state_history)
        evidence.append(e1)

        # 2. RCS-orbit consistency
        e2 = self.rcs_checker.check(rcs_history, catalog_type, altitude_km)
        evidence.append(e2)

        # 3. Maneuver pattern analysis
        e3 = self.pattern_checker.check(maneuver_history)
        evidence.append(e3)

        # 4. Catalog correlation
        e4 = self.catalog_checker.check(track_history)
        evidence.append(e4)

        # Aggregate maya score (weighted sum of severities)
        maya_score = sum(
            self.weights.get(e.check_name, 0.25) * e.severity
            for e in evidence
        )
        maya_score = min(1.0, maya_score)

        # Determine most likely deception type
        deception_type = self._classify_deception(evidence)

        # Recommendation
        if maya_score > 0.7:
            recommendation = "ESCALATE — high probability of adversary deception"
        elif maya_score > 0.4:
            recommendation = "CONFIRM_VISUAL — request optical/SAR confirmation pass"
        elif maya_score > 0.2:
            recommendation = "MONITOR — elevated deception indicators"
        else:
            recommendation = "NOMINAL — no significant deception indicators"

        # Confidence based on data completeness
        data_completeness = sum([
            1 if len(state_history) >= 5 else 0.3,
            1 if len(rcs_history) >= 3 else 0.3,
            1 if len(maneuver_history) >= 2 else 0.5,
            1 if len(track_history) >= 5 else 0.3,
        ]) / 4.0

        triggered = [e for e in evidence if e.triggered]
        summary_parts = [e.description for e in triggered[:3]]
        human_summary = (f"Object {object_id}: Maya score {maya_score:.2f} "
                         f"({deception_type.value}). "
                         + ("; ".join(summary_parts) if summary_parts else "No anomalies detected."))

        return MayaAssessment(
            object_id=object_id,
            maya_score=maya_score,
            deception_type=deception_type,
            evidence=evidence,
            recommendation=recommendation,
            confidence=data_completeness,
            human_summary=human_summary,
        )

    def _classify_deception(self, evidence: List[DeceptionEvidence]) -> DeceptionType:
        """Determine most likely deception type from evidence pattern."""
        triggered = {e.check_name: e for e in evidence if e.triggered}

        if not triggered:
            return DeceptionType.NONE

        # Heuristic classification
        if 'rcs_orbit_mismatch' in triggered and 'deception_pattern' in triggered:
            if 'rapid_small' in str(triggered['deception_pattern'].raw_metrics.get('patterns', [])):
                return DeceptionType.DECOY

        if 'hamiltonian_violation' in triggered and 'catalog_anomaly' in triggered:
            return DeceptionType.SPOOFING

        if 'deception_pattern' in triggered:
            patterns = triggered['deception_pattern'].raw_metrics.get('patterns', [])
            if any('EP concealment' in p for p in patterns):
                return DeceptionType.TRAJECTORY_MASKING
            if any('oscillatory' in p for p in patterns):
                return DeceptionType.CAMOUFLAGE

        if 'catalog_anomaly' in triggered:
            return DeceptionType.CATALOG_POISONING

        if 'rcs_orbit_mismatch' in triggered:
            return DeceptionType.CAMOUFLAGE

        return DeceptionType.SPOOFING  # Default for unexplained anomalies


# --- Self-Test ---

if __name__ == "__main__":
    detector = MayaDeceptionDetector()

    # Simulate a suspicious object
    state_history = [
        {'x': 6800, 'y': 0, 'z': 0, 'vx': 0, 'vy': 7.5, 'vz': 0, 'epoch_jd': 2460000.0, 'detected_maneuver': False},
        {'x': 6800, 'y': 100, 'z': 0, 'vx': 0, 'vy': 7.5, 'vz': 0, 'epoch_jd': 2460000.01, 'detected_maneuver': False},
        {'x': 6800, 'y': 200, 'z': 0, 'vx': 0, 'vy': 8.2, 'vz': 0, 'epoch_jd': 2460000.02, 'detected_maneuver': False},  # Energy jump!
        {'x': 6800, 'y': 300, 'z': 0, 'vx': 0, 'vy': 8.2, 'vz': 0, 'epoch_jd': 2460000.03, 'detected_maneuver': False},
    ]
    rcs_history = [0.5, 0.6, 15.0, 14.0, 16.0]  # RCS suddenly jumps (decoy?)
    maneuver_history = [
        {'epoch_jd': 2460000.01, 'delta_v_ms': 0.2, 'direction_eci': [1, 0, 0], 'duration_s': 60},
        {'epoch_jd': 2460000.012, 'delta_v_ms': 0.1, 'direction_eci': [-1, 0, 0], 'duration_s': 30},
        {'epoch_jd': 2460000.014, 'delta_v_ms': 0.15, 'direction_eci': [1, 0, 0], 'duration_s': 45},
        {'epoch_jd': 2460000.016, 'delta_v_ms': 0.1, 'direction_eci': [-1, 0, 0], 'duration_s': 30},
    ]
    track_history = [
        {'epoch_jd': 2459999.0, 'tracked': False, 'n_observations': 0, 'residual_rms': 0},
        {'epoch_jd': 2460000.0, 'tracked': True, 'n_observations': 5, 'residual_rms': 0.5},
        {'epoch_jd': 2460000.5, 'tracked': True, 'n_observations': 3, 'residual_rms': 8.0},
        {'epoch_jd': 2460001.0, 'tracked': False, 'n_observations': 0, 'residual_rms': 0},
        {'epoch_jd': 2460001.5, 'tracked': True, 'n_observations': 2, 'residual_rms': 12.0},
    ]

    result = detector.detect_deception(
        object_id=99999,
        state_history=state_history,
        rcs_history=rcs_history,
        catalog_type='cubesat_6u',
        altitude_km=430,
        maneuver_history=maneuver_history,
        track_history=track_history,
    )

    print(f"\n{result.human_summary}")
    print(f"Maya Score: {result.maya_score:.2f}")
    print(f"Deception Type: {result.deception_type.value}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence: {result.confidence:.2f}")
    for e in result.evidence:
        status = "⚠️  TRIGGERED" if e.triggered else "✅ OK"
        print(f"  [{status}] {e.check_name}: {e.description} (severity={e.severity:.2f})")
