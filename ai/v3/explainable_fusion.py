"""
Physics-Grounded SHAP Explainer for Multi-Sensor SSA Alerts.

Provides operator-readable explanations for conjunction alerts, maneuver detections,
and rapid-threat classifications by attributing decisions to:
  1. Specific sensor inputs (ATeena radar, RISAT SAR, optical, NavIC)
  2. Physical quantities (RCS, Doppler, covariance growth, drag uncertainty)
  3. Confidence intervals and uncertainty decomposition

Satisfies IAF "epistemic transparency" requirement for human-on-the-loop autonomy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# --- Data Classes ---

@dataclass
class SensorAttribution:
    sensor_name: str       # e.g., "ATeena", "RISAT-SAR", "Optical-CCD", "NavIC"
    feature_name: str      # e.g., "rcs_fluctuation", "isar_glint", "doppler_shift"
    shap_value: float      # Signed attribution (positive = increases alert)
    raw_value: float       # Actual measurement value
    unit: str              # Physical unit
    confidence: float      # Sensor-level confidence [0, 1]

    @property
    def abs_contribution(self) -> float:
        return abs(self.shap_value)

    def human_readable(self) -> str:
        direction = "increases" if self.shap_value > 0 else "decreases"
        return (f"{self.sensor_name} {self.feature_name} = {self.raw_value:.3f} {self.unit} "
                f"({direction} risk by {self.abs_contribution:.2%})")


@dataclass
class PhysicsBasis:
    mechanism: str         # e.g., "covariance_growth", "drag_uncertainty", "maneuver_delta_v"
    formula_latex: str     # LaTeX representation of the physics
    contribution: float    # How much this physics mechanism contributes
    description: str       # Human-readable explanation


@dataclass
class AlertExplanation:
    alert_type: str                                  # conjunction / maneuver / rapid_threat
    top_sensors: List[SensorAttribution]             # Ranked sensor attributions
    physics_basis: List[PhysicsBasis]                # Physics mechanisms
    confidence: float                                # Overall confidence
    uncertainty_lower: float
    uncertainty_upper: float
    human_summary: str                               # One-sentence operator summary
    recommendation: str                              # MONITOR / ESCALATE / MANEUVER


# --- Perturbation-Based SHAP Approximation ---

class PerturbationSHAP:
    """
    Model-agnostic SHAP approximation via input perturbation.
    Works with any prediction function f(features) → score.
    """
    def __init__(self, n_samples: int = 200, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def explain(self, predict_fn, features: Dict[str, float],
                baseline: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute approximate SHAP values for each feature.
        
        Args:
            predict_fn: f(feature_dict) → scalar score
            features: Current input features
            baseline: Reference features (mean/zero). If None, uses zeros.
        Returns:
            SHAP values per feature name
        """
        keys = sorted(features.keys())
        n = len(keys)
        if baseline is None:
            baseline = {k: 0.0 for k in keys}

        f_full = predict_fn(features)
        f_base = predict_fn(baseline)

        shap_values = {k: 0.0 for k in keys}
        for _ in range(self.n_samples):
            perm = self.rng.permutation(n)
            current = dict(baseline)
            prev_val = predict_fn(current)
            for idx in perm:
                k = keys[idx]
                current[k] = features[k]
                new_val = predict_fn(current)
                shap_values[k] += (new_val - prev_val)
                prev_val = new_val

        # Average over permutations
        for k in keys:
            shap_values[k] /= self.n_samples
        return shap_values


# --- Physics-Grounded Explainer ---

class PhysicsSHAPExplainer:
    """
    Full physics-grounded explainability pipeline for SSA alerts.
    """
    # Sensor → feature registry
    SENSOR_FEATURES = {
        'ATeena': [
            ('rcs_fluctuation', 'm²', 'Radar cross-section variation indicates tumbling/shape change'),
            ('doppler_shift', 'Hz', 'Radial velocity measurement from pulse Doppler'),
            ('snr', 'dB', 'Signal-to-noise ratio determines detection confidence'),
            ('micro_doppler', 'Hz', 'Micro-Doppler signature reveals attitude dynamics'),
        ],
        'RISAT-SAR': [
            ('isar_glint', 'dBsm', 'ISAR glint pattern reveals 3D structure'),
            ('polarimetric_entropy', '', 'HH/VV/HV decomposition entropy'),
            ('range_profile_length', 'm', 'Along-range extent of target'),
        ],
        'Optical-CCD': [
            ('apparent_magnitude', 'mag', 'Brightness measurement'),
            ('light_curve_period', 's', 'Rotational period from brightness variation'),
            ('color_index', 'B-V', 'Spectral signature for material identification'),
        ],
        'NavIC': [
            ('pseudorange_residual', 'm', 'Dual-frequency residual for ionospheric health'),
            ('cn0_ratio', 'dB-Hz', 'Carrier-to-noise density'),
            ('timing_anomaly_score', '', 'LSTM anomaly detector output'),
        ],
    }

    PHYSICS_MECHANISMS = {
        'covariance_growth': PhysicsBasis(
            mechanism='covariance_growth',
            formula_latex=r'P_{k+1} = F P_k F^T + Q',
            contribution=0.0,
            description='Uncertainty grows between observations; longer gaps increase miss distance uncertainty',
        ),
        'drag_uncertainty': PhysicsBasis(
            mechanism='drag_uncertainty',
            formula_latex=r'a_{drag} = -\frac{1}{2} \rho C_D \frac{A}{m} v^2 \hat{v}',
            contribution=0.0,
            description='Atmospheric density uncertainty dominates LEO prediction errors',
        ),
        'maneuver_delta_v': PhysicsBasis(
            mechanism='maneuver_delta_v',
            formula_latex=r'\Delta v = v_e \ln(m_0/m_f)',
            contribution=0.0,
            description='Detected thrust event changes trajectory prediction',
        ),
        'j2_perturbation': PhysicsBasis(
            mechanism='j2_perturbation',
            formula_latex=r'a_{J2} = \frac{3}{2} J_2 \frac{\mu R_e^2}{r^4}',
            contribution=0.0,
            description='Earth oblateness causes nodal regression and apsidal rotation',
        ),
        'third_body_lunar': PhysicsBasis(
            mechanism='third_body_lunar',
            formula_latex=r'a_{moon} = G M_{moon} \left(\frac{\mathbf{r}_{moon} - \mathbf{r}}{|\mathbf{r}_{moon} - \mathbf{r}|^3} - \frac{\mathbf{r}_{moon}}{|\mathbf{r}_{moon}|^3}\right)',
            contribution=0.0,
            description='Lunar gravitational perturbation significant for GEO/HEO objects',
        ),
    }

    def __init__(self, n_shap_samples: int = 200):
        self.shap = PerturbationSHAP(n_samples=n_shap_samples)

    def explain_conjunction_alert(self, sensor_data: Dict[str, Dict[str, float]],
                                   predict_fn, miss_distance_km: float,
                                   collision_probability: float,
                                   covariance_age_hours: float = 0.0,
                                   orbital_regime: str = 'LEO') -> AlertExplanation:
        """
        Generate full physics-grounded explanation for a conjunction alert.
        
        Args:
            sensor_data: {sensor_name: {feature_name: value}}
            predict_fn: f(flat_feature_dict) → collision_probability
            miss_distance_km: Predicted miss distance
            collision_probability: Pc from Foster/Alfano
            covariance_age_hours: Hours since last observation
            orbital_regime: LEO/MEO/GEO/HEO
        """
        # 1. Flatten sensor data for SHAP
        flat_features = {}
        feature_meta = {}
        for sensor, feats in sensor_data.items():
            for feat_name, value in feats.items():
                key = f"{sensor}__{feat_name}"
                flat_features[key] = value
                unit = self._get_unit(sensor, feat_name)
                feature_meta[key] = (sensor, feat_name, unit)

        # 2. Compute SHAP values
        shap_values = self.shap.explain(predict_fn, flat_features)

        # 3. Build sensor attributions
        attributions = []
        total_abs = sum(abs(v) for v in shap_values.values()) or 1e-12
        for key, shap_val in shap_values.items():
            sensor, feat_name, unit = feature_meta[key]
            attributions.append(SensorAttribution(
                sensor_name=sensor,
                feature_name=feat_name,
                shap_value=shap_val / total_abs,
                raw_value=flat_features[key],
                unit=unit,
                confidence=min(1.0, abs(flat_features[key]) / (abs(flat_features[key]) + 1e-6)),
            ))
        attributions.sort(key=lambda a: a.abs_contribution, reverse=True)

        # 4. Physics basis
        physics = self._compute_physics_basis(covariance_age_hours, orbital_regime, miss_distance_km)

        # 5. Confidence & uncertainty
        confidence = min(0.99, 1.0 - 0.1 * covariance_age_hours / 24.0)
        margin = max(0.05, 0.5 * (1.0 - confidence))
        pc_lower = max(0, collision_probability - margin * collision_probability)
        pc_upper = min(1, collision_probability + margin * collision_probability)

        # 6. Human summary
        top = attributions[:2] if len(attributions) >= 2 else attributions
        sensor_evidence = "; ".join(a.human_readable() for a in top)
        physics_evidence = physics[0].description if physics else "N/A"

        if collision_probability > 1e-4:
            recommendation = "ESCALATE"
        elif collision_probability > 1e-6:
            recommendation = "MONITOR"
        else:
            recommendation = "LOG"

        return AlertExplanation(
            alert_type="conjunction",
            top_sensors=attributions[:5],
            physics_basis=physics,
            confidence=confidence,
            uncertainty_lower=pc_lower,
            uncertainty_upper=pc_upper,
            human_summary=f"Conjunction Pc={collision_probability:.2e} (miss={miss_distance_km:.1f}km). "
                          f"Primary evidence: {sensor_evidence}. Physics: {physics_evidence}.",
            recommendation=recommendation,
        )

    def explain_maneuver_detection(self, delta_v: float, hamiltonian_violation: float,
                                    cusum_score: float, ml_confidence: float) -> AlertExplanation:
        """Explain a maneuver detection alert."""
        attributions = []
        total = abs(delta_v) + abs(hamiltonian_violation) + abs(cusum_score) + 1e-12
        attributions.append(SensorAttribution("Propagator", "delta_v", delta_v / total, delta_v, "m/s", ml_confidence))
        attributions.append(SensorAttribution("Hamiltonian", "energy_violation", hamiltonian_violation / total,
                                               hamiltonian_violation, "J/kg", 0.95))
        attributions.append(SensorAttribution("CUSUM", "change_score", cusum_score / total, cusum_score, "", 0.90))
        attributions.sort(key=lambda a: a.abs_contribution, reverse=True)

        confidence = ml_confidence
        recommendation = "ESCALATE" if delta_v > 0.1 else "MONITOR"
        return AlertExplanation(
            alert_type="maneuver",
            top_sensors=attributions,
            physics_basis=[PhysicsBasis("maneuver_delta_v", r'\Delta v', delta_v, "Detected thrust event")],
            confidence=confidence,
            uncertainty_lower=max(0, ml_confidence - 0.15),
            uncertainty_upper=min(1, ml_confidence + 0.15),
            human_summary=f"Maneuver detected: ΔV={delta_v:.3f} m/s, Hamiltonian violation={hamiltonian_violation:.2e}, "
                          f"CUSUM={cusum_score:.2f}. Confidence={confidence:.2f}.",
            recommendation=recommendation,
        )

    def explain_rapid_threat(self, mach: float, altitude_km: float, distance_km: float,
                              rcs: float, score: float, ml_confidence: float) -> AlertExplanation:
        """Explain a hypersonic rapid-threat classification."""
        features = {'mach': mach, 'altitude_km': altitude_km, 'distance_km': distance_km, 'rcs': rcs}
        total = sum(abs(v) for v in features.values()) or 1e-12
        attributions = [
            SensorAttribution("Kinematics", "mach", mach / total, mach, "Ma", ml_confidence),
            SensorAttribution("Kinematics", "altitude", altitude_km / total, altitude_km, "km", 0.95),
            SensorAttribution("Geometry", "distance_to_asset", distance_km / total, distance_km, "km", 0.90),
            SensorAttribution("ATeena", "rcs", rcs / total, rcs, "m²", 0.85),
        ]
        attributions.sort(key=lambda a: a.abs_contribution, reverse=True)

        recommendation = "ESCALATE" if score > 80 else ("MONITOR" if score > 50 else "LOG")
        return AlertExplanation(
            alert_type="rapid_threat",
            top_sensors=attributions,
            physics_basis=[],
            confidence=ml_confidence,
            uncertainty_lower=max(0, score - 15),
            uncertainty_upper=min(100, score + 15),
            human_summary=f"Rapid threat score={score:.0f} (Mach {mach:.1f}, alt={altitude_km:.0f}km, "
                          f"range={distance_km:.0f}km, RCS={rcs:.2f}m²). Confidence={ml_confidence:.2f}.",
            recommendation=recommendation,
        )

    # --- Helpers ---

    def _get_unit(self, sensor: str, feature_name: str) -> str:
        for name, unit, _ in self.SENSOR_FEATURES.get(sensor, []):
            if name == feature_name:
                return unit
        return ""

    def _compute_physics_basis(self, cov_age_hours: float, regime: str,
                                miss_km: float) -> List[PhysicsBasis]:
        basis = []
        # Covariance growth dominates when observations are old
        cov_contrib = min(1.0, cov_age_hours / 48.0)
        if cov_contrib > 0.1:
            b = PhysicsBasis(**{**self.PHYSICS_MECHANISMS['covariance_growth'].__dict__})
            b.contribution = cov_contrib
            basis.append(b)
        # Drag uncertainty dominates in LEO
        if regime == 'LEO':
            b = PhysicsBasis(**{**self.PHYSICS_MECHANISMS['drag_uncertainty'].__dict__})
            b.contribution = 0.4
            basis.append(b)
        # J2 always present
        b = PhysicsBasis(**{**self.PHYSICS_MECHANISMS['j2_perturbation'].__dict__})
        b.contribution = 0.15
        basis.append(b)
        # Lunar perturbation significant for GEO/HEO
        if regime in ('GEO', 'HEO'):
            b = PhysicsBasis(**{**self.PHYSICS_MECHANISMS['third_body_lunar'].__dict__})
            b.contribution = 0.25
            basis.append(b)

        basis.sort(key=lambda p: p.contribution, reverse=True)
        return basis


# --- Quick Self-Test ---

if __name__ == "__main__":
    explainer = PhysicsSHAPExplainer(n_shap_samples=50)

    # Mock conjunction
    sensor_data = {
        'ATeena': {'rcs_fluctuation': 2.5, 'doppler_shift': 150.0, 'snr': 12.0},
        'RISAT-SAR': {'isar_glint': -8.0, 'range_profile_length': 3.2},
        'Optical-CCD': {'apparent_magnitude': 8.5},
    }
    predict_fn = lambda f: 1e-4 * (f.get('ATeena__rcs_fluctuation', 0) * 0.4 +
                                     abs(f.get('RISAT-SAR__isar_glint', 0)) * 0.3 +
                                     f.get('ATeena__snr', 0) * 0.01)
    explanation = explainer.explain_conjunction_alert(
        sensor_data, predict_fn,
        miss_distance_km=0.8, collision_probability=3.5e-5,
        covariance_age_hours=6.0, orbital_regime='LEO',
    )
    print(f"Summary: {explanation.human_summary}")
    print(f"Recommendation: {explanation.recommendation}")
    for a in explanation.top_sensors[:3]:
        print(f"  {a.human_readable()}")
    for p in explanation.physics_basis[:2]:
        print(f"  Physics: {p.description} (contribution={p.contribution:.2f})")
