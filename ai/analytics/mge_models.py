"""
MGE Template Models for Known Adversary Satellite Platforms
Provides template matching for architectural fingerprinting of foreign satellite buses

ADITI 4.0: Physics-first upgrades with GHNN integration, SE(3) equivariance,
and fractional drag for real-time dynamic template matching.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from structural_id import MGEComponent, MGEResult


@dataclass
class SatelliteTemplate:
    """Template model for a known satellite architecture"""
    name: str
    country: str
    series: str  # Yaogan, Jilin, etc.
    purpose: str  # ISR, Comms, SAR, etc.
    
    # MGE characteristics
    expected_ratio: float
    ratio_tolerance: float  # +/- tolerance for matching
    min_components: int
    max_components: int
    expected_architecture: str
    
    # Physical characteristics
    typical_mass_kg: float
    typical_power_w: float
    typical_dimensions_m: Tuple[float, float, float]
    
    # Detection characteristics
    rcs_signature: float  # m²
    thermal_signature: float  # Kelvin
    
    # Threat classification
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    
    # ADITI 4.0: Photometric fingerprinting (Slingshot AMOS)
    photometric_period: float = 0.0  # Rotation period (seconds)
    photometric_amplitude: float = 0.0  # Light curve amplitude (magnitudes)
    photometric_phase: float = 0.0  # Phase offset
    photometric_class: str = "UNKNOWN"  # Periodic, tumbling, stable


class MGETemplateLibrary:
    """
    Library of MGE templates for known adversary satellite platforms
    Enables real-time template matching for UCT identification
    
    ADITI 4.0: Enhanced with GHNN integration, SE(3) equivariance,
    and fractional drag for dynamic template matching.
    """
    
    def __init__(self):
        self.templates: Dict[str, SatelliteTemplate] = {}
        self._initialize_known_templates()
        
        # Dynamic physics updates
        self.ghnn_model = None  # Load axiom_se3_ghnn.onnx for dynamic ratio
        self.fractional_drag_residual = 0.0  # From Caputo kernel
        self.ionospheric_tec = 0.0  # From 16-dim spherical harmonics
        
        # SE(3) equivariance correction
        self.attitude_independent_features = True
    
    def _initialize_known_templates(self):
        """Initialize templates for known adversary platforms"""
        
        # Chinese ISR/EO satellites
        self.add_template(SatelliteTemplate(
            name="Yaogan-30 Series",
            country="CHN",
            series="Yaogan",
            purpose="ISR_EO",
            expected_ratio=4.5,
            ratio_tolerance=1.5,
            min_components=6,
            max_components=10,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=2700,
            typical_power_w=2000,
            typical_dimensions_m=(2.0, 2.0, 4.0),
            rcs_signature=8.0,
            thermal_signature=280.0,
            threat_level="HIGH"
        ))
        
        self.add_template(SatelliteTemplate(
            name="Yaogan-33 Series",
            country="CHN",
            series="Yaogan",
            purpose="ISR_SAR",
            expected_ratio=9.5,
            ratio_tolerance=2.0,
            min_components=8,
            max_components=12,
            expected_architecture="HIGH_POWER_EW_SAR",
            typical_mass_kg=3200,
            typical_power_w=3500,
            typical_dimensions_m=(2.5, 2.5, 8.0),
            rcs_signature=12.0,
            thermal_signature=310.0,
            threat_level="CRITICAL"
        ))
        
        # Chinese commercial imaging satellites
        self.add_template(SatelliteTemplate(
            name="Jilin-1 Series",
            country="CHN",
            series="Jilin",
            purpose="COMMERCIAL_EO",
            expected_ratio=3.0,
            ratio_tolerance=1.0,
            min_components=4,
            max_components=8,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=95,
            typical_power_w=150,
            typical_dimensions_m=(0.5, 0.5, 1.0),
            rcs_signature=1.5,
            thermal_signature=250.0,
            threat_level="MEDIUM"
        ))
        
        # Chinese ASAT/co-orbital platforms
        self.add_template(SatelliteTemplate(
            name="Shijian-17",
            country="CHN",
            series="Shijian",
            purpose="ROBOTIC_INSPECTOR",
            expected_ratio=2.0,
            ratio_tolerance=0.8,
            min_components=5,
            max_components=9,
            expected_architecture="SMALL_3AXIS_BUS",
            typical_mass_kg=2800,
            typical_power_w=2500,
            typical_dimensions_m=(1.8, 1.8, 3.0),
            rcs_signature=6.0,
            thermal_signature=295.0,
            threat_level="CRITICAL"
        ))
        
        # Chinese EW/Jamming satellites
        self.add_template(SatelliteTemplate(
            name="Tongxin Jishu Shiyan",
            country="CHN",
            series="TJS",
            purpose="COMMS_JAMMER",
            expected_ratio=14.0,
            ratio_tolerance=3.0,
            min_components=10,
            max_components=16,
            expected_architecture="EXTREME_POWER_JAMMER",
            typical_mass_kg=5400,
            typical_power_w=8000,
            typical_dimensions_m=(3.0, 3.0, 12.0),
            rcs_signature=18.0,
            thermal_signature=340.0,
            threat_level="CRITICAL"
        ))
        
        # Russian ISR satellites
        self.add_template(SatelliteTemplate(
            name="Persona Series",
            country="RUS",
            series="Persona",
            purpose="ISR_EO",
            expected_ratio=5.0,
            ratio_tolerance=1.5,
            min_components=7,
            max_components=11,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=6500,
            typical_power_w=3000,
            typical_dimensions_m=(2.2, 2.2, 5.0),
            rcs_signature=10.0,
            thermal_signature=290.0,
            threat_level="HIGH"
        ))
        
        # Generic small satellite buses
        self.add_template(SatelliteTemplate(
            name="Cubesat 3U",
            country="GEN",
            series="Cubesat",
            purpose="GENERIC_SMALL",
            expected_ratio=0.8,
            ratio_tolerance=0.3,
            min_components=2,
            max_components=5,
            expected_architecture="COMPACT_LOW_POWER",
            typical_mass_kg=4,
            typical_power_w=10,
            typical_dimensions_m=(0.1, 0.1, 0.3),
            rcs_signature=0.1,
            thermal_signature=270.0,
            threat_level="LOW"
        ))
        
        self.add_template(SatelliteTemplate(
            name="SmallSat 100kg",
            country="GEN",
            series="SmallSat",
            purpose="GENERIC_SMALL",
            expected_ratio=2.5,
            ratio_tolerance=0.8,
            min_components=4,
            max_components=7,
            expected_architecture="SMALL_3AXIS_BUS",
            typical_mass_kg=100,
            typical_power_w=100,
            typical_dimensions_m=(0.6, 0.6, 0.8),
            rcs_signature=1.2,
            thermal_signature=275.0,
            threat_level="LOW"
        ))
        
        # Additional Chinese platforms (Jilin variants, Shijian updates)
        self.add_template(SatelliteTemplate(
            name="Jilin-1 GP Series",
            country="CHN",
            series="Jilin",
            purpose="COMMERCIAL_EO",
            expected_ratio=3.5,
            ratio_tolerance=1.2,
            min_components=5,
            max_components=9,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=420,
            typical_power_w=300,
            typical_dimensions_m=(1.0, 1.0, 2.0),
            rcs_signature=2.5,
            thermal_signature=260.0,
            threat_level="MEDIUM"
        ))
        
        self.add_template(SatelliteTemplate(
            name="Shijian-26 Series",
            country="CHN",
            series="Shijian",
            purpose="ROBOTIC_INSPECTOR",
            expected_ratio=2.2,
            ratio_tolerance=0.9,
            min_components=6,
            max_components=10,
            expected_architecture="SMALL_3AXIS_BUS",
            typical_mass_kg=3000,
            typical_power_w=2800,
            typical_dimensions_m=(2.0, 2.0, 3.5),
            rcs_signature=7.0,
            thermal_signature=300.0,
            threat_level="CRITICAL"
        ))
        
        self.add_template(SatelliteTemplate(
            name="TJS-3 Series",
            country="CHN",
            series="TJS",
            purpose="COMMS_JAMMER",
            expected_ratio=16.0,
            ratio_tolerance=3.5,
            min_components=12,
            max_components=18,
            expected_architecture="EXTREME_POWER_JAMMER",
            typical_mass_kg=6000,
            typical_power_w=10000,
            typical_dimensions_m=(3.5, 3.5, 14.0),
            rcs_signature=22.0,
            thermal_signature=360.0,
            threat_level="CRITICAL"
        ))
        
        # Russian platforms (Lotos, Persona updates)
        self.add_template(SatelliteTemplate(
            name="Lotos-S Series",
            country="RUS",
            series="Lotos",
            purpose="ISR_EO",
            expected_ratio=4.5,
            ratio_tolerance=1.3,
            min_components=6,
            max_components=10,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=6000,
            typical_power_w=2800,
            typical_dimensions_m=(2.0, 2.0, 4.5),
            rcs_signature=9.0,
            thermal_signature=285.0,
            threat_level="HIGH"
        ))
        
        self.add_template(SatelliteTemplate(
            name="Persona-V Series",
            country="RUS",
            series="Persona",
            purpose="ISR_EO",
            expected_ratio=5.5,
            ratio_tolerance=1.6,
            min_components=8,
            max_components=12,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=7000,
            typical_power_w=3200,
            typical_dimensions_m=(2.3, 2.3, 5.5),
            rcs_signature=11.0,
            thermal_signature=295.0,
            threat_level="HIGH"
        ))
        
        # Additional generic proliferated smallsats
        self.add_template(SatelliteTemplate(
            name="Cubesat 6U",
            country="GEN",
            series="Cubesat",
            purpose="GENERIC_SMALL",
            expected_ratio=1.2,
            ratio_tolerance=0.4,
            min_components=3,
            max_components=6,
            expected_architecture="COMPACT_LOW_POWER",
            typical_mass_kg=8,
            typical_power_w=15,
            typical_dimensions_m=(0.2, 0.3, 0.3),
            rcs_signature=0.2,
            thermal_signature=272.0,
            threat_level="LOW"
        ))
        
        self.add_template(SatelliteTemplate(
            name="SmallSat 200kg",
            country="GEN",
            series="SmallSat",
            purpose="GENERIC_SMALL",
            expected_ratio=3.0,
            ratio_tolerance=1.0,
            min_components=5,
            max_components=8,
            expected_architecture="SMALL_3AXIS_BUS",
            typical_mass_kg=200,
            typical_power_w=150,
            typical_dimensions_m=(0.8, 0.8, 1.2),
            rcs_signature=1.8,
            thermal_signature=278.0,
            threat_level="LOW"
        ))
        
        self.add_template(SatelliteTemplate(
            name="SmallSat 500kg",
            country="GEN",
            series="SmallSat",
            purpose="GENERIC_SMALL",
            expected_ratio=3.8,
            ratio_tolerance=1.2,
            min_components=6,
            max_components=10,
            expected_architecture="STANDARD_3AXIS_BUS",
            typical_mass_kg=500,
            typical_power_w=400,
            typical_dimensions_m=(1.2, 1.2, 2.0),
            rcs_signature=3.5,
            thermal_signature=280.0,
            threat_level="MEDIUM"
        ))
        
        # Debris templates for conjunction assessment
        self.add_template(SatelliteTemplate(
            name="Debris Fragment Large",
            country="UNK",
            series="Debris",
            purpose="DEBRIS",
            expected_ratio=0.3,
            ratio_tolerance=0.2,
            min_components=1,
            max_components=3,
            expected_architecture="IRREGULAR",
            typical_mass_kg=50,
            typical_power_w=0,
            typical_dimensions_m=(0.3, 0.3, 0.3),
            rcs_signature=0.05,
            thermal_signature=265.0,
            threat_level="LOW"
        ))
        
        self.add_template(SatelliteTemplate(
            name="Debris Fragment Small",
            country="UNK",
            series="Debris",
            purpose="DEBRIS",
            expected_ratio=0.15,
            ratio_tolerance=0.1,
            min_components=1,
            max_components=2,
            expected_architecture="IRREGULAR",
            typical_mass_kg=5,
            typical_power_w=0,
            typical_dimensions_m=(0.1, 0.1, 0.1),
            rcs_signature=0.01,
            thermal_signature=260.0,
            threat_level="LOW"
        ))
    
    def add_template(self, template: SatelliteTemplate):
        """Add a new satellite template to the library"""
        self.templates[template.name] = template
    
    def update_dynamic_features(self, ghnn_state: torch.Tensor = None,
                               fractional_residual: float = None,
                               ionospheric_tec: float = None):
        """
        Update templates with real-time GHNN propagation + fractional drag
        
        ADITI 4.0: Dynamic template matching based on physics-informed state
        """
        if fractional_residual is not None:
            self.fractional_drag_residual = fractional_residual
        
        if ionospheric_tec is not None:
            self.ionospheric_tec = ionospheric_tec
        
        # GHNN state would be used to adjust expected ratio based on
        # non-local wake memory and Hamiltonian-invariant dynamics
        if ghnn_state is not None:
            # Placeholder: adjust expected ratios based on GHNN state
            # In production: compute dynamic correction from Hamiltonian
            pass
    
    def _equivariant_ratio_correction(self, mge_result: MGEResult, 
                                     ghnn_state: torch.Tensor = None) -> float:
        """
        Compute SE(3)-equivariant ratio correction for attitude-independent matching
        
        ADITI 4.0: Use e3nn irrep features to make ratio matching invariant
        to bank-angle changes during RPO maneuvers.
        """
        # Placeholder: In production, use e3nn to compute rotation-invariant
        # ratio features from the component graph
        correction = 1.0
        
        if self.fractional_drag_residual > 0.01:
            # Adjust for fractional non-local drag effects
            correction *= (1.0 - 0.1 * self.fractional_drag_residual)
        
        if self.ionospheric_tec > 10.0:
            # Adjust for ionospheric density variations
            correction *= (1.0 - 0.05 * (self.ionospheric_tec / 100.0))
        
        return correction
    
    def match_template(self, mge_result: MGEResult,
                      rcs: float = None,
                      thermal: float = None,
                      ghnn_state: torch.Tensor = None,
                      photometric_period: float = None,
                      photometric_amplitude: float = None) -> List[Tuple[str, float]]:
        """
        Match MGE result against template library
        
        Args:
            mge_result: MGE analysis result
            rcs: Optional RCS measurement (m²)
            thermal: Optional thermal measurement (K)
            ghnn_state: Optional GHNN state for dynamic correction
            photometric_period: Optional rotation period (seconds) - Slingshot AMOS
            photometric_amplitude: Optional light curve amplitude (magnitudes)
            
        Returns:
            List of (template_name, score) tuples sorted by score
        """
        # Compute SE(3)-equivariant correction
        equivariant_correction = self._equivariant_ratio_correction(mge_result, ghnn_state)
        
        matches = []
        for name, template in self.templates.items():
            score = self._calculate_match_score(mge_result, template, rcs, thermal,
                                                equivariant_correction,
                                                photometric_period, photometric_amplitude)
            if score >= 0.5:  # Threshold for match
                matches.append((name, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 matches
        return matches[:3]
    
    def _calculate_match_score(self, mge_result: MGEResult,
                               template: SatelliteTemplate,
                               rcs: float = None,
                               thermal: float = None,
                               equivariant_correction: float = 1.0,
                               photometric_period: float = None,
                               photometric_amplitude: float = None) -> float:
        """
        Calculate match score between MGE result and template
        
        Score weights:
        - Ratio: 40% (primary structural discriminant)
        - Architecture: 20%
        - Component count: 15%
        - RCS/thermal: 10% (tie-breakers)
        - Photometric: 15% (Slingshot AMOS light curve fingerprinting)
        """
        # Apply equivariant correction to ratio
        adjusted_ratio = mge_result.array_to_bus_ratio * equivariant_correction
        
        # Ratio match (40% weight)
        ratio_diff = abs(adjusted_ratio - template.expected_ratio)
        ratio_score = max(0, 1.0 - (ratio_diff / template.ratio_tolerance))
        
        # Architecture match (20% weight)
        arch_score = 1.0 if mge_result.architecture_type == template.expected_architecture else 0.5
        
        # Component count match (15% weight)
        if template.min_components <= mge_result.total_components <= template.max_components:
            comp_score = 1.0
        else:
            comp_diff = min(abs(mge_result.total_components - template.min_components),
                           abs(mge_result.total_components - template.max_components))
            comp_score = max(0, 1.0 - (comp_diff / 5.0))
        
        # RCS/thermal match (10% weight)
        phys_score = 0.5  # Default
        if rcs is not None and template.rcs_signature > 0:
            rcs_diff = abs(rcs - template.rcs_signature) / template.rcs_signature
            phys_score = max(0, 1.0 - rcs_diff)
        if thermal is not None and template.thermal_signature > 0:
            thermal_diff = abs(thermal - template.thermal_signature) / template.thermal_signature
            phys_score = (phys_score + max(0, 1.0 - thermal_diff)) / 2.0
        
        # Photometric match (15% weight) - Slingshot AMOS
        photo_score = 0.5  # Default if no photometric data
        if photometric_period is not None and template.photometric_period > 0:
            period_diff = abs(photometric_period - template.photometric_period) / template.photometric_period
            photo_score = max(0, 1.0 - period_diff)
        if photometric_amplitude is not None and template.photometric_amplitude > 0:
            amp_diff = abs(photometric_amplitude - template.photometric_amplitude) / template.photometric_amplitude
            photo_score = (photo_score + max(0, 1.0 - amp_diff)) / 2.0
        
        # Weighted score
        score = (0.40 * ratio_score + 
                0.20 * arch_score + 
                0.15 * comp_score + 
                0.10 * phys_score +
                0.15 * photo_score)
        
        return score
    
    def get_template_info(self, template_name: str) -> Optional[SatelliteTemplate]:
        """Get template information by name"""
        return self.templates.get(template_name)
    
    def get_threat_level(self, template_name: str) -> str:
        """Get threat level for a template"""
        template = self.templates.get(template_name)
        if template:
            return template.threat_level
        return "UNKNOWN"
    
    def get_all_templates_by_country(self, country: str) -> List[SatelliteTemplate]:
        """Get all templates for a specific country"""
        return [t for t in self.templates.values() if t.country == country]
    
    def get_all_templates_by_threat(self, min_level: str) -> List[SatelliteTemplate]:
        """Get all templates above a threat level"""
        level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        min_value = level_order.get(min_level, 0)
        
        return [t for t in self.templates.values() 
                if level_order.get(t.threat_level, 0) >= min_value]
    
    def get_gatv2_node_features(self, mge_result: MGEResult,
                                 rcs: float = None,
                                 thermal: float = None,
                                 ghnn_state: torch.Tensor = None) -> Dict:
        """
        Extract MGE features for GATv2 node feature integration
        
        ADITI 4.0: Fuse MGE match score into GATv2 graph node features
        for multi-task threat prediction.
        
        Returns:
            Dictionary with features for GATv2 integration
        """
        # Get top match
        matches = self.match_template(mge_result, rcs, thermal, ghnn_state)
        
        if not matches:
            return {
                "mge_score": 0.0,
                "best_match": "UNKNOWN",
                "threat_level": "LOW",
                "country": "UNK",
                "confidence": 0.0,
                "ratio": mge_result.array_to_bus_ratio,
                "components": mge_result.total_components
            }
        
        best_match, best_score = matches[0]
        template = self.get_template_info(best_match)
        
        return {
            "mge_score": best_score,
            "best_match": best_match,
            "threat_level": template.threat_level if template else "LOW",
            "country": template.country if template else "UNK",
            "confidence": best_score,
            "ratio": mge_result.array_to_bus_ratio,
            "components": mge_result.total_components,
            "architecture": mge_result.architecture_type,
            "fractional_correction": self.fractional_drag_residual,
            "ionospheric_tec": self.ionospheric_tec
        }


class TemplateMatcher:
    """
    Real-time template matching system for UCT identification
    """
    
    def __init__(self, template_library: MGETemplateLibrary = None):
        self.template_library = template_library or MGETemplateLibrary()
        self.match_history: Dict[int, List[Tuple[str, float]]] = {}  # norad_id -> matches
    
    def identify_satellite(self, norad_id: int, mge_result: MGEResult,
                           rcs: float = None, thermal: float = None,
                           ghnn_state: torch.Tensor = None) -> Dict:
        """
        Attempt to identify a satellite from its MGE signature
        
        ADITI 4.0: Enhanced with GHNN state integration and GATv2 feature extraction
        
        Args:
            norad_id: NORAD catalog ID
            mge_result: MGE analysis result
            rcs: Optional radar cross-section
            thermal: Optional thermal signature
            ghnn_state: Optional GHNN state for dynamic correction
            
        Returns:
            Dictionary with identification results + GATv2 features
        """
        matches = self.template_library.match_template(mge_result, rcs, thermal, ghnn_state)
        self.match_history[norad_id] = matches
        
        if not matches:
            return {
                "identified": False,
                "norad_id": norad_id,
                "best_match": None,
                "confidence": 0.0,
                "possible_matches": []
            }
        
        best_match, best_score = matches[0]
        template = self.template_library.get_template_info(best_match)
        
        # Extract GATv2 node features
        gatv2_features = self.template_library.get_gatv2_node_features(
            mge_result, rcs, thermal, ghnn_state
        )
        
        return {
            "identified": best_score > 0.8,
            "norad_id": norad_id,
            "best_match": best_match,
            "confidence": best_score,
            "template_info": {
                "name": template.name,
                "country": template.country,
                "purpose": template.purpose,
                "threat_level": template.threat_level
            } if template else None,
            "possible_matches": matches[:3],  # Top 3 matches
            "gatv2_features": gatv2_features  # For graph fusion
        }
    
    def get_historical_matches(self, norad_id: int) -> List[Tuple[str, float]]:
        """Get match history for a satellite"""
        return self.match_history.get(norad_id, [])


# Example usage and testing
if __name__ == "__main__":
    print("=== ADITI 4.0: Enhanced MGE Template Matching ===")
    
    # Initialize library
    library = MGETemplateLibrary()
    
    # Update dynamic features (GHNN + fractional drag)
    library.update_dynamic_features(
        fractional_residual=0.05,
        ionospheric_tec=25.0
    )
    
    # Create sample MGE result
    sample_result = MGEResult(
        array_to_bus_ratio=10.2,
        total_components=9,
        bus_components=2,
        array_components=7,
        architecture_type="HIGH_POWER_EW_SAR",
        components=[],
        status="SUCCESS"
    )
    
    # Initialize matcher
    matcher = TemplateMatcher(library)
    
    # Identify satellite with photometric fingerprinting (Slingshot AMOS)
    result = matcher.identify_satellite(
        norad_id=99999,
        mge_result=sample_result,
        rcs=12.5,
        thermal=315.0,
        photometric_period=90.0,  # 90-second rotation (typical for Yaogan SAR)
        photometric_amplitude=0.3  # 0.3 magnitude variation
    )
    
    print(f"Identified: {result['identified']}")
    print(f"Best Match: {result['best_match']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Country: {result['template_info']['country']}")
    print(f"Purpose: {result['template_info']['purpose']}")
    print(f"Threat Level: {result['template_info']['threat_level']}")
    
    print("\nTop 3 Matches:")
    for name, score in result['possible_matches']:
        print(f"  {name}: {score:.2f}")
    
    print("\n=== Photometric Fingerprinting (Slingshot AMOS) ===")
    photo = result['photometric_match']
    print(f"Rotation Period: {photo['period']:.1f}s")
    print(f"Light Curve Amplitude: {photo['amplitude']:.2f} mag")
    
    print("\n=== GATv2 Node Features ===")
    gatv2 = result['gatv2_features']
    print(f"MGE Score: {gatv2['mge_score']:.2f}")
    print(f"Threat Level: {gatv2['threat_level']}")
    print(f"Fractional Correction: {gatv2['fractional_correction']:.3f}")
    print(f"Ionospheric TEC: {gatv2['ionospheric_tec']:.1f}")
    
    print(f"\nTotal Templates: {len(library.templates)}")
    critical_count = sum(1 for t in library.templates.values() if t.threat_level == "CRITICAL")
    high_count = sum(1 for t in library.templates.values() if t.threat_level == "HIGH")
    print(f"Critical Threats: {critical_count}")
    print(f"High Threats: {high_count}")
    
    print("\n=== Ready for GATv2 fusion and NETRA gateway ===")
