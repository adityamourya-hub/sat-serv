"""
MGE Template Models for Known Adversary Satellite Platforms
Provides template matching for architectural fingerprinting of foreign satellite buses
"""

import numpy as np
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


class MGETemplateLibrary:
    """
    Library of MGE templates for known adversary satellite platforms
    Enables real-time template matching for UCT identification
    """
    
    def __init__(self):
        self.templates: Dict[str, SatelliteTemplate] = {}
        self._initialize_known_templates()
    
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
    
    def add_template(self, template: SatelliteTemplate):
        """Add a new satellite template to the library"""
        self.templates[template.name] = template
    
    def match_template(self, mge_result: MGEResult, 
                      rcs: float = None,
                      thermal: float = None) -> List[Tuple[str, float]]:
        """
        Match an MGE result against known templates
        
        Args:
            mge_result: MGE analysis result
            rcs: Optional radar cross-section
            thermal: Optional thermal signature
            
        Returns:
            List of (template_name, match_score) tuples, sorted by score
        """
        matches = []
        
        for name, template in self.templates.items():
            score = self._calculate_match_score(mge_result, template, rcs, thermal)
            if score > 0.5:  # Minimum match threshold
                matches.append((name, score))
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _calculate_match_score(self, mge_result: MGEResult, 
                               template: SatelliteTemplate,
                               rcs: float = None,
                               thermal: float = None) -> float:
        """
        Calculate match score between MGE result and template
        
        Returns:
            Score between 0.0 and 1.0
        """
        scores = []
        
        # Ratio match (primary indicator)
        ratio_diff = abs(mge_result.array_to_bus_ratio - template.expected_ratio)
        if ratio_diff <= template.ratio_tolerance:
            ratio_score = 1.0 - (ratio_diff / template.ratio_tolerance) * 0.5
            scores.append(ratio_score)
        else:
            return 0.0  # Outside tolerance
        
        # Architecture type match
        if mge_result.architecture_type == template.expected_architecture:
            scores.append(1.0)
        else:
            scores.append(0.5)  # Partial credit
        
        # Component count match
        if template.min_components <= mge_result.total_components <= template.max_components:
            scores.append(1.0)
        else:
            scores.append(0.3)
        
        # RCS match (if available)
        if rcs is not None:
            rcs_diff = abs(rcs - template.rcs_signature)
            if rcs_diff < template.rcs_signature * 0.3:
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Thermal match (if available)
        if thermal is not None:
            thermal_diff = abs(thermal - template.thermal_signature)
            if thermal_diff < 30.0:  # Within 30K
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Calculate weighted average
        if not scores:
            return 0.0
        
        # Weight ratio more heavily
        weights = [0.4] + [0.15] * (len(scores) - 1)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return weighted_score
    
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


class TemplateMatcher:
    """
    Real-time template matching system for UCT identification
    """
    
    def __init__(self, template_library: MGETemplateLibrary = None):
        self.template_library = template_library or MGETemplateLibrary()
        self.match_history: Dict[int, List[Tuple[str, float]]] = {}  # norad_id -> matches
    
    def identify_satellite(self, norad_id: int, mge_result: MGEResult,
                           rcs: float = None, thermal: float = None) -> Dict:
        """
        Attempt to identify a satellite from its MGE signature
        
        Args:
            norad_id: NORAD catalog ID
            mge_result: MGE analysis result
            rcs: Optional radar cross-section
            thermal: Optional thermal signature
            
        Returns:
            Dictionary with identification results
        """
        matches = self.template_library.match_template(mge_result, rcs, thermal)
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
            "possible_matches": matches[:3]  # Top 3 matches
        }
    
    def get_historical_matches(self, norad_id: int) -> List[Tuple[str, float]]:
        """Get match history for a satellite"""
        return self.match_history.get(norad_id, [])


# Example usage and testing
if __name__ == "__main__":
    # Create template library
    library = MGETemplateLibrary()
    
    # Create a sample MGE result that matches Yaogan-33
    sample_result = MGEResult(
        array_to_bus_ratio=10.2,
        total_components=9,
        bus_components=2,
        array_components=7,
        architecture_type="HIGH_POWER_EW_SAR",
        components=[],
        status="SUCCESS",
        timestamp=0.0
    )
    
    # Match against templates
    matcher = TemplateMatcher(library)
    result = matcher.identify_satellite(
        norad_id=99999,
        mge_result=sample_result,
        rcs=12.5,
        thermal=315.0
    )
    
    print("Template Matching Results:")
    print(f"Identified: {result['identified']}")
    print(f"Best Match: {result['best_match']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result['template_info']:
        info = result['template_info']
        print(f"Country: {info['country']}")
        print(f"Purpose: {info['purpose']}")
        print(f"Threat Level: {info['threat_level']}")
    
    print("\nTop 3 Matches:")
    for name, score in result['possible_matches']:
        print(f"  {name}: {score:.2f}")
