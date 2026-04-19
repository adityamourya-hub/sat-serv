"""
Spectroscopic Characterization Engine for RSO Material Identification
Uses MPDAF for spectral analysis and cross-correlation with material library
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaterialSignature:
    """Spectral signature for a satellite material"""
    name: str
    wavelengths: np.ndarray  # Wavelength array (nm or cm^-1)
    intensity: np.ndarray    # Normalized intensity
    peaks: list              # Peak locations for quick matching
    tactical_significance: str
    threat_level: int        # 0-10 scale


class MaterialLibrary:
    """
    Library of known satellite material spectral signatures
    """
    
    def __init__(self):
        self.materials: Dict[str, MaterialSignature] = {}
        self._init_standard_materials()
    
    def _init_standard_materials(self):
        """Initialize standard satellite material signatures"""
        
        # GaAs (Gallium Arsenide) - High-efficiency solar arrays
        gaas_wl = np.linspace(200, 800, 600)  # nm
        gaas_intensity = self._generate_gaas_spectrum(gaas_wl)
        self.materials["GaAs_SOLAR_PANEL"] = MaterialSignature(
            name="GaAs (Gallium Arsenide)",
            wavelengths=gaas_wl,
            intensity=gaas_intensity,
            peaks=[265, 289],  # LO/TO peaks in cm^-1
            tactical_significance="High-efficiency solar arrays (Active Power)",
            threat_level=2
        )
        
        # Silicon Solar Panel - Standard arrays
        si_wl = np.linspace(200, 1200, 1000)
        si_intensity = self._generate_silicon_spectrum(si_wl)
        self.materials["Si_SOLAR_PANEL"] = MaterialSignature(
            name="Silicon Solar Panel",
            wavelengths=si_wl,
            intensity=si_intensity,
            peaks=[520],  # Silicon characteristic
            tactical_significance="Standard solar arrays",
            threat_level=1
        )
        
        # Kevlar/Composites - Structural shielding
        kevlar_wl = np.linspace(1000, 1800, 800)
        kevlar_intensity = self._generate_kevlar_spectrum(kevlar_wl)
        self.materials["KEVLAR_COMPOSITE"] = MaterialSignature(
            name="Kevlar/Composite Structure",
            wavelengths=kevlar_wl,
            intensity=kevlar_intensity,
            peaks=[1613],  # C-C stretching
            tactical_significance="Structural shielding or robotic arm joints",
            threat_level=5
        )
        
        # Kapton (Aluminized) - Multi-layer insulation
        kapton_wl = np.linspace(500, 2500, 2000)
        kapton_intensity = self._generate_kapton_spectrum(kapton_wl)
        self.materials["KAPTON_MLI"] = MaterialSignature(
            name="Kapton MLI",
            wavelengths=kapton_wl,
            intensity=kapton_intensity,
            peaks=[1370, 1720],  # NIR emissivity
            tactical_significance="Multi-layer insulation (Standard Satellite)",
            threat_level=1
        )
        
        # Stealth Coating - Broadband absorption
        stealth_wl = np.linspace(200, 2500, 2300)
        stealth_intensity = self._generate_stealth_spectrum(stealth_wl)
        self.materials["STEALTH_COATING"] = MaterialSignature(
            name="Stealth/Anti-Reflection Coating",
            wavelengths=stealth_wl,
            intensity=stealth_intensity,
            peaks=[],  # Flat, low reflection
            tactical_significance="THREAT: Signature reduction/ASAT",
            threat_level=9
        )
        
        # Optical Glass (SiO2) - High-res lenses
        glass_wl = np.linspace(300, 1100, 800)
        glass_intensity = self._generate_glass_spectrum(glass_wl)
        self.materials["OPTICAL_GLASS"] = MaterialSignature(
            name="Optical Glass (SiO2)",
            wavelengths=glass_wl,
            intensity=glass_intensity,
            peaks=[450, 1050],  # Transmission bands
            tactical_significance="High-resolution optical payload",
            threat_level=7
        )
    
    def _generate_gaas_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate GaAs spectral signature with characteristic peaks"""
        intensity = np.ones_like(wl, dtype=float)
        # Absorption edge around 870nm
        intensity[wl > 870] *= 0.1
        # Add LO/TO phonon peaks (would be in IR, simplified here)
        intensity += 0.3 * np.exp(-((wl - 650)**2) / (2 * 50**2))
        return intensity / np.max(intensity)
    
    def _generate_silicon_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate silicon solar panel spectrum"""
        intensity = np.ones_like(wl, dtype=float)
        # Band gap absorption
        intensity[wl > 1100] *= 0.05
        # Reflectance minimum around 500nm
        intensity -= 0.2 * np.exp(-((wl - 500)**2) / (2 * 100**2))
        return np.clip(intensity / np.max(intensity), 0.1, 1.0)
    
    def _generate_kevlar_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate Kevlar composite signature"""
        intensity = np.ones_like(wl, dtype=float) * 0.5
        # C-C stretching peak
        intensity += 0.4 * np.exp(-((wl - 1613)**2) / (2 * 30**2))
        # N-H stretch
        intensity += 0.2 * np.exp(-((wl - 1540)**2) / (2 * 40**2))
        return intensity / np.max(intensity)
    
    def _generate_kapton_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate Kapton MLI signature"""
        intensity = np.ones_like(wl, dtype=float) * 0.3
        # NIR emissivity features
        intensity += 0.5 * np.exp(-((wl - 1370)**2) / (2 * 50**2))
        intensity += 0.3 * np.exp(-((wl - 1720)**2) / (2 * 60**2))
        # Aluminum reflection
        intensity[wl < 800] += 0.4
        return np.clip(intensity / np.max(intensity), 0.1, 1.0)
    
    def _generate_stealth_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate stealth coating (broadband absorption)"""
        # Flat, low intensity across spectrum
        return np.ones_like(wl, dtype=float) * 0.1 + np.random.normal(0, 0.02, len(wl))
    
    def _generate_glass_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate optical glass (SiO2) transmission"""
        intensity = np.ones_like(wl, dtype=float) * 0.9
        # UV absorption
        intensity[wl < 350] *= 0.1
        # IR absorption
        intensity[wl > 1000] *= 0.3
        # Transmission peaks
        intensity += 0.1 * np.exp(-((wl - 550)**2) / (2 * 100**2))
        return np.clip(intensity, 0.1, 1.0)
    
    def get_material(self, name: str) -> Optional[MaterialSignature]:
        """Get material signature by name"""
        return self.materials.get(name)
    
    def list_materials(self) -> list:
        """List all available materials"""
        return list(self.materials.keys())


class SpectralAnalyzer:
    """
    Performs spectral cross-correlation for material identification
    """
    
    def __init__(self, library: Optional[MaterialLibrary] = None):
        self.library = library or MaterialLibrary()
    
    def identify_material(
        self,
        observed_wavelengths: np.ndarray,
        observed_intensity: np.ndarray,
        top_n: int = 3
    ) -> Tuple[str, float, list]:
        """
        Identify material by cross-correlation with library
        
        Returns:
            (best_match, confidence_score, top_matches_list)
        """
        # Normalize observed spectrum
        obs_norm = self._normalize_spectrum(observed_intensity)
        
        scores = {}
        
        for mat_name, signature in self.library.materials.items():
            # Interpolate library spectrum to observed wavelengths
            ref_interp = np.interp(
                observed_wavelengths,
                signature.wavelengths,
                signature.intensity,
                left=0, right=0
            )
            ref_norm = self._normalize_spectrum(ref_interp)
            
            # Cross-correlation
            correlation = np.correlate(obs_norm, ref_norm, mode='full')
            max_corr = np.max(correlation)
            
            # Normalize by theoretical maximum
            scores[mat_name] = max_corr / len(obs_norm)
        
        # Get best match
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]
        
        # Get top N matches
        sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        logger.info(f"Material identification: {best_match} (confidence: {confidence:.3f})")
        
        return best_match, confidence, sorted_matches
    
    def _normalize_spectrum(self, intensity: np.ndarray) -> np.ndarray:
        """Normalize spectrum to [0, 1] range"""
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        if max_val - min_val < 1e-10:
            return np.zeros_like(intensity)
        return (intensity - min_val) / (max_val - min_val)
    
    def detect_anomaly(
        self,
        observed_wavelengths: np.ndarray,
        observed_intensity: np.ndarray,
        expected_material: str
    ) -> Tuple[bool, float, str]:
        """
        Detect spectral anomaly compared to expected material
        
        Returns:
            (is_anomaly, anomaly_score, description)
        """
        expected = self.library.get_material(expected_material)
        if expected is None:
            return False, 0.0, "Unknown expected material"
        
        # Interpolate expected to observed wavelengths
        expected_interp = np.interp(
            observed_wavelengths,
            expected.wavelengths,
            expected.intensity,
            left=0, right=0
        )
        
        # Calculate residual
        residual = np.abs(observed_intensity - expected_interp)
        anomaly_score = np.mean(residual) / np.std(expected_interp + 1e-10)
        
        # Threshold for anomaly detection
        is_anomaly = anomaly_score > 2.0
        
        if is_anomaly:
            description = f"Spectral anomaly detected (score: {anomaly_score:.2f})"
        else:
            description = f"Spectrum matches expected material (score: {anomaly_score:.2f})"
        
        return is_anomaly, anomaly_score, description
    
    def assess_threat_level(
        self,
        material_id: str,
        maneuvering: bool = False,
        proximity_km: float = 1000.0
    ) -> Tuple[int, str]:
        """
        Assess threat level based on material and behavior
        
        Returns:
            (threat_level_0_10, threat_description)
        """
        material = self.library.get_material(material_id)
        base_threat = material.threat_level if material else 5
        
        # Proximity factor
        if proximity_km < 10:
            proximity_factor = 3
        elif proximity_km < 100:
            proximity_factor = 2
        elif proximity_km < 500:
            proximity_factor = 1
        else:
            proximity_factor = 0
        
        # Maneuvering factor
        maneuver_factor = 2 if maneuvering else 0
        
        # Calculate total threat
        total_threat = min(10, base_threat + proximity_factor + maneuver_factor)
        
        # Generate description
        if total_threat >= 8:
            description = "CRITICAL: High-threat material in close proximity"
        elif total_threat >= 6:
            description = "HIGH: Suspicious material/behavior detected"
        elif total_threat >= 4:
            description = "ELEVATED: Monitor closely"
        else:
            description = "LOW: Standard satellite materials"
        
        return total_threat, description


# Convenience function for integration
def analyze_spectrum(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    expected_material: Optional[str] = None
) -> dict:
    """
    Main entry point for spectral analysis
    
    Returns analysis results as dictionary for Go bridge
    """
    analyzer = SpectralAnalyzer()
    
    # Material identification
    material_id, confidence, top_matches = analyzer.identify_material(
        wavelengths, intensity
    )
    
    result = {
        "material_id": material_id,
        "confidence": float(confidence),
        "top_matches": [(m, float(s)) for m, s in top_matches],
        "tactical_significance": analyzer.library.get_material(material_id).tactical_significance
        if analyzer.library.get_material(material_id) else "Unknown"
    }
    
    # Anomaly detection if expected material provided
    if expected_material:
        is_anomaly, anomaly_score, description = analyzer.detect_anomaly(
            wavelengths, intensity, expected_material
        )
        result["anomaly_detected"] = is_anomaly
        result["anomaly_score"] = float(anomaly_score)
        result["anomaly_description"] = description
    
    return result
