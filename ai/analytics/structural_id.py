"""
Multi-Gaussian Expansion (MGE) for Satellite Structural Characterization
Implements shape decomposition for payload discrimination and architectural fingerprinting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# MGE Component thresholds for satellite classification
SIGMA_THRESHOLD = 2.5  # Pixels - components smaller than this are "Bus"
Q_THRESHOLD = 0.7       # Circularity - components more circular than this are "Bus"


@dataclass
class MGEComponent:
    """Represents a single Gaussian component in MGE decomposition"""
    height: float      # Intensity
    sigma: float       # Standard deviation (size)
    q: float          # Axial ratio (1.0 = circular, <1.0 = elongated)
    flux: float       # Calculated flux for this component
    component_type: str  # "BUS" or "ARRAY"


@dataclass
class MGEResult:
    """Result of MGE fitting and classification"""
    array_to_bus_ratio: float
    total_components: int
    bus_components: int
    array_components: int
    architecture_type: str
    components: List[MGEComponent]
    status: str


class MGEAnalyzer:
    """
    Multi-Gaussian Expansion analyzer for satellite structural characterization
    Decomposes satellite images into Gaussian components for architectural fingerprinting
    """
    
    def __init__(self, sigma_threshold: float = SIGMA_THRESHOLD, 
                 q_threshold: float = Q_THRESHOLD):
        self.sigma_threshold = sigma_threshold
        self.q_threshold = q_threshold
        
    def fit_mge(self, image: np.ndarray, ngauss: int = 8) -> List[MGEComponent]:
        """
        Perform Multi-Gaussian Expansion fitting on satellite image
        
        Args:
            image: 2D numpy array representing satellite image
            ngauss: Number of Gaussian components to fit
            
        Returns:
            List of MGEComponent objects
        """
        # Simplified MGE fitting (in production, use mgefit.mge_fit_sectors)
        # This is a placeholder that demonstrates the logic
        
        components = []
        
        # Find peaks in the image using center-of-mass for initial guess
        y_indices, x_indices = np.indices(image.shape)
        
        # Simple peak detection for initial component positions
        # In production, use proper MGE fitting with sectors
        total_flux = np.sum(image)
        
        if total_flux == 0:
            return components
        
        # Weighted center calculation
        center_y = np.sum(y_indices * image) / total_flux
        center_x = np.sum(x_indices * image) / total_flux
        
        # Fit multiple Gaussians with decreasing weights
        # This is a simplified approximation of MGE fitting
        for i in range(ngauss):
            # Decreasing height for each component
            height = np.max(image) * (0.6 ** i)
            
            # Increasing sigma for outer components
            sigma = 1.0 + i * 0.8
            
            # Varying axial ratio (core is more circular)
            if i < 2:
                q = 0.9  # Nearly circular (bus)
            else:
                q = 0.6 + np.random.random() * 0.3  # Elongated (arrays)
            
            # Calculate flux for this component
            flux = 2 * np.pi * height * (sigma ** 2) * q
            
            # Classify component
            component_type = self._classify_component(sigma, q)
            
            component = MGEComponent(
                height=height,
                sigma=sigma,
                q=q,
                flux=flux,
                component_type=component_type
            )
            components.append(component)
        
        return components
    
    def _classify_component(self, sigma: float, q: float) -> str:
        """
        Classify Gaussian component as Bus or Array based on geometry
        
        Args:
            sigma: Standard deviation (size)
            q: Axial ratio (circularity)
            
        Returns:
            "BUS" or "ARRAY"
        """
        # Small and circular = Bus component
        if sigma < self.sigma_threshold and q > self.q_threshold:
            return "BUS"
        else:
            # Large or elongated = Array/Appendage
            return "ARRAY"
    
    def analyze_structure(self, image: np.ndarray, ngauss: int = 8) -> MGEResult:
        """
        Perform complete structural analysis including MGE fitting and classification
        
        Args:
            image: 2D numpy array representing satellite image
            ngauss: Number of Gaussian components to fit
            
        Returns:
            MGEResult with architectural fingerprinting
        """
        # Perform MGE fitting
        components = self.fit_mge(image, ngauss)
        
        if not components:
            return MGEResult(
                array_to_bus_ratio=0.0,
                total_components=0,
                bus_components=0,
                array_components=0,
                architecture_type="UNKNOWN",
                components=[],
                status="FAILED"
            )
        
        # Calculate flux ratios
        bus_flux = sum(c.flux for c in components if c.component_type == "BUS")
        array_flux = sum(c.flux for c in components if c.component_type == "ARRAY")
        
        # Calculate ratio
        if bus_flux == 0:
            ratio = float('inf') if array_flux > 0 else 0.0
        else:
            ratio = array_flux / bus_flux
        
        # Count components
        bus_count = sum(1 for c in components if c.component_type == "BUS")
        array_count = sum(1 for c in components if c.component_type == "ARRAY")
        
        # Determine architecture type
        architecture = self._determine_architecture(ratio)
        
        return MGEResult(
            array_to_bus_ratio=ratio,
            total_components=len(components),
            bus_components=bus_count,
            array_components=array_count,
            architecture_type=architecture,
            components=components,
            status="SUCCESS"
        )
    
    def _determine_architecture(self, ratio: float) -> str:
        """
        Determine satellite architecture type based on array-to-bus ratio
        
        Args:
            ratio: Array-to-bus flux ratio
            
        Returns:
            Architecture classification string
        """
        if ratio < 1.0:
            return "COMPACT_LOW_POWER"
        elif 1.0 <= ratio < 2.0:
            return "SMALL_3AXIS_BUS"
        elif 2.0 <= ratio < 5.0:
            return "STANDARD_3AXIS_BUS"
        elif 5.0 <= ratio < 8.0:
            return "LARGE_ARRAY_BUS"
        elif 8.0 <= ratio < 12.0:
            return "HIGH_POWER_EW_SAR"
        else:
            return "EXTREME_POWER_JAMMER"
    
    def get_array_to_bus_ratio(self, image: np.ndarray, ngauss: int = 8) -> Dict:
        """
        Simplified interface for getting just the ratio
        
        Args:
            image: 2D numpy array representing satellite image
            ngauss: Number of Gaussian components to fit
            
        Returns:
            Dictionary with ratio and metadata
        """
        result = self.analyze_structure(image, ngauss)
        
        return {
            "array_to_bus_ratio": round(result.array_to_bus_ratio, 3),
            "total_components": result.total_components,
            "bus_components": result.bus_components,
            "array_components": result.array_components,
            "architecture_type": result.architecture_type,
            "status": result.status
        }


def get_array_to_bus_ratio_from_plasma(object_id_str: str, 
                                        plasma_socket: str = "/tmp/plasma") -> Dict:
    """
    Retrieves image from Plasma Store and calculates the Area Ratio via MGE decomposition
    
    Args:
        object_id_str: Hex string of the Plasma ObjectID
        plasma_socket: Path to Plasma store socket
        
    Returns:
        Dictionary with MGE analysis results
    """
    try:
        # Import pyarrow.plasma
        import pyarrow.plasma as plasma
        
        # Connect to Plasma Store
        client = plasma.connect(plasma_socket)
        object_id = plasma.ObjectID(bytes.fromhex(object_id_str))
        
        # Retrieve image from shared memory (zero-copy)
        image = client.get(object_id)
        
        if image is None:
            return {
                "array_to_bus_ratio": 0.0,
                "total_components": 0,
                "status": "OBJECT_NOT_FOUND"
            }
        
        # Perform MGE analysis
        analyzer = MGEAnalyzer()
        result = analyzer.get_array_to_bus_ratio(image)
        
        # Release the object
        client.release(object_id)
        
        return result
        
    except ImportError:
        # Fallback if pyarrow not available
        return {
            "array_to_bus_ratio": 0.0,
            "total_components": 0,
            "status": "PYARROW_NOT_AVAILABLE"
        }
    except Exception as e:
        return {
            "array_to_bus_ratio": 0.0,
            "total_components": 0,
            "status": f"ERROR: {str(e)}"
        }


class StructuralFingerprintDB:
    """
    Database for storing and querying MGE-derived structural fingerprints
    """
    
    def __init__(self):
        self.fingerprints = {}
        
    def add_fingerprint(self, norad_id: int, result: MGEResult, 
                       timestamp: float = None) -> None:
        """
        Add a structural fingerprint to the database
        
        Args:
            norad_id: NORAD catalog ID
            result: MGE analysis result
            timestamp: Unix timestamp (default: now)
        """
        import time
        if timestamp is None:
            timestamp = time.time()
            
        if norad_id not in self.fingerprints:
            self.fingerprints[norad_id] = []
            
        self.fingerprints[norad_id].append({
            "timestamp": timestamp,
            "ratio": result.array_to_bus_ratio,
            "architecture": result.architecture_type,
            "components": result.total_components
        })
    
    def get_fingerprint(self, norad_id: int) -> Optional[Dict]:
        """Get latest fingerprint for a satellite"""
        if norad_id in self.fingerprints and self.fingerprints[norad_id]:
            return self.fingerprints[norad_id][-1]
        return None
    
    def detect_anomaly(self, norad_id: int, current_result: MGEResult,
                      threshold: float = 2.0) -> bool:
        """
        Detect structural anomaly by comparing to baseline
        
        Args:
            norad_id: NORAD catalog ID
            current_result: Current MGE analysis
            threshold: Ratio change threshold for anomaly
            
        Returns:
            True if anomaly detected
        """
        baseline = self.get_fingerprint(norad_id)
        if baseline is None:
            return False
            
        ratio_change = abs(current_result.array_to_bus_ratio - baseline["ratio"])
        return ratio_change > threshold


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic satellite image for testing
    size = 64
    image = np.zeros((size, size))
    
    # Add compact central bus
    y, x = np.indices((size, size))
    center_y, center_x = size // 2, size // 2
    
    # Bus component (compact, high intensity)
    bus_sigma = 2.0
    bus = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * bus_sigma**2))
    image += bus * 100
    
    # Array components (elongated, lower intensity)
    array_sigma_y = 1.5
    array_sigma_x = 8.0
    array_offset = 10
    
    # Left array
    array_left = np.exp(-((y - center_y)**2 / (2 * array_sigma_y**2) + 
                          (x - (center_x - array_offset))**2 / (2 * array_sigma_x**2)))
    image += array_left * 50
    
    # Right array
    array_right = np.exp(-((y - center_y)**2 / (2 * array_sigma_y**2) + 
                           (x - (center_x + array_offset))**2 / (2 * array_sigma_x**2)))
    image += array_right * 50
    
    # Analyze
    analyzer = MGEAnalyzer()
    result = analyzer.analyze_structure(image, ngauss=6)
    
    print("MGE Analysis Results:")
    print(f"Array-to-Bus Ratio: {result.array_to_bus_ratio:.3f}")
    print(f"Architecture Type: {result.architecture_type}")
    print(f"Total Components: {result.total_components}")
    print(f"Bus Components: {result.bus_components}")
    print(f"Array Components: {result.array_components}")
    
    for i, comp in enumerate(result.components):
        print(f"Component {i+1}: type={comp.component_type}, sigma={comp.sigma:.2f}, q={comp.q:.2f}")
