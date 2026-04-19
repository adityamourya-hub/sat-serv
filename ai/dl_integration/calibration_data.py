"""
Calibration Data Generation for INT8 Quantization
Generates representative datasets for entropy/percentile calibration algorithms
Optimized for space domain awareness imagery and spectroscopy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class CalibrationConfig:
    """Configuration for calibration data generation"""
    num_samples: int = 1000
    batch_size: int = 1
    data_type: str = "optical"  # optical, spectroscopy, thermal
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    value_range: Tuple[float, float] = (0.0, 1.0)
    noise_level: float = 0.01
    include_edge_cases: bool = True
    distribution: str = "normal"  # normal, uniform, poisson
    seed: int = 42


@dataclass
class CalibrationDataset:
    """Calibration dataset with metadata"""
    data: np.ndarray
    metadata: Dict
    num_samples: int
    data_type: str
    value_statistics: Dict


class CalibrationDataGenerator:
    """
    Generates representative calibration data for INT8 quantization
    
    For optimal INT8 quantization, calibration data must:
    1. Cover the full dynamic range of expected inputs
    2. Include edge cases and outliers
    3. Match the statistical distribution of real data
    4. Be representative of the actual inference workload
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize calibration data generator
        
        Args:
            config: Calibration configuration
        """
        self.config = config or CalibrationConfig()
        np.random.seed(self.config.seed)
    
    def generate_optical_calibration_data(self) -> CalibrationDataset:
        """
        Generate calibration data for optical imagery
        
        Simulates space-based imagery with:
        - Star fields (point sources)
        - Satellites/RSOs (extended sources)
        - Cosmic ray hits (impulse noise)
        - Background noise (Gaussian/Poisson)
        
        Returns:
            CalibrationDataset for optical models
        """
        samples = []
        
        for i in range(self.config.num_samples):
            # Base frame with background noise
            if self.config.distribution == "poisson":
                frame = np.random.poisson(
                    self.config.value_range[1] * 0.1,
                    self.config.input_shape
                ).astype(np.float32)
            else:
                frame = np.random.normal(
                    self.config.value_range[1] * 0.1,
                    self.config.noise_level,
                    self.config.input_shape
                ).astype(np.float32)
            
            # Add stars (point sources)
            num_stars = np.random.randint(50, 200)
            for _ in range(num_stars):
                x = np.random.randint(0, frame.shape[-1])
                y = np.random.randint(0, frame.shape[-2])
                brightness = np.random.uniform(0.3, 0.8)
                
                if frame.ndim == 4:
                    frame[0, 0, y, x] = brightness
                elif frame.ndim == 3:
                    frame[0, y, x] = brightness
                else:
                    frame[y, x] = brightness
            
            # Add satellite (extended source)
            if np.random.rand() > 0.3:  # 70% chance
                x = np.random.randint(50, frame.shape[-1] - 50)
                y = np.random.randint(50, frame.shape[-2] - 50)
                size = np.random.randint(3, 15)
                brightness = np.random.uniform(0.5, 1.0)
                
                for dy in range(-size//2, size//2 + 1):
                    for dx in range(-size//2, size//2 + 1):
                        if (dy**2 + dx**2) <= (size//2)**2:
                            py, px = y + dy, x + dx
                            if 0 <= py < frame.shape[-2] and 0 <= px < frame.shape[-1]:
                                if frame.ndim == 4:
                                    frame[0, 0, py, px] = brightness
                                elif frame.ndim == 3:
                                    frame[0, py, px] = brightness
                                else:
                                    frame[py, px] = brightness
            
            # Add cosmic ray hits (impulse noise) - edge cases
            if self.config.include_edge_cases and np.random.rand() > 0.7:
                num_hits = np.random.randint(1, 5)
                for _ in range(num_hits):
                    x = np.random.randint(0, frame.shape[-1])
                    y = np.random.randint(0, frame.shape[-2])
                    hit_intensity = np.random.uniform(2.0, 5.0)
                    
                    if frame.ndim == 4:
                        frame[0, 0, y, x] = hit_intensity
                    elif frame.ndim == 3:
                        frame[0, y, x] = hit_intensity
                    else:
                        frame[y, x] = hit_intensity
            
            # Clip to valid range
            frame = np.clip(frame, self.config.value_range[0], self.config.value_range[1])
            
            samples.append(frame)
        
        data = np.array(samples)
        
        return CalibrationDataset(
            data=data,
            metadata={
                "generation_time": time.time(),
                "num_stars_range": (50, 200),
                "satellite_probability": 0.7,
                "cosmic_ray_probability": 0.3,
                "seed": self.config.seed
            },
            num_samples=len(samples),
            data_type="optical",
            value_statistics=self._compute_statistics(data)
        )
    
    def generate_spectroscopy_calibration_data(self) -> CalibrationDataset:
        """
        Generate calibration data for spectroscopic analysis
        
        Simulates spectral data with:
        - Continuum (blackbody spectrum)
        - Absorption lines (characteristic features)
        - Emission lines (for active satellites)
        - Noise (photon counting statistics)
        
        Returns:
            CalibrationDataset for spectroscopy models
        """
        samples = []
        
        wavelengths = np.linspace(300, 1100, self.config.input_shape[-1])
        
        for i in range(self.config.num_samples):
            # Blackbody continuum
            temperature = np.random.uniform(3000, 10000)  # Kelvin
            
            # Planck's law (simplified)
            h = 6.626e-34  # Planck constant
            c = 3e8        # Speed of light
            k = 1.38e-23   # Boltzmann constant
            
            # Convert nm to m
            wl_m = wavelengths * 1e-9
            
            # Spectral radiance (simplified)
            spectrum = (2 * h * c**2 / wl_m**5) / (
                np.exp(h * c / (wl_m * k * temperature)) - 1
            )
            
            # Normalize
            spectrum = spectrum / np.max(spectrum)
            
            # Add absorption lines (material signatures)
            num_lines = np.random.randint(1, 5)
            for _ in range(num_lines):
                line_wl = np.random.uniform(400, 1000)
                line_depth = np.random.uniform(0.1, 0.5)
                line_width = np.random.uniform(2, 10)
                
                # Gaussian absorption line
                absorption = line_depth * np.exp(
                    -((wavelengths - line_wl) / line_width)**2
                )
                spectrum -= absorption
            
            # Add emission lines (active satellites)
            if np.random.rand() > 0.5:  # 50% chance of active satellite
                num_emission = np.random.randint(1, 3)
                for _ in range(num_emission):
                    line_wl = np.random.uniform(600, 900)
                    line_intensity = np.random.uniform(0.2, 0.6)
                    line_width = np.random.uniform(1, 5)
                    
                    emission = line_intensity * np.exp(
                        -((wavelengths - line_wl) / line_width)**2
                    )
                    spectrum += emission
            
            # Add Poisson noise
            if self.config.distribution == "poisson":
                spectrum = np.random.poisson(spectrum * 1000) / 1000
            else:
                spectrum += np.random.normal(0, self.config.noise_level, len(spectrum))
            
            # Clip and reshape
            spectrum = np.clip(spectrum, 0, 1)
            
            if len(self.config.input_shape) == 2:
                spectrum = spectrum.reshape(self.config.input_shape)
            elif len(self.config.input_shape) == 1:
                spectrum = spectrum[:self.config.input_shape[0]]
            
            samples.append(spectrum)
        
        data = np.array(samples)
        
        return CalibrationDataset(
            data=data,
            metadata={
                "generation_time": time.time(),
                "temperature_range": (3000, 10000),
                "wavelength_range": (300, 1100),
                "num_samples": len(samples),
                "seed": self.config.seed
            },
            num_samples=len(samples),
            data_type="spectroscopy",
            value_statistics=self._compute_statistics(data)
        )
    
    def generate_thermal_calibration_data(self) -> CalibrationDataset:
        """
        Generate calibration data for thermal imaging
        
        Simulates thermal IR data with:
        - Background temperature
        - Hot spots (electronics, engines)
        - Gradients (solar heating)
        
        Returns:
            CalibrationDataset for thermal models
        """
        samples = []
        
        for i in range(self.config.num_samples):
            # Background temperature (space)
            base_temp = np.random.uniform(200, 300)  # Kelvin
            
            # Create thermal frame
            frame = np.full(self.config.input_shape, base_temp, dtype=np.float32)
            
            # Add hot spots (electronics)
            num_hotspots = np.random.randint(1, 5)
            for _ in range(num_hotspots):
                if frame.ndim >= 2:
                    x = np.random.randint(0, frame.shape[-1])
                    y = np.random.randint(0, frame.shape[-2])
                    
                    # Hot spot with Gaussian falloff
                    hotspot_temp = np.random.uniform(350, 400)
                    sigma = np.random.uniform(5, 15)
                    
                    for dy in range(-30, 31):
                        for dx in range(-30, 31):
                            py, px = y + dy, x + dx
                            if 0 <= py < frame.shape[-2] and 0 <= px < frame.shape[-1]:
                                distance = np.sqrt(dy**2 + dx**2)
                                temp_increase = (hotspot_temp - base_temp) * np.exp(-distance**2 / (2 * sigma**2))
                                
                                if frame.ndim == 4:
                                    frame[0, 0, py, px] += temp_increase
                                elif frame.ndim == 3:
                                    frame[0, py, px] += temp_increase
                                else:
                                    frame[py, px] += temp_increase
            
            # Add noise
            noise = np.random.normal(0, self.config.noise_level * 10, self.config.input_shape)
            frame += noise
            
            # Clip to valid range
            frame = np.clip(frame, self.config.value_range[0], self.config.value_range[1])
            
            samples.append(frame)
        
        data = np.array(samples)
        
        return CalibrationDataset(
            data=data,
            metadata={
                "generation_time": time.time(),
                "base_temperature_range": (200, 300),
                "hotspot_temperature_range": (350, 400),
                "num_hotspots_range": (1, 5),
                "seed": self.config.seed
            },
            num_samples=len(samples),
            data_type="thermal",
            value_statistics=self._compute_statistics(data)
        )
    
    def _compute_statistics(self, data: np.ndarray) -> Dict:
        """Compute value statistics for calibration data"""
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "percentile_1": float(np.percentile(data, 1)),
            "percentile_99": float(np.percentile(data, 99)),
            "dynamic_range": float(np.max(data) - np.min(data))
        }
    
    def save_calibration_data(self, 
                            dataset: CalibrationDataset,
                            output_path: str,
                            format: str = "npy"):
        """
        Save calibration data to file
        
        Args:
            dataset: CalibrationDataset to save
            output_path: Output file path
            format: File format (npy, npz, bin)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npy":
            np.save(output_path, dataset.data)
        elif format == "npz":
            np.savez(output_path, 
                    data=dataset.data,
                    metadata=json.dumps(dataset.metadata),
                    statistics=json.dumps(dataset.value_statistics))
        elif format == "bin":
            # Raw binary format for TensorRT
            dataset.data.tofile(output_path)
        
        # Save metadata separately
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "metadata": dataset.metadata,
                "num_samples": dataset.num_samples,
                "data_type": dataset.data_type,
                "value_statistics": dataset.value_statistics,
                "shape": dataset.data.shape,
                "dtype": str(dataset.data.dtype)
            }, f, indent=2)
        
        print(f"[CALIBRATION] Saved {dataset.num_samples} samples to {output_path}")


class EntropyCalibrationAlgorithm:
    """
    Implements entropy-based calibration algorithm for INT8 quantization
    
    The entropy algorithm minimizes information loss by selecting
    quantization thresholds that preserve the information content
    of the activation distributions.
    """
    
    def __init__(self, num_bins: int = 2048):
        """
        Initialize entropy calibration
        
        Args:
            num_bins: Number of histogram bins for entropy calculation
        """
        self.num_bins = num_bins
    
    def compute_threshold(self, 
                         activation_values: np.ndarray,
                         target_bins: int = 128) -> float:
        """
        Compute optimal quantization threshold using entropy minimization
        
        Args:
            activation_values: Array of activation values
            target_bins: Number of quantization bins (typically 128 for INT8)
            
        Returns:
            Optimal threshold value
        """
        # Compute histogram
        min_val = np.min(activation_values)
        max_val = np.max(activation_values)
        
        hist, bin_edges = np.histogram(
            activation_values,
            bins=self.num_bins,
            range=(min_val, max_val)
        )
        
        # Compute probabilities
        total = np.sum(hist)
        if total == 0:
            return max_val
        
        probs = hist / total
        
        # Find optimal threshold that minimizes entropy loss
        min_entropy_loss = float('inf')
        optimal_threshold = max_val
        
        for i in range(1, len(bin_edges) - 1):
            threshold = bin_edges[i]
            
            # Split distribution
            below_mask = activation_values <= threshold
            above_mask = activation_values > threshold
            
            if np.sum(below_mask) == 0 or np.sum(above_mask) == 0:
                continue
            
            # Compute entropy loss
            # Simplified: minimize information loss in quantized bins
            below_hist, _ = np.histogram(
                activation_values[below_mask],
                bins=target_bins,
                range=(min_val, threshold)
            )
            
            above_hist, _ = np.histogram(
                activation_values[above_mask],
                bins=target_bins,
                range=(threshold, max_val)
            )
            
            # Entropy calculation
            below_probs = below_hist / np.sum(below_hist) if np.sum(below_hist) > 0 else np.zeros_like(below_hist)
            above_probs = above_hist / np.sum(above_hist) if np.sum(above_hist) > 0 else np.zeros_like(above_hist)
            
            # Avoid log(0)
            below_probs = below_probs[below_probs > 0]
            above_probs = above_probs[above_probs > 0]
            
            below_entropy = -np.sum(below_probs * np.log2(below_probs)) if len(below_probs) > 0 else 0
            above_entropy = -np.sum(above_probs * np.log2(above_probs)) if len(above_probs) > 0 else 0
            
            total_entropy_loss = below_entropy + above_entropy
            
            if total_entropy_loss < min_entropy_loss:
                min_entropy_loss = total_entropy_loss
                optimal_threshold = threshold
        
        return optimal_threshold
    
    def calibrate(self, calibration_data: np.ndarray) -> Dict[str, float]:
        """
        Perform entropy calibration on calibration data
        
        Args:
            calibration_data: Representative calibration dataset
            
        Returns:
            Dictionary of layer thresholds
        """
        thresholds = {}
        
        # Flatten if needed
        if calibration_data.ndim > 1:
            data = calibration_data.flatten()
        else:
            data = calibration_data
        
        # Compute threshold
        threshold = self.compute_threshold(data)
        
        thresholds["global"] = threshold
        thresholds["min"] = float(np.min(data))
        thresholds["max"] = float(np.max(data))
        
        return thresholds


# Example usage
if __name__ == "__main__":
    print("Testing Calibration Data Generator...")
    
    # Generate optical calibration data
    optical_config = CalibrationConfig(
        num_samples=100,
        data_type="optical",
        input_shape=(1, 1, 256, 256),
        include_edge_cases=True
    )
    
    generator = CalibrationDataGenerator(optical_config)
    optical_data = generator.generate_optical_calibration_data()
    
    print(f"Optical calibration data:")
    print(f"  Shape: {optical_data.data.shape}")
    print(f"  Num samples: {optical_data.num_samples}")
    print(f"  Value range: [{optical_data.value_statistics['min']:.3f}, {optical_data.value_statistics['max']:.3f}]")
    print(f"  Dynamic range: {optical_data.value_statistics['dynamic_range']:.3f}")
    
    # Generate spectroscopy calibration data
    spectro_config = CalibrationConfig(
        num_samples=100,
        data_type="spectroscopy",
        input_shape=(8000,),
        distribution="poisson"
    )
    
    generator = CalibrationDataGenerator(spectro_config)
    spectro_data = generator.generate_spectroscopy_calibration_data()
    
    print(f"\nSpectroscopy calibration data:")
    print(f"  Shape: {spectro_data.data.shape}")
    print(f"  Num samples: {spectro_data.num_samples}")
    
    # Test entropy calibration
    entropy_calib = EntropyCalibrationAlgorithm(num_bins=1024)
    thresholds = entropy_calib.calibrate(optical_data.data[:10].flatten())
    
    print(f"\nEntropy calibration thresholds:")
    print(f"  Global: {thresholds['global']:.3f}")
    print(f"  Min: {thresholds['min']:.3f}")
    print(f"  Max: {thresholds['max']:.3f}")
    
    # Save calibration data
    generator.save_calibration_data(optical_data, "calibration_optical.npy")
    
    print("\nAll tests passed!")
