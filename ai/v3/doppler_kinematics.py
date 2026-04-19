"""
Doppler-Shift Kinematics for Hypersonic Threat Detection
Calculates radial velocity from spectral shift without waiting for TLE updates
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
from scipy.signal import correlate, find_peaks
from scipy.optimize import minimize
import logging

from mpdaf.obj import Spectrum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VelocityMeasurement:
    """Radial velocity measurement from Doppler analysis"""
    radial_velocity_kms: float  # km/s (positive = receding, negative = approaching)
    confidence: float           # 0-1 confidence score
    wavelength_shift_nm: float  # Measured shift in nanometers
    rest_wavelength_nm: float   # Reference wavelength
    doppler_factor: float       # 1 + z (redshift)
    spectrum_correlation: float # Template match quality
    method: str                 # "CROSS_CORRELATION" or "PEAK_FITTING"
    material_match: str         # Best matching material from library
    timestamp: str


class DopplerAnalyzer:
    """
    Analyzes Doppler-shifted spectra to determine radial velocity
    
    Uses cross-correlation or peak-fitting to measure wavelength shifts
    and derive line-of-sight velocity without orbital element updates.
    """
    
    def __init__(
        self,
        speed_of_light_kms: float = 299792.458,  # km/s
        wavelength_precision_nm: float = 0.001  # 0.001 nm = 1 pm precision
    ):
        self.c = speed_of_light_kms
        self.wavelength_precision = wavelength_precision_nm
        self.material_library = self._load_material_library()
        
    def _load_material_library(self) -> Dict[str, np.ndarray]:
        """
        Load rest-frame spectral templates for common space materials
        
        These are laboratory-measured spectra at zero velocity
        """
        library = {}
        
        # Carbon-Carbon composite (hypersonic nose cones, re-entry vehicles)
        # Characteristic absorption features at 380nm (C2 Swan bands) and 589nm (Na D lines)
        wavelengths = np.linspace(300, 900, 600)  # 1 nm resolution
        
        # C-C composite template
        cc_spectrum = np.ones_like(wavelengths)
        cc_spectrum[np.abs(wavelengths - 380) < 10] *= 0.7  # C2 absorption
        cc_spectrum[np.abs(wavelengths - 589) < 5] *= 0.85  # Sodium D lines
        cc_spectrum[np.abs(wavelengths - 656) < 3] *= 0.9   # H-alpha (if heated)
        library['CARBON_CARBON'] = np.column_stack([wavelengths, cc_spectrum])
        
        # Aluminum (rocket bodies, debris)
        al_spectrum = np.ones_like(wavelengths) * 0.9
        al_spectrum[np.abs(wavelengths - 396) < 2] *= 0.5  # Al I absorption
        library['ALUMINUM'] = np.column_stack([wavelengths, al_spectrum])
        
        # Titanium (SR-71-style, some satellite structures)
        ti_spectrum = np.ones_like(wavelengths) * 0.85
        ti_spectrum[np.abs(wavelengths - 500) < 20] *= 0.9  # Broad TiO bands
        library['TITANIUM'] = np.column_stack([wavelengths, ti_spectrum])
        
        # Silicon (solar panels, electronics)
        si_spectrum = np.ones_like(wavelengths) * 0.8
        si_spectrum[np.abs(wavelengths - 900) < 50] *= 0.6  # Silicon band edge
        library['SILICON'] = np.column_stack([wavelengths, si_spectrum])
        
        # Kapton (thermal blankets)
        kapton_spectrum = np.ones_like(wavelengths) * 0.75
        # Kapton has relatively flat spectrum with slight NIR absorption
        kapton_spectrum[wavelengths > 700] *= 0.9
        library['KAPTON'] = np.column_stack([wavelengths, kapton_spectrum])
        
        return library
    
    def measure_velocity(
        self,
        observed_spectrum: Spectrum,
        wavelength_array_nm: np.ndarray,
        initial_guess_kms: float = 0.0,
        material_hints: Optional[list] = None
    ) -> VelocityMeasurement:
        """
        Measure radial velocity from Doppler-shifted spectrum
        
        Args:
            observed_spectrum: MPDAF Spectrum object with target data
            wavelength_array_nm: Wavelength calibration (pixel -> nm)
            initial_guess_kms: Expected velocity range (for search window)
            material_hints: List of suspected materials (narrow search)
            
        Returns:
            VelocityMeasurement with radial velocity and confidence
        """
        observed_data = observed_spectrum.data
        
        # Ensure same wavelength grid
        if len(wavelength_array_nm) != len(observed_data):
            logger.error("Wavelength calibration mismatch")
            return None
        
        # Find best matching template
        best_match, correlation_score = self._find_best_template(
            wavelength_array_nm,
            observed_data,
            material_hints
        )
        
        if best_match is None:
            logger.warning("No spectral template match found")
            return None
        
        # Method 1: Cross-correlation for velocity
        velocity_cc, shift_cc = self._cross_correlation_method(
            wavelength_array_nm,
            observed_data,
            best_match
        )
        
        # Method 2: Peak fitting for validation
        velocity_peak, shift_peak = self._peak_fitting_method(
            wavelength_array_nm,
            observed_data,
            best_match
        )
        
        # Compare methods for confidence
        velocity_agreement = abs(velocity_cc - velocity_peak) / (abs(velocity_cc) + 1.0)
        
        if velocity_agreement < 0.2:  # Within 20%
            # Methods agree - high confidence
            velocity = (velocity_cc + velocity_peak) / 2.0
            method = "CROSS_CORRELATION"
            wavelength_shift = (shift_cc + shift_peak) / 2.0
            confidence = min(correlation_score * (1.0 - velocity_agreement), 0.99)
        else:
            # Methods disagree - use cross-correlation (more robust) but lower confidence
            velocity = velocity_cc
            method = "CROSS_CORRELATION"
            wavelength_shift = shift_cc
            confidence = correlation_score * 0.5  # Reduced confidence
        
        # Calculate Doppler factor
        doppler_factor = 1.0 + (velocity / self.c)
        
        # Identify material from best match
        material_name = "UNKNOWN"
        for name, template in self.material_library.items():
            if np.array_equal(best_match[:, 1], template[:, 1]):
                material_name = name
                break
        
        return VelocityMeasurement(
            radial_velocity_kms=velocity,
            confidence=confidence,
            wavelength_shift_nm=wavelength_shift,
            rest_wavelength_nm=np.mean(wavelength_array_nm),
            doppler_factor=doppler_factor,
            spectrum_correlation=correlation_score,
            method=method,
            material_match=material_name,
            timestamp=datetime.now().isoformat()
        )
    
    def _find_best_template(
        self,
        wavelengths: np.ndarray,
        observed: np.ndarray,
        material_hints: Optional[list]
    ) -> Tuple[Optional[np.ndarray], float]:
        """Find best matching spectral template from library"""
        
        best_score = -1.0
        best_template = None
        
        candidates = material_hints if material_hints else self.material_library.keys()
        
        for material in candidates:
            if material not in self.material_library:
                continue
                
            template = self.material_library[material]
            
            # Interpolate template to observed wavelength grid
            template_interp = np.interp(
                wavelengths,
                template[:, 0],
                template[:, 1],
                left=1.0,
                right=1.0
            )
            
            # Calculate correlation
            score = np.corrcoef(observed, template_interp)[0, 1]
            
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template, best_score
    
    def _cross_correlation_method(
        self,
        wavelengths: np.ndarray,
        observed: np.ndarray,
        template: np.ndarray,
        velocity_range_kms: Tuple[float, float] = (-10.0, 10.0)
    ) -> Tuple[float, float]:
        """
        Measure velocity using cross-correlation
        
        Returns:
            (velocity_kms, wavelength_shift_nm)
        """
        # Interpolate template
        template_interp = np.interp(
            wavelengths,
            template[:, 0],
            template[:, 1],
            left=1.0,
            right=1.0
        )
        
        # Search velocity range
        best_correlation = -1.0
        best_velocity = 0.0
        best_shift = 0.0
        
        # Velocity steps (coarse then fine)
        velocities = np.linspace(velocity_range_kms[0], velocity_range_kms[1], 200)
        
        for v in velocities:
            # Doppler shift: observed = rest * (1 + v/c)
            shift_factor = 1.0 + v / self.c
            shifted_wavelengths = wavelengths / shift_factor
            
            # Interpolate observed to rest frame
            shifted_spectrum = np.interp(
                template[:, 0],
                shifted_wavelengths,
                observed,
                left=1.0,
                right=1.0
            )
            
            # Cross-correlation
            correlation = np.corrcoef(shifted_spectrum, template_interp)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_velocity = v
                best_shift = np.mean(wavelengths) * (shift_factor - 1.0)
        
        # Fine search around best velocity
        fine_velocities = np.linspace(best_velocity - 0.5, best_velocity + 0.5, 50)
        for v in fine_velocities:
            shift_factor = 1.0 + v / self.c
            shifted_wavelengths = wavelengths / shift_factor
            
            shifted_spectrum = np.interp(
                template[:, 0],
                shifted_wavelengths,
                observed,
                left=1.0,
                right=1.0
            )
            
            correlation = np.corrcoef(shifted_spectrum, template_interp)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_velocity = v
                best_shift = np.mean(wavelengths) * (shift_factor - 1.0)
        
        return best_velocity, best_shift
    
    def _peak_fitting_method(
        self,
        wavelengths: np.ndarray,
        observed: np.ndarray,
        template: np.ndarray,
        velocity_range_kms: Tuple[float, float] = (-10.0, 10.0)
    ) -> Tuple[float, float]:
        """
        Measure velocity by fitting peaks in spectrum
        
        Uses absorption/emission features as reference points
        """
        # Find peaks in template (reference features)
        template_interp = np.interp(wavelengths, template[:, 0], template[:, 1])
        
        # Find minima (absorption features)
        template_minima, _ = find_peaks(-template_interp, distance=10, prominence=0.1)
        observed_minima, _ = find_peaks(-observed, distance=10, prominence=0.1)
        
        if len(template_minima) < 2 or len(observed_minima) < 2:
            logger.debug("Insufficient spectral features for peak fitting")
            return 0.0, 0.0
        
        # Match peaks
        template_peaks_nm = wavelengths[template_minima]
        observed_peaks_nm = wavelengths[observed_minima]
        
        # Calculate average shift
        if len(template_peaks_nm) > 0 and len(observed_peaks_nm) > 0:
            # Use closest matching pairs
            shifts = []
            for obs_peak in observed_peaks_nm[:min(5, len(observed_peaks_nm))]:
                # Find nearest template peak
                distances = np.abs(template_peaks_nm - obs_peak)
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < 50:  # Within 50 nm
                    shift = obs_peak - template_peaks_nm[nearest_idx]
                    shifts.append(shift)
            
            if len(shifts) > 0:
                mean_shift = np.mean(shifts)
                
                # Convert to velocity: v = c * (delta_lambda / lambda)
                mean_wavelength = np.mean(wavelengths)
                velocity = self.c * (mean_shift / mean_wavelength)
                
                return velocity, mean_shift
        
        return 0.0, 0.0
    
    def estimate_trajectory_from_doppler_series(
        self,
        measurements: list,
        timestamps: list,
        observation_geometry: Dict
    ) -> Dict:
        """
        Estimate orbital trajectory from series of Doppler measurements
        
        Args:
            measurements: List of VelocityMeasurement
            timestamps: Observation times
            observation_geometry: Sensor position, look angles, etc.
            
        Returns:
            Trajectory estimate (position, velocity vector)
        """
        if len(measurements) < 3:
            logger.warning("Need at least 3 measurements for trajectory estimation")
            return None
        
        # Extract radial velocities
        radial_velocities = np.array([m.radial_velocity_kms for m in measurements])
        
        # Fit polynomial to velocity trend
        # For hypersonic/sub-orbital: velocity changes rapidly
        # For orbital: relatively constant or sinusoidal
        
        times_hours = np.array([(t - timestamps[0]).total_seconds() / 3600.0 
                                for t in timestamps])
        
        # Check for orbital vs. sub-orbital
        velocity_variance = np.var(radial_velocities)
        
        if velocity_variance < 1.0:  # Low variance suggests orbital
            # Fit sinusoidal (orbital motion)
            trajectory_type = "ORBITAL"
            
            # Simple linear trend + sinusoidal
            from scipy.optimize import curve_fit
            
            def orbital_velocity(t, v0, amp, period, phase):
                return v0 + amp * np.sin(2 * np.pi * t / period + phase)
            
            try:
                popt, _ = curve_fit(orbital_velocity, times_hours, radial_velocities,
                                   p0=[np.mean(radial_velocities), 1.0, 1.5, 0.0])
                
                predicted_velocities = orbital_velocity(times_hours, *popt)
                rms_error = np.sqrt(np.mean((radial_velocities - predicted_velocities)**2))
                
                return {
                    'trajectory_type': trajectory_type,
                    'mean_velocity': popt[0],
                    'velocity_amplitude': popt[1],
                    'orbital_period_hours': popt[2],
                    'rms_fit_error': rms_error,
                    'is_suborbital': False,
                    'confidence': 1.0 - min(rms_error / 10.0, 1.0)
                }
            except:
                return None
                
        else:  # High variance suggests sub-orbital/hypersonic
            trajectory_type = "SUBORBITAL"
            
            # Fit polynomial (accelerating/decelerating)
            coeffs = np.polyfit(times_hours, radial_velocities, 2)
            predicted = np.polyval(coeffs, times_hours)
            rms_error = np.sqrt(np.mean((radial_velocities - predicted)**2))
            
            # Acceleration from polynomial
            acceleration = coeffs[0] * 2  # km/s per hour
            
            return {
                'trajectory_type': trajectory_type,
                'initial_velocity': coeffs[2],
                'acceleration_kms_per_hour': acceleration,
                'polynomial_coeffs': coeffs.tolist(),
                'rms_fit_error': rms_error,
                'is_suborbital': True,
                'is_accelerating': acceleration > 0,
                'confidence': 1.0 - min(rms_error / 20.0, 1.0)
            }


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    # Create synthetic Doppler-shifted spectrum
    wavelengths = np.linspace(400, 800, 400)  # 400-800nm, 1nm resolution
    
    # Rest-frame C-C composite spectrum
    rest_spectrum = np.ones_like(wavelengths)
    rest_spectrum[np.abs(wavelengths - 589) < 5] *= 0.7  # Na D lines
    
    # Apply Doppler shift: v = +2.5 km/s (receding)
    velocity_true = 2.5  # km/s
    doppler_factor = 1.0 + velocity_true / 299792.458
    shifted_wavelengths = wavelengths * doppler_factor
    
    # Interpolate back to original grid
    observed_spectrum = np.interp(wavelengths, shifted_wavelengths, rest_spectrum)
    
    # Add noise
    observed_spectrum += np.random.normal(0, 0.02, observed_spectrum.shape)
    
    # Create MPDAF Spectrum
    spec = Spectrum(data=observed_spectrum, wave=wavelengths)
    
    # Analyze
    analyzer = DopplerAnalyzer()
    result = analyzer.measure_velocity(
        spec,
        wavelengths,
        material_hints=['CARBON_CARBON']
    )
    
    if result:
        print(f"Doppler Analysis Results:")
        print(f"  Radial velocity: {result.radial_velocity_kms:.2f} km/s")
        print(f"  True velocity: {velocity_true:.2f} km/s")
        print(f"  Error: {abs(result.radial_velocity_kms - velocity_true):.3f} km/s")
        print(f"  Wavelength shift: {result.wavelength_shift_nm:.4f} nm")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Material match: {result.material_match}")
