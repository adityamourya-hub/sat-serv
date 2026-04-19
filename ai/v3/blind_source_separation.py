"""
Blind Source Separation (BSS) for Rendezvous and Proximity Operations (RPO) Detection
Detects "Sleeper" satellites hiding in the glare of Indian assets using PCA/ICA
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Scikit-learn for BSS algorithms
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

# MPDAF for datacube handling
from mpdaf.obj import Cube, Spectrum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RPODetection:
    """Detected proximity operation / sleeper satellite"""
    host_norad_id: int           # Indian asset ID
    primary_period_hours: float  # Host satellite rotation period
    secondary_period_hours: float  # Sleeper satellite rotation period
    separation_km: float         # Estimated separation distance
    confidence: float          # Detection confidence (0-1)
    detection_time: datetime
    method: str                # "PCA" or "ICA"
    spectral_signatures: List[np.ndarray]  # Separated spectra


class BlindSourceSeparator:
    """
    Blind Source Separation for RPO detection
    
    Detects multiple objects in single pixel by separating mixed light curves
    using PCA/ICA on time-series spectral data.
    """
    
    def __init__(
        self,
        n_components: int = 2,        # Expected number of sources
        method: str = "ICA",          # "PCA" or "ICA"
        snr_threshold: float = 3.0,   # Minimum SNR for detection
        rotation_period_range: Tuple[float, float] = (0.1, 24.0)  # Hours
    ):
        self.n_components = n_components
        self.method = method.upper()
        self.snr_threshold = snr_threshold
        self.rotation_period_range = rotation_period_range
        
        logger.info(f"BSS initialized: method={method}, n_components={n_components}")
    
    def analyze_pixel(
        self,
        spectral_cube: Cube,
        x: int,
        y: int,
        norad_id: int,
        host_period_hours: Optional[float] = None
    ) -> Optional[RPODetection]:
        """
        Analyze single pixel for RPO using BSS
        
        Args:
            spectral_cube: 3D datacube [wavelength, y, x]
            x, y: Pixel coordinates
            norad_id: Catalog ID of expected host satellite
            host_period_hours: Known rotation period of host (optional)
            
        Returns:
            RPODetection if sleeper satellite detected, None otherwise
        """
        # Extract spectrum time series from cube
        # Assume cube is [time, wavelength, y, x] or reshape appropriately
        time_series = self._extract_time_series(spectral_cube, x, y)
        
        if time_series is None or len(time_series) < 10:
            logger.debug(f"Insufficient data at pixel ({x}, {y})")
            return None
        
        # Apply BSS
        separated_sources, mixing_matrix = self._separate_sources(time_series)
        
        if separated_sources is None:
            return None
        
        # Analyze rotation periods in separated signals
        periods = []
        for i, source in enumerate(separated_sources):
            period = self._estimate_rotation_period(source)
            if period is not None:
                periods.append((i, period, source))
        
        # Check if we found multiple distinct periods
        if len(periods) < 2:
            logger.debug(f"Only {len(periods)} period found, no RPO detected")
            return None
        
        # Validate against expected host period
        if host_period_hours is not None:
            # Find source matching host period
            host_match = None
            sleeper_match = None
            
            for idx, period, source in periods:
                if abs(period - host_period_hours) / host_period_hours < 0.2:
                    # Within 20% of expected - this is the host
                    host_match = (idx, period, source)
                else:
                    # Different period - potential sleeper
                    sleeper_match = (idx, period, source)
            
            if host_match is None or sleeper_match is None:
                logger.debug("Periods don't match expected host signature")
                return None
            
            # Calculate confidence based on separation
            confidence = self._calculate_confidence(
                host_match[2], sleeper_match[2], mixing_matrix
            )
            
            # Estimate physical separation from flux ratio
            separation = self._estimate_separation(
                host_match[2], sleeper_match[2], mixing_matrix
            )
            
            return RPODetection(
                host_norad_id=norad_id,
                primary_period_hours=host_match[1],
                secondary_period_hours=sleeper_match[1],
                separation_km=separation,
                confidence=confidence,
                detection_time=datetime.now(),
                method=self.method,
                spectral_signatures=[host_match[2], sleeper_match[2]]
            )
        
        # No host period given - check for any multiple periods
        else:
            # Find two most distinct periods
            periods.sort(key=lambda p: p[1])  # Sort by period
            
            if len(periods) >= 2:
                period_ratio = max(periods[0][1], periods[1][1]) / min(periods[0][1], periods[1][1])
                
                if period_ratio > 1.5:  # At least 50% different
                    confidence = self._calculate_confidence(
                        periods[0][2], periods[1][2], mixing_matrix
                    )
                    
                    if confidence > 0.6:  # 60% confidence threshold
                        return RPODetection(
                            host_norad_id=norad_id,
                            primary_period_hours=periods[0][1],
                            secondary_period_hours=periods[1][1],
                            separation_km=0.1,  # Unknown, default to 100m
                            confidence=confidence,
                            detection_time=datetime.now(),
                            method=self.method,
                            spectral_signatures=[periods[0][2], periods[1][2]]
                        )
        
        return None
    
    def _extract_time_series(self, cube: Cube, x: int, y: int) -> Optional[np.ndarray]:
        """Extract spectral time series from pixel (x, y)"""
        try:
            # Handle different cube shapes
            if len(cube.shape) == 3:
                # [time, y, x] - single wavelength band
                return cube[:, y, x].data
            elif len(cube.shape) == 4:
                # [time, wavelength, y, x] - flatten wavelength into features
                n_time = cube.shape[0]
                n_wave = cube.shape[1]
                
                time_series = np.zeros((n_time, n_wave))
                for t in range(n_time):
                    for w in range(n_wave):
                        time_series[t, w] = cube[t, w, y, x]
                
                return time_series
            else:
                logger.warning(f"Unsupported cube shape: {cube.shape}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting time series: {e}")
            return None
    
    def _separate_sources(self, time_series: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply PCA or ICA to separate mixed sources"""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(time_series)
            
            # Apply BSS method
            if self.method == "PCA":
                model = PCA(n_components=self.n_components, whiten=True)
                sources = model.fit_transform(X_scaled)
                mixing = model.components_
                
            elif self.method == "ICA":
                model = FastICA(
                    n_components=self.n_components,
                    max_iter=500,
                    tol=0.0001,
                    whiten=True
                )
                sources = model.fit_transform(X_scaled)
                mixing = model.mixing_
                
            else:
                logger.error(f"Unknown BSS method: {self.method}")
                return None, None
            
            # Check explained variance for PCA
            if self.method == "PCA":
                explained_variance = model.explained_variance_ratio_
                logger.info(f"PCA explained variance: {explained_variance}")
                
                # Require at least 2 significant components
                if len(explained_variance) < 2 or explained_variance[1] < 0.1:
                    logger.debug("Insufficient secondary component")
                    return None, None
            
            return sources.T, mixing  # [n_components, n_samples]
            
        except Exception as e:
            logger.error(f"BSS separation failed: {e}")
            return None, None
    
    def _estimate_rotation_period(self, signal: np.ndarray) -> Optional[float]:
        """Estimate rotation period from signal using FFT"""
        try:
            # Detrend
            signal_detrended = signal - np.mean(signal)
            
            # FFT for frequency analysis
            fft = np.fft.fft(signal_detrended)
            freqs = np.fft.fftfreq(len(signal_detrended))
            
            # Find dominant frequency (excluding DC)
            magnitude = np.abs(fft)
            magnitude[0] = 0  # Ignore DC component
            
            # Find peak
            peak_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            peak_freq = freqs[peak_idx]
            
            if peak_freq == 0:
                return None
            
            # Convert to period in hours
            # Assume sampling is regular, estimate from signal length
            # This is simplified - real implementation would use actual timestamps
            n_samples = len(signal)
            sampling_interval_hours = 0.5  # Assume 30-minute sampling
            
            period_hours = 1.0 / abs(peak_freq) * sampling_interval_hours
            
            # Validate range
            if self.rotation_period_range[0] <= period_hours <= self.rotation_period_range[1]:
                return period_hours
            
            return None
            
        except Exception as e:
            logger.error(f"Period estimation failed: {e}")
            return None
    
    def _calculate_confidence(
        self,
        source1: np.ndarray,
        source2: np.ndarray,
        mixing_matrix: np.ndarray
    ) -> float:
        """Calculate detection confidence based on source separation quality"""
        try:
            # Correlation between sources (should be low for good separation)
            correlation = np.abs(np.corrcoef(source1, source2)[0, 1])
            
            # SNR of each source
            snr1 = np.std(source1) / (np.mean(np.abs(source1)) + 1e-10)
            snr2 = np.std(source2) / (np.mean(np.abs(source2)) + 1e-10)
            
            # Mixing matrix condition (well-conditioned = good separation)
            condition_number = np.linalg.cond(mixing_matrix)
            
            # Combined confidence score
            # Low correlation is good, high SNR is good, low condition number is good
            confidence = (
                (1.0 - correlation) * 0.4 +      # 40% weight on independence
                min(snr1 / 10.0, 1.0) * 0.3 +  # 30% weight on primary SNR
                min(snr2 / 10.0, 1.0) * 0.2 +  # 20% weight on secondary SNR
                (1.0 / (1.0 + condition_number / 100.0)) * 0.1  # 10% weight on conditioning
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _estimate_separation(
        self,
        host_signal: np.ndarray,
        sleeper_signal: np.ndarray,
        mixing_matrix: np.ndarray
    ) -> float:
        """Estimate physical separation from flux ratio"""
        try:
            # Flux ratio from mixing matrix
            host_flux = np.max(np.abs(host_signal))
            sleeper_flux = np.max(np.abs(sleeper_signal))
            
            if sleeper_flux == 0:
                return 0.0
            
            flux_ratio = host_flux / sleeper_flux
            
            # Estimate separation assuming similar albedos
            # Brighter object dominates, flux ~ 1/r^2
            # If flux ratio is 100:1, distance ratio is ~10:1
            distance_ratio = np.sqrt(flux_ratio)
            
            # Assume host is typical LEO satellite ~500 km
            # Sleeper could be very close (RPO)
            estimated_separation = 500.0 / distance_ratio
            
            # Cap at reasonable RPO distances (10m to 10km)
            return max(0.01, min(estimated_separation, 10.0))
            
        except Exception as e:
            logger.error(f"Separation estimation failed: {e}")
            return 0.1  # Default 100m


def scan_for_rpo_situations(
    spectral_cube: Cube,
    indian_asset_ids: List[int],
    db_connection: str
) -> List[RPODetection]:
    """
    Scan multiple pixels around Indian assets for RPO
    
    Args:
        spectral_cube: Time-series spectral data
        indian_asset_ids: List of NORAD IDs to check
        db_connection: PostgreSQL connection string
    """
    import psycopg2
    
    detections = []
    bss = BlindSourceSeparator(method="ICA", n_components=2)
    
    # Query predicted positions of Indian assets
    conn = psycopg2.connect(db_connection)
    cur = conn.cursor()
    
    for norad_id in indian_asset_ids:
        # Get predicted pixel coordinates for this asset
        cur.execute("""
            SELECT 
                ST_X(position) as lon,
                ST_Y(position) as lat,
                mean_motion  -- For period estimation
            FROM current_states
            WHERE norad_cat_id = %s
            AND last_updated > NOW() - INTERVAL '5 minutes'
        """, (norad_id,))
        
        row = cur.fetchone()
        if not row:
            continue
        
        lon, lat, mean_motion = row
        
        # Estimate rotation period from mean motion (simplified)
        # Real period would come from light curve analysis
        period_hours = 24.0 / mean_motion if mean_motion > 0 else 1.5
        
        # Analyze pixels around predicted position (3x3 grid)
        # Convert lat/lon to pixel coordinates (requires WCS)
        center_x, center_y = 512, 512  # Placeholder
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = center_x + dx * 5  # 5-pixel steps
                y = center_y + dy * 5
                
                # Skip if out of bounds
                if x < 0 or x >= spectral_cube.shape[-1]:
                    continue
                if y < 0 or y >= spectral_cube.shape[-2]:
                    continue
                
                # Analyze this pixel
                detection = bss.analyze_pixel(
                    spectral_cube, x, y, norad_id, period_hours
                )
                
                if detection and detection.confidence > 0.7:
                    detections.append(detection)
                    logger.warning(
                        f"RPO DETECTED: NORAD {norad_id} at ({x}, {y}), "
                        f"confidence={detection.confidence:.2f}, "
                        f"separation={detection.separation_km:.3f} km"
                    )
    
    cur.close()
    conn.close()
    
    return detections


# Example usage
if __name__ == "__main__":
    # Create synthetic test data
    # Two rotating objects with different periods
    n_time = 100
    n_wave = 10
    
    # Host: 2-hour period
    t = np.linspace(0, 24, n_time)  # 24 hours
    host_signal = np.sin(2 * np.pi * t / 2.0)  # 2-hour period
    
    # Sleeper: 4-hour period
    sleeper_signal = np.sin(2 * np.pi * t / 4.0)  # 4-hour period
    
    # Mix signals (70% host, 30% sleeper)
    mixed_signal = 0.7 * host_signal + 0.3 * sleeper_signal
    
    # Create cube with multiple wavelengths (duplicate for simplicity)
    cube_data = np.zeros((n_time, n_wave, 32, 32))
    for i in range(n_wave):
        cube_data[:, i, 16, 16] = mixed_signal
    
    from mpdaf.obj import Cube
    cube = Cube(data=cube_data)
    
    # Analyze
    bss = BlindSourceSeparator(method="ICA", n_components=2)
    detection = bss.analyze_pixel(cube, 16, 16, 12345, host_period_hours=2.0)
    
    if detection:
        print(f"RPO Detected!")
        print(f"  Host period: {detection.primary_period_hours:.1f} hours")
        print(f"  Sleeper period: {detection.secondary_period_hours:.1f} hours")
        print(f"  Confidence: {detection.confidence:.2f}")
        print(f"  Estimated separation: {detection.separation_km:.3f} km")
    else:
        print("No RPO detected")
