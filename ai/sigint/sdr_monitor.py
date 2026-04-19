"""
RTL-SDR GPS Monitoring for Spoofing Detection
Monitors L1/L2 GPS bands for spoofing attempts using RTL-SDR
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import time
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPSBandConfig:
    """GPS band configuration"""
    name: str
    center_freq_hz: float
    bandwidth_hz: float
    sample_rate: int
    fft_size: int
    spoof_threshold_db: float


# GPS band configurations
GPS_BANDS = {
    'L1': GPSBandConfig(
        name='GPS_L1',
        center_freq_hz=1575.42e6,
        bandwidth_hz=20.46e6,
        sample_rate=2.4e6,
        fft_size=4096,
        spoof_threshold_db=20.0
    ),
    'L2': GPSBandConfig(
        name='GPS_L2',
        center_freq_hz=1227.60e6,
        bandwidth_hz=20.46e6,
        sample_rate=2.4e6,
        fft_size=4096,
        spoof_threshold_db=20.0
    ),
    'L5': GPSBandConfig(
        name='GPS_L5',
        center_freq_hz=1176.45e6,
        bandwidth_hz=20.46e6,
        sample_rate=2.4e6,
        fft_size=4096,
        spoof_threshold_db=20.0
    )
}


class RTLSDRMonitor:
    """
    RTL-SDR monitor for GPS spoofing detection
    Monitors L1/L2/L5 bands for anomalous signal patterns
    """
    
    def __init__(self, bands: list = None):
        """
        Initialize RTL-SDR monitor
        
        Args:
            bands: List of GPS bands to monitor (default: all)
        """
        self.bands = bands or ['L1', 'L2', 'L5']
        self.baseline_power = {band: None for band in self.bands}
        self.current_power = {band: None for band in self.bands}
        self.spoof_score = {band: 0.0 for band in self.bands}
        self.is_spoofed = {band: False for band in self.bands}
        self.history = {band: [] for band in self.bands}
        
        # Initialize rtl-sdr (mock for now, requires pyrtlsdr)
        self.sdr = None
        self.is_connected = False
        
    def connect(self, device_index: int = 0) -> bool:
        """
        Connect to RTL-SDR device
        
        Args:
            device_index: RTL-SDR device index
            
        Returns:
            True if connection successful
        """
        try:
            # TODO: Import and connect to pyrtlsdr
            # from rtlsdr import RtlSdr
            # self.sdr = RtlSdr(device_index)
            # self.sdr.sample_rate = GPS_BANDS['L1'].sample_rate
            # self.sdr.center_freq = GPS_BANDS['L1'].center_freq_hz
            # self.sdr.gain = 'auto'
            # self.is_connected = True
            logger.info("RTL-SDR connection not implemented - using mock mode")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to RTL-SDR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from RTL-SDR device"""
        if self.sdr:
            # self.sdr.close()
            self.sdr = None
        self.is_connected = False
        logger.info("RTL-SDR disconnected")
    
    def read_samples(self, band: str, num_samples: int = 256000) -> np.ndarray:
        """
        Read IQ samples from RTL-SDR
        
        Args:
            band: GPS band to monitor
            num_samples: Number of samples to read
            
        Returns:
            Complex IQ samples
        """
        if not self.is_connected:
            # Generate mock data for testing
            config = GPS_BANDS[band]
            t = np.arange(num_samples) / config.sample_rate
            # Mock GPS signal (C/A code like)
            samples = np.exp(2j * np.pi * 1000 * t) + 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            return samples
        
        # TODO: Read from actual RTL-SDR
        # samples = self.sdr.read_samples(num_samples)
        return np.array([])
    
    def compute_psd(self, samples: np.ndarray, config: GPSBandConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density
        
        Args:
            samples: IQ samples
            config: GPS band configuration
            
        Returns:
            Tuple of (frequencies, power_db)
        """
        # Compute FFT
        fft_result = fft(samples)
        fft_magnitude = np.abs(fft_result)
        
        # Convert to power spectral density
        psd = fft_magnitude ** 2
        psd_db = 10 * np.log10(psd + 1e-10)
        
        # Frequency axis
        freqs = fftfreq(config.fft_size, 1/config.sample_rate)
        
        # Shift to center frequency
        freqs += config.center_freq_hz
        
        return freqs, psd_db
    
    def detect_signal_power(self, psd_db: np.ndarray) -> float:
        """
        Detect signal power from PSD
        
        Args:
            psd_db: Power spectral density in dB
            
        Returns:
            Average power in dB
        """
        # Find peaks above noise floor
        noise_floor = np.percentile(psd_db, 10)
        signal_mask = psd_db > (noise_floor + 10)
        
        if not np.any(signal_mask):
            return noise_floor
        
        signal_power = np.mean(psd_db[signal_mask])
        return signal_power
    
    def update_baseline(self, band: str, num_samples: int = 10):
        """
        Update baseline power for a GPS band
        
        Args:
            band: GPS band
            num_samples: Number of samples to average
        """
        config = GPS_BANDS[band]
        powers = []
        
        for _ in range(num_samples):
            samples = self.read_samples(band)
            freqs, psd_db = self.compute_psd(samples, config)
            power = self.detect_signal_power(psd_db)
            powers.append(power)
        
        self.baseline_power[band] = np.mean(powers)
        logger.info(f"Baseline power for {band}: {self.baseline_power[band]:.2f} dB")
    
    def monitor_band(self, band: str) -> Dict:
        """
        Monitor a GPS band for spoofing
        
        Args:
            band: GPS band to monitor
            
        Returns:
            Dictionary with monitoring results
        """
        config = GPS_BANDS[band]
        
        # Read samples
        samples = self.read_samples(band)
        
        # Compute PSD
        freqs, psd_db = self.compute_psd(samples, config)
        
        # Detect power
        current_power = self.detect_signal_power(psd_db)
        self.current_power[band] = current_power
        
        # Check for spoofing
        if self.baseline_power[band] is not None:
            power_diff = current_power - self.baseline_power[band]
            
            # Update spoof score
            if power_diff > config.spoof_threshold_db:
                self.spoof_score[band] += 0.5
            else:
                self.spoof_score[band] = max(0, self.spoof_score[band] - 0.1)
            
            # Check if spoofed
            if self.spoof_score[band] > 0.7:
                self.is_spoofed[band] = True
            else:
                self.is_spoofed[band] = False
        
        # Store history
        self.history[band].append({
            'timestamp': time.time(),
            'power': current_power,
            'spoof_score': self.spoof_score[band],
            'is_spoofed': self.is_spoofed[band]
        })
        
        # Limit history size
        if len(self.history[band]) > 100:
            self.history[band] = self.history[band][-100:]
        
        return {
            'band': band,
            'power': current_power,
            'baseline': self.baseline_power[band],
            'spoof_score': self.spoof_score[band],
            'is_spoofed': self.is_spoofed[band],
            'timestamp': time.time()
        }
    
    def monitor_all_bands(self) -> Dict[str, Dict]:
        """
        Monitor all configured GPS bands
        
        Returns:
            Dictionary with results for each band
        """
        results = {}
        
        for band in self.bands:
            results[band] = self.monitor_band(band)
        
        return results
    
    def is_spoofing_detected(self) -> bool:
        """
        Check if spoofing is detected in any band
        
        Returns:
            True if spoofing detected
        """
        return any(self.is_spoofed.values())
    
    def get_spoofed_bands(self) -> list:
        """
        Get list of spoofed bands
        
        Returns:
            List of spoofed band names
        """
        return [band for band, spoofed in self.is_spoofed.items() if spoofed]
    
    def get_history(self, band: str) -> list:
        """
        Get monitoring history for a band
        
        Args:
            band: GPS band
            
        Returns:
            List of historical measurements
        """
        return self.history.get(band, [])


class GPSNMEAParser:
    """
    Parse NMEA data from GPS receiver
    Compare with RTL-SDR measurements for spoofing detection
    """
    
    def __init__(self):
        self.current_position = None
        self.current_velocity = None
        self.satellite_count = 0
        self.hdop = 0.0
        self.vdop = 0.0
        self.fix_quality = 0
    
    def parse_nmea(self, sentence: str) -> Optional[Dict]:
        """
        Parse NMEA sentence
        
        Args:
            sentence: NMEA sentence string
            
        Returns:
            Parsed data dictionary or None
        """
        if not sentence.startswith('$'):
            return None
        
        # Remove checksum
        if '*' in sentence:
            sentence = sentence.split('*')[0]
        
        fields = sentence.split(',')
        
        if fields[0] == '$GPGGA':
            return self._parse_gga(fields)
        elif fields[0] == '$GPRMC':
            return self._parse_rmc(fields)
        elif fields[0] == '$GPVTG':
            return self._parse_vtg(fields)
        
        return None
    
    def _parse_gga(self, fields: list) -> Dict:
        """Parse GGA sentence (position and fix data)"""
        try:
            lat = self._parse_latlon(fields[2], fields[3])
            lon = self._parse_latlon(fields[4], fields[5])
            altitude = float(fields[9]) if fields[9] else 0.0
            fix_quality = int(fields[6]) if fields[6] else 0
            satellites = int(fields[7]) if fields[7] else 0
            hdop = float(fields[8]) if fields[8] else 0.0
            
            self.current_position = {
                'latitude': lat,
                'longitude': lon,
                'altitude': altitude
            }
            self.fix_quality = fix_quality
            self.satellite_count = satellites
            self.hdop = hdop
            
            return self.current_position
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse GGA: {e}")
            return None
    
    def _parse_rmc(self, fields: list) -> Dict:
        """Parse RMC sentence (recommended minimum data)"""
        try:
            lat = self._parse_latlon(fields[3], fields[4])
            lon = self._parse_latlon(fields[5], fields[6])
            speed_knots = float(fields[7]) if fields[7] else 0.0
            heading = float(fields[8]) if fields[8] else 0.0
            
            self.current_position = {
                'latitude': lat,
                'longitude': lon
            }
            self.current_velocity = {
                'speed': speed_knots * 0.514444,  # Convert to m/s
                'heading': heading
            }
            
            return self.current_position
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse RMC: {e}")
            return None
    
    def _parse_vtg(self, fields: list) -> Dict:
        """Parse VTG sentence (course over ground)"""
        try:
            heading_true = float(fields[1]) if fields[1] else 0.0
            speed_knots = float(fields[5]) if fields[5] else 0.0
            
            self.current_velocity = {
                'heading': heading_true,
                'speed': speed_knots * 0.514444  # Convert to m/s
            }
            
            return self.current_velocity
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse VTG: {e}")
            return None
    
    def _parse_latlon(self, value: str, direction: str) -> float:
        """Parse latitude/longitude from NMEA format"""
        if not value:
            return 0.0
        
        degrees = float(value[:2]) if direction in ['N', 'S'] else float(value[:3])
        minutes = float(value[2:]) if direction in ['N', 'S'] else float(value[3:])
        
        decimal = degrees + minutes / 60.0
        
        if direction in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def get_position_quality(self) -> Dict:
        """
        Get GPS position quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        return {
            'satellite_count': self.satellite_count,
            'hdop': self.hdop,
            'vdop': self.vdop,
            'fix_quality': self.fix_quality,
            'is_valid': self.fix_quality >= 1 and self.satellite_count >= 4
        }


class SpoofingDetector:
    """
    Integrated GPS spoofing detector using RTL-SDR and NMEA
    """
    
    def __init__(self):
        self.sdr_monitor = RTLSDRMonitor()
        self.nmea_parser = GPSNMEAParser()
        self.alerts = []
    
    def initialize(self):
        """Initialize the spoofing detector"""
        # Connect to RTL-SDR
        self.sdr_monitor.connect()
        
        # Update baselines for all bands
        for band in self.sdr_monitor.bands:
            self.sdr_monitor.update_baseline(band)
    
    def check_spoofing(self) -> Dict:
        """
        Perform comprehensive spoofing check
        
        Returns:
            Dictionary with spoofing assessment
        """
        # Monitor RTL-SDR bands
        sdr_results = self.sdr_monitor.monitor_all_bands()
        
        # Get GPS quality
        gps_quality = self.nmea_parser.get_position_quality()
        
        # Check for spoofing indicators
        spoofing_detected = False
        indicators = []
        
        # Check RTL-SDR spoofing
        if self.sdr_monitor.is_spoofing_detected():
            spoofing_detected = True
            indicators.append("RTL-SDR power anomaly detected")
            indicators.extend([f"Band {band}: spoofing" for band in self.sdr_monitor.get_spoofed_bands()])
        
        # Check GPS quality
        if not gps_quality['is_valid']:
            spoofing_detected = True
            indicators.append("GPS fix invalid or insufficient satellites")
        
        if gps_quality['hdop'] > 5.0:
            spoofing_detected = True
            indicators.append("High HDOP indicates position uncertainty")
        
        # Check for abnormal velocity
        if self.nmea_parser.current_velocity:
            speed = self.nmea_parser.current_velocity.get('speed', 0)
            if speed > 1000:  # > 1 km/s is unrealistic for ground vehicle
                spoofing_detected = True
                indicators.append(f"Abnormal velocity detected: {speed:.2f} m/s")
        
        return {
            'spoofing_detected': spoofing_detected,
            'indicators': indicators,
            'sdr_results': sdr_results,
            'gps_quality': gps_quality,
            'timestamp': time.time()
        }
    
    def generate_alert(self) -> Optional[Dict]:
        """
        Generate spoofing alert if detected
        
        Returns:
            Alert dictionary or None
        """
        result = self.check_spoofing()
        
        if result['spoofing_detected']:
            alert = {
                'alert_id': f"GPS-SPOOF-{int(time.time())}",
                'severity': 'CRITICAL',
                'indicators': result['indicators'],
                'timestamp': time.time(),
                'action': 'Switch to inertial navigation, air-gap systems'
            }
            self.alerts.append(alert)
            return alert
        
        return None
    
    def get_alerts(self) -> list:
        """Get all alerts"""
        return self.alerts
    
    def clear_alerts(self):
        """Clear alert history"""
        self.alerts = []


def main():
    """Main entry point for RTL-SDR GPS monitoring"""
    logger.info("Starting RTL-SDR GPS Spoofing Monitor")
    
    # Create spoofing detector
    detector = SpoofingDetector()
    
    # Initialize
    detector.initialize()
    
    # Monitor loop
    try:
        while True:
            # Check for spoofing
            result = detector.check_spoofing()
            
            if result['spoofing_detected']:
                logger.warning(f"Spoofing detected: {result['indicators']}")
                alert = detector.generate_alert()
                if alert:
                    logger.critical(f"Alert generated: {alert}")
            else:
                logger.info("No spoofing detected")
            
            # Wait before next check
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        detector.sdr_monitor.disconnect()


if __name__ == "__main__":
    main()
