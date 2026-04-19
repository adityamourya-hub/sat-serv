"""
Android RTL-SDR Bridge for Mobile SIGINT
Provides TCP streaming interface for RTL-SDR data from Android devices
"""

import socket
import numpy as np
import time
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    host: str = "192.168.1.101"
    port: int = 1234
    sample_rate: int = 2048000  # 2.048 MHz default
    center_freq: float = 1575.42e6  # GPS L1 default
    gain: str = "auto"
    buffer_size: int = 16384


class AndroidSDRBridge:
    """
    Android RTL-SDR Bridge via TCP streaming
    Connects to rtl_tcp running on Android device
    """
    
    def __init__(self, config: SDRConfig = None):
        self.config = config or SDRConfig()
        self.socket: Optional[socket.socket] = None
        self.is_connected = False
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.data_queue: queue.Queue = queue.Queue(maxsize=100)
        self.callbacks: list[Callable[[np.ndarray], None]] = []
        
        # GPS spoofing detection state
        self.gps_reference_position = None
        self.last_samples = None
        
    def connect(self) -> bool:
        """
        Connect to Android RTL-SDR via TCP
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((self.config.host, self.config.port))
            self.is_connected = True
            logger.info(f"Connected to Android SDR at {self.config.host}:{self.config.port}")
            
            # Configure SDR parameters
            self._configure_sdr()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Android SDR: {e}")
            self.is_connected = False
            return False
    
    def _configure_sdr(self):
        """Send configuration commands to rtl_tcp"""
        if not self.is_connected:
            return
        
        try:
            # rtl_tcp command format: command (1 byte) + parameter (4 bytes, big-endian)
            
            # Set center frequency
            freq_cmd = bytes([0x01]) + int(self.config.center_freq).to_bytes(4, 'big')
            self.socket.send(freq_cmd)
            
            # Set sample rate
            rate_cmd = bytes([0x02]) + int(self.config.sample_rate).to_bytes(4, 'big')
            self.socket.send(rate_cmd)
            
            # Set gain (if not auto)
            if self.config.gain != "auto":
                gain_val = int(float(self.config.gain) * 10)  # rtl_tcp uses tenths of dB
                gain_cmd = bytes([0x04]) + gain_val.to_bytes(4, 'big')
                self.socket.send(gain_cmd)
            else:
                # Enable AGC
                agc_cmd = bytes([0x08]) + (1).to_bytes(4, 'big')
                self.socket.send(agc_cmd)
            
            logger.info(f"SDR configured: Freq={self.config.center_freq/1e6:.3f} MHz, "
                       f"Rate={self.config.sample_rate/1e6:.3f} MHz")
        except Exception as e:
            logger.error(f"Failed to configure SDR: {e}")
    
    def disconnect(self):
        """Disconnect from Android SDR"""
        self.stop_streaming()
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.is_connected = False
        logger.info("Disconnected from Android SDR")
    
    def get_realtime_iq(self, num_samples: int = 2048) -> Optional[np.ndarray]:
        """
        Read real-time IQ samples from Android SDR
        
        Args:
            num_samples: Number of complex samples to read
            
        Returns:
            Complex64 numpy array or None if error
        """
        if not self.is_connected:
            logger.warning("Not connected to Android SDR")
            return None
        
        try:
            # Read raw bytes (2 bytes per sample: I + Q)
            num_bytes = num_samples * 2
            raw_data = self.socket.recv(num_bytes)
            
            if len(raw_data) < num_bytes:
                logger.warning(f"Received {len(raw_data)} bytes, expected {num_bytes}")
                return None
            
            # Convert to complex64
            iq = self._convert_to_complex(raw_data)
            self.last_samples = iq
            return iq
            
        except Exception as e:
            logger.error(f"Failed to read IQ samples: {e}")
            return None
    
    def _convert_to_complex(self, raw_data: bytes) -> np.ndarray:
        """
        Convert raw 8-bit unsigned IQ bytes to complex64
        
        Args:
            raw_data: Raw bytes from rtl_tcp (uint8 interleaved I,Q)
            
        Returns:
            Complex64 numpy array
        """
        # Convert to numpy array
        iq_uint8 = np.frombuffer(raw_data, dtype=np.uint8)
        
        # Convert to float32 and normalize (0-255 -> -1 to 1)
        iq_float = (iq_uint8.astype(np.float32) - 127.5) / 127.5
        
        # Deinterleave I and Q
        i_samples = iq_float[0::2]
        q_samples = iq_float[1::2]
        
        # Create complex array
        return i_samples + 1j * q_samples
    
    def start_streaming(self, callback: Callable[[np.ndarray], None] = None):
        """
        Start continuous streaming thread
        
        Args:
            callback: Optional callback function to process samples
        """
        if not self.is_connected:
            logger.error("Cannot start streaming - not connected")
            return
        
        if callback:
            self.callbacks.append(callback)
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Started streaming thread")
    
    def stop_streaming(self):
        """Stop streaming thread"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            self.stream_thread = None
        logger.info("Stopped streaming thread")
    
    def _streaming_loop(self):
        """Internal streaming loop"""
        while self.is_streaming and self.is_connected:
            try:
                # Read samples
                samples = self.get_realtime_iq(self.config.buffer_size // 2)
                if samples is not None:
                    # Add to queue
                    try:
                        self.data_queue.put_nowait(samples)
                    except queue.Full:
                        # Drop oldest sample
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(samples)
                        except queue.Empty:
                            pass
                    
                    # Call registered callbacks
                    for callback in self.callbacks:
                        try:
                            callback(samples)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                            
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                time.sleep(0.1)
    
    def get_queued_samples(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get samples from the queue (non-blocking)
        
        Args:
            timeout: Maximum time to wait for samples
            
        Returns:
            Complex64 numpy array or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_frequency(self, freq_hz: float):
        """
        Change center frequency
        
        Args:
            freq_hz: New center frequency in Hz
        """
        self.config.center_freq = freq_hz
        if self.is_connected:
            try:
                freq_cmd = bytes([0x01]) + int(freq_hz).to_bytes(4, 'big')
                self.socket.send(freq_cmd)
                logger.info(f"Set frequency to {freq_hz/1e6:.3f} MHz")
            except Exception as e:
                logger.error(f"Failed to set frequency: {e}")
    
    def detect_gps_spoofing(self, android_gps_pos: dict, sgp4_truth_pos: dict) -> dict:
        """
        Detect GPS spoofing by comparing Android GPS with SGP4 truth
        
        Args:
            android_gps_pos: Android GPS position {'lat': float, 'lon': float, 'alt': float}
            sgp4_truth_pos: SGP4 calculated position {'lat': float, 'lon': float, 'alt': float}
            
        Returns:
            Spoofing detection result dict
        """
        # Calculate position delta
        lat_diff = abs(android_gps_pos['lat'] - sgp4_truth_pos['lat'])
        lon_diff = abs(android_gps_pos['lon'] - sgp4_truth_pos['lon'])
        alt_diff = abs(android_gps_pos['alt'] - sgp4_truth_pos['alt'])
        
        # Thresholds for spoofing detection
        LAT_THRESHOLD = 0.01  # degrees
        LON_THRESHOLD = 0.01  # degrees
        ALT_THRESHOLD = 100.0  # meters
        
        is_spoofed = (lat_diff > LAT_THRESHOLD or 
                     lon_diff > LON_THRESHOLD or 
                     alt_diff > ALT_THRESHOLD)
        
        # Check spectrum for jamming indicators
        jamming_detected = False
        if self.last_samples is not None:
            # Simple energy detection in GPS L1 band
            power = np.mean(np.abs(self.last_samples)**2)
            if power > 0.5:  # High power indicates jamming
                jamming_detected = True
        
        return {
            'spoofing_detected': is_spoofed,
            'jamming_detected': jamming_detected,
            'lat_delta_deg': lat_diff,
            'lon_delta_deg': lon_diff,
            'alt_delta_m': alt_diff,
            'timestamp': time.time(),
            'action': 'TACTICAL_ALERT' if (is_spoofed or jamming_detected) else 'MONITOR'
        }
    
    def scan_frequency_range(self, start_freq: float, end_freq: float, 
                            step_hz: float = 1e6, dwell_time: float = 0.5) -> list:
        """
        Scan a frequency range for signal activity
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            step_hz: Frequency step size
            dwell_time: Time to dwell on each frequency
            
        Returns:
            List of detected signals [{'freq': float, 'power': float}]
        """
        detected_signals = []
        
        current_freq = start_freq
        while current_freq <= end_freq:
            self.set_frequency(current_freq)
            time.sleep(dwell_time)
            
            # Read samples and calculate power
            samples = self.get_realtime_iq(2048)
            if samples is not None:
                power = 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-10)
                
                # If power is above threshold, record signal
                if power > -60:  # -60 dBm threshold
                    detected_signals.append({
                        'freq_hz': current_freq,
                        'power_dbm': power,
                        'timestamp': time.time()
                    })
                    logger.info(f"Detected signal at {current_freq/1e6:.3f} MHz: {power:.1f} dBm")
            
            current_freq += step_hz
        
        return detected_signals


class JAXFFTEngine:
    """
    JAX-based FFT engine for real-time spectrogram processing
    Provides FFT fingerprinting for signal identification
    """
    
    def __init__(self, fft_size: int = 4096):
        self.fft_size = fft_size
        self.jax_available = False
        
        # Try to import JAX
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            self.jax_available = True
            logger.info("JAX FFT engine initialized")
        except ImportError:
            logger.warning("JAX not available, falling back to numpy FFT")
            self.jax_available = False
    
    def compute_spectrogram(self, iq_samples: np.ndarray, 
                           window_size: int = 256) -> np.ndarray:
        """
        Compute spectrogram using FFT
        
        Args:
            iq_samples: Complex IQ samples
            window_size: FFT window size
            
        Returns:
            2D spectrogram array
        """
        if self.jax_available:
            return self._compute_spectrogram_jax(iq_samples, window_size)
        else:
            return self._compute_spectrogram_numpy(iq_samples, window_size)
    
    def _compute_spectrogram_jax(self, iq_samples: np.ndarray, 
                                 window_size: int) -> np.ndarray:
        """Compute spectrogram using JAX for acceleration"""
        # Convert to JAX array
        samples_jax = self.jnp.array(iq_samples)
        
        # Create overlapping windows
        hop_length = window_size // 4
        num_frames = (len(iq_samples) - window_size) // hop_length
        
        # Hann window
        window = self.jnp.hanning(window_size)
        
        # Compute STFT
        spectrogram = []
        for i in range(num_frames):
            start = i * hop_length
            frame = samples_jax[start:start + window_size] * window
            fft_result = self.jnp.fft.fft(frame)
            spectrogram.append(self.jnp.abs(fft_result[:window_size // 2]))
        
        return np.array(self.jnp.stack(spectrogram))
    
    def _compute_spectrogram_numpy(self, iq_samples: np.ndarray,
                                   window_size: int) -> np.ndarray:
        """Compute spectrogram using NumPy (fallback)"""
        hop_length = window_size // 4
        num_frames = (len(iq_samples) - window_size) // hop_length
        
        window = np.hanning(window_size)
        spectrogram = []
        
        for i in range(num_frames):
            start = i * hop_length
            frame = iq_samples[start:start + window_size] * window
            fft_result = np.fft.fft(frame)
            spectrogram.append(np.abs(fft_result[:window_size // 2]))
        
        return np.array(spectrogram)
    
    def generate_fft_fingerprint(self, iq_samples: np.ndarray) -> str:
        """
        Generate FFT-based fingerprint (hash) of signal
        Similar to Shazam algorithm for audio fingerprinting
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Fingerprint string
        """
        # Compute spectrogram
        spec = self.compute_spectrogram(iq_samples, window_size=512)
        
        # Find peaks in time-frequency domain
        peaks = self._find_spectral_peaks(spec)
        
        # Generate hash from peak pairs
        fingerprint = self._hash_peaks(peaks)
        
        return fingerprint
    
    def _find_spectral_peaks(self, spectrogram: np.ndarray) -> list:
        """Find peaks in spectrogram"""
        peaks = []
        
        # Simple peak detection
        for t in range(0, spectrogram.shape[0], 10):  # Sample every 10th frame
            frame = spectrogram[t]
            
            # Find local maxima
            for f in range(1, len(frame) - 1):
                if frame[f] > frame[f-1] and frame[f] > frame[f+1] and frame[f] > np.mean(frame):
                    peaks.append((t, f, frame[f]))
        
        # Keep top 100 peaks
        peaks.sort(key=lambda x: x[2], reverse=True)
        return peaks[:100]
    
    def _hash_peaks(self, peaks: list) -> str:
        """Generate hash from peak constellation"""
        if len(peaks) < 2:
            return ""
        
        hashes = []
        for i in range(min(len(peaks) - 1, 50)):
            for j in range(i + 1, min(i + 10, len(peaks))):
                t1, f1, _ = peaks[i]
                t2, f2, _ = peaks[j]
                
                # Time-frequency offset hash
                dt = t2 - t1
                df = f2 - f1
                hash_val = hash((f1, df, dt)) % (2**32)
                hashes.append(f"{hash_val:08x}")
        
        return "|".join(hashes[:100])


# Convenience function for quick SDR connection
def connect_android_sdr(host: str = "192.168.1.101", port: int = 1234) -> Optional[AndroidSDRBridge]:
    """
    Quick connect to Android SDR
    
    Args:
        host: Android device IP address
        port: rtl_tcp port
        
    Returns:
        Connected AndroidSDRBridge or None
    """
    config = SDRConfig(host=host, port=port)
    bridge = AndroidSDRBridge(config)
    
    if bridge.connect():
        return bridge
    return None


if __name__ == "__main__":
    # Example usage
    logger.info("Starting Android SDR Bridge demo")
    
    # Connect to Android SDR
    sdr = connect_android_sdr("192.168.1.101", 1234)
    
    if sdr:
        logger.info("Connected successfully")
        
        # Scan GPS L1 band
        logger.info("Scanning GPS L1 band...")
        signals = sdr.scan_frequency_range(
            start_freq=1575.0e6,
            end_freq=1576.0e6,
            step_hz=100e3,
            dwell_time=0.5
        )
        
        logger.info(f"Detected {len(signals)} signals")
        for sig in signals:
            logger.info(f"  {sig['freq_hz']/1e6:.3f} MHz: {sig['power_dbm']:.1f} dBm")
        
        # Read some samples
        samples = sdr.get_realtime_iq(2048)
        if samples is not None:
            logger.info(f"Read {len(samples)} IQ samples")
            
            # Initialize JAX FFT engine
            fft_engine = JAXFFTEngine()
            
            # Generate fingerprint
            fingerprint = fft_engine.generate_fft_fingerprint(samples)
            logger.info(f"Generated fingerprint: {fingerprint[:50]}...")
        
        sdr.disconnect()
    else:
        logger.error("Failed to connect to Android SDR")
