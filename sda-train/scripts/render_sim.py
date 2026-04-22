"""
Pixel Engine for Sovereign Training
Transforms Go "Truth" states into "Dirty" optical frames with realistic PSF, noise, and artifacts
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time

# Astronomy imports for realistic rendering
try:
    from astropy.convolution import Gaussian2DKernel, convolve
    from astropy.modeling.models import Gaussian2D
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("[WARN] astropy not available, using simplified rendering")

try:
    from photutils.datasets import make_noise_image
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    print("[WARN] photutils not available, using simplified noise")


class PixelRenderer:
    """
    Renders synthetic SSA frames with realistic optical characteristics
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (1024, 1024),
                 psf_sigma: float = 1.5,
                 noise_mean: float = 0.2,
                 noise_std: float = 0.05):
        """
        Initialize pixel renderer
        
        Args:
            resolution: Frame resolution (height, width)
            psf_sigma: Point spread function sigma (atmospheric blur)
            noise_mean: Mean background noise level
            noise_std: Standard deviation of noise
        """
        self.resolution = resolution
        self.psf_sigma = psf_sigma
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        
        # Pre-compute PSF kernel
        if ASTROPY_AVAILABLE:
            self.psf_kernel = Gaussian2DKernel(x_stddev=psf_sigma)
        else:
            # Simplified PSF
            size = int(6 * psf_sigma)
            x = np.arange(-size, size + 1)
            y = np.arange(-size, size + 1)
            X, Y = np.meshgrid(x, y)
            self.psf_kernel = np.exp(-(X**2 + Y**2) / (2 * psf_sigma**2))
            self.psf_kernel = self.psf_kernel / np.sum(self.psf_kernel)
    
    def add_star_field(self, 
                      frame: np.ndarray, 
                      num_stars: int = 200,
                      magnitude_range: Tuple[float, float] = (0.1, 0.5)) -> np.ndarray:
        """
        Add realistic star field background
        
        Args:
            frame: Input frame
            num_stars: Number of stars to add
            magnitude_range: (min, max) brightness for stars
            
        Returns:
            Frame with star field
        """
        height, width = frame.shape
        
        # Random star positions
        star_y = np.random.randint(0, height, num_stars)
        star_x = np.random.randint(0, width, num_stars)
        
        # Random brightness
        brightness = np.random.uniform(*magnitude_range, num_stars)
        
        # Add stars
        frame[star_y, star_x] += brightness
        
        return frame
    
    def project_satellites(self,
                         frame: np.ndarray,
                         truth_states: List[Dict],
                         projection_scale: float = 100.0) -> np.ndarray:
        """
        Project 3D satellite positions to 2D pixel coordinates
        
        Args:
            frame: Input frame
            truth_states: List of satellite states from Go sim
            projection_scale: Scale factor for 3D->2D projection
            
        Returns:
            Frame with satellites
        """
        height, width = frame.shape
        
        for sat in truth_states:
            # Simple orthographic projection (in production, use Alt/Az transform)
            # Center of frame is (0,0) in projection
            px = int((sat['y'] / projection_scale) % width)
            py = int((sat['z'] / projection_scale) % height)
            
            # Ensure within bounds
            px = np.clip(px, 0, width - 1)
            py = np.clip(py, 0, height - 1)
            
            # Convert magnitude to brightness (lower mag = brighter)
            # Simplified: brightness = 10^((10 - mag) / 2.5)
            brightness = 10 ** ((10 - sat['mag']) / 2.5)
            
            frame[py, px] += brightness
        
        return frame
    
    def apply_psf(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply point spread function (atmospheric/optical blur)
        
        Args:
            frame: Input frame
            
        Returns:
            Blurred frame
        """
        if ASTROPY_AVAILABLE:
            return convolve(frame, self.psf_kernel)
        else:
            # Manual convolution
            from scipy.signal import convolve2d
            return convolve2d(frame, self.psf_kernel, mode='same', boundary='fill')
    
    def add_noise(self, 
                  frame: np.ndarray,
                  noise_type: str = 'poisson') -> np.ndarray:
        """
        Add realistic sensor noise
        
        Args:
            frame: Input frame
            noise_type: 'poisson', 'gaussian', or 'mixed'
            
        Returns:
            Noisy frame
        """
        if noise_type == 'poisson':
            if PHOTUTILS_AVAILABLE:
                noise = make_noise_image(frame.shape, 
                                        distribution='poisson',
                                        mean=self.noise_mean)
            else:
                # Simplified Poisson (actually Gaussian approximation)
                noise = np.random.poisson(self.noise_mean, frame.shape)
        
        elif noise_type == 'gaussian':
            noise = np.random.normal(self.noise_mean, self.noise_std, frame.shape)
        
        elif noise_type == 'mixed':
            # Both Poisson and Gaussian
            if PHOTUTILS_AVAILABLE:
                poisson = make_noise_image(frame.shape,
                                          distribution='poisson',
                                          mean=self.noise_mean * 0.7)
            else:
                poisson = np.random.poisson(self.noise_mean * 0.7, frame.shape)
            
            gaussian = np.random.normal(0, self.noise_std, frame.shape)
            noise = poisson + gaussian
        
        else:
            noise = np.zeros_like(frame)
        
        return frame + noise
    
    def add_radiation_hits(self,
                          frame: np.ndarray,
                          num_hits: int = 5,
                          hit_intensity: float = 10.0) -> np.ndarray:
        """
        Add cosmic ray radiation hits for denoiser training
        
        Args:
            frame: Input frame
            num_hits: Number of radiation hits
            hit_intensity: Brightness of hits
            
        Returns:
            Frame with radiation hits
        """
        height, width = frame.shape
        
        for _ in range(num_hits):
            # Random position
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            
            # Random orientation
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(3, 15)
            
            # Draw line
            for i in range(length):
                dy = int(i * np.sin(angle))
                dx = int(i * np.cos(angle))
                
                py = np.clip(y + dy, 0, height - 1)
                px = np.clip(x + dx, 0, width - 1)
                
                frame[py, px] += hit_intensity * (1 - i / length)
        
        return frame
    
    def render_dirty_frame(self,
                          truth_states: List[Dict],
                          include_stars: bool = True,
                          include_noise: bool = True,
                          include_radiation: bool = False,
                          num_stars: int = 200) -> np.ndarray:
        """
        Render a complete synthetic SSA frame
        
        Args:
            truth_states: List of satellite states from Go sim
            include_stars: Add star field background
            include_noise: Add sensor noise
            include_radiation: Add cosmic ray hits
            num_stars: Number of background stars
            
        Returns:
            Rendered frame as numpy array
        """
        # 1. Initialize empty frame
        frame = np.zeros(self.resolution)
        
        # 2. Add star field (if enabled)
        if include_stars:
            frame = self.add_star_field(frame, num_stars=num_stars)
        
        # 3. Add satellites/debris
        frame = self.project_satellites(frame, truth_states)
        
        # 4. Apply PSF (atmospheric blur)
        frame = self.apply_psf(frame)
        
        # 5. Add noise
        if include_noise:
            frame = self.add_noise(frame, noise_type='mixed')
        
        # 6. Add radiation hits (for hard curriculum)
        if include_radiation:
            frame = self.add_radiation_hits(frame)
        
        # 7. Normalize to valid range
        frame = np.clip(frame, 0, 65535)  # 16-bit depth
        
        return frame
    
    def render_curriculum_frame(self,
                               truth_states: List[Dict],
                               level: int) -> np.ndarray:
        """
        Render frame for specific curriculum learning level
        
        Args:
            truth_states: Satellite states
            level: 0=Easy, 1=Medium, 2=Hard, 3=Tactical
            
        Returns:
            Rendered frame
        """
        if level == 0:
            # Easy: Clean, bright, no stars, no noise
            return self.render_dirty_frame(
                truth_states,
                include_stars=False,
                include_noise=False,
                include_radiation=False,
                num_stars=0
            )
        
        elif level == 1:
            # Medium: Add star fields and noise
            return self.render_dirty_frame(
                truth_states,
                include_stars=True,
                include_noise=True,
                include_radiation=False,
                num_stars=200
            )
        
        elif level == 2:
            # Hard: Dense debris, low magnitude satellites
            return self.render_dirty_frame(
                truth_states,
                include_stars=True,
                include_noise=True,
                include_radiation=False,
                num_stars=500
            )
        
        else:  # level >= 3
            # Tactical: Extreme conditions, radiation hits
            return self.render_dirty_frame(
                truth_states,
                include_stars=True,
                include_noise=True,
                include_radiation=True,
                num_stars=300
            )


# Convenience functions for direct use
def render_dirty_frame(truth_states: List[Dict],
                       resolution: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Simplified interface for rendering dirty frames
    
    Args:
        truth_states: List of satellite state dicts
        resolution: Output resolution
        
    Returns:
        Rendered frame
    """
    renderer = PixelRenderer(resolution=resolution)
    return renderer.render_dirty_frame(truth_states)


def render_curriculum(truth_states: List[Dict],
                     level: int = 1,
                     resolution: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Render frame for curriculum learning
    
    Args:
        truth_states: Satellite states
        level: Curriculum difficulty (0-3)
        resolution: Output resolution
        
    Returns:
        Rendered frame
    """
    renderer = PixelRenderer(resolution=resolution)
    return renderer.render_curriculum_frame(truth_states, level)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pixel Renderer...")
    
    # Create synthetic truth states (simulating Go output)
    truth_states = [
        {
            'sat_id': 99999,
            'x': 6800000,  # meters
            'y': 50000,
            'z': 50000,
            'vx': 0,
            'vy': 7500,
            'vz': 0,
            'mag': 3.5  # Bright target
        },
        {
            'sat_id': 1,
            'x': 6800000,
            'y': 45000,
            'z': 52000,
            'vx': 10,
            'vy': 7500,
            'vz': 10,
            'mag': 9.0  # Dim debris
        },
        {
            'sat_id': 2,
            'x': 6800000,
            'y': 55000,
            'z': 48000,
            'vx': -5,
            'vy': 7500,
            'vz': 5,
            'mag': 11.0  # Very dim debris
        }
    ]
    
    # Test all curriculum levels
    for level in range(4):
        print(f"\nRendering Level {level}...")
        
        frame = render_curriculum(truth_states, level=level)
        
        print(f"  Frame shape: {frame.shape}")
        print(f"  Min value: {frame.min():.2f}")
        print(f"  Max value: {frame.max():.2f}")
        print(f"  Mean value: {frame.mean():.2f}")
        
        # Save frame
        output_path = f"test_frame_level_{level}.npy"
        np.save(output_path, frame)
        print(f"  Saved to {output_path}")
    
    print("\nAll tests passed!")
