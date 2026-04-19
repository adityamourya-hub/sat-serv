"""
PSF Deconvolution & Super-Resolution for IAF SSA
Sharpens blurry satellite images using Richardson-Lucy deconvolution
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import logging

# scikit-image for deconvolution
from skimage import restoration
from skimage.measure import label, regionprops

# MPDAF for PSF modeling
from mpdaf.obj import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeconvolutionResult:
    """Result of PSF deconvolution"""
    original_image: np.ndarray
    deconvolved_image: np.ndarray
    psf_model: np.ndarray
    sharpness_metric: float
    recovered_resolution: float  # Estimated resolution after deconvolution (arcsec)
    convergence_history: list   # Richardson-Lucy iteration history


class PSFModeler:
    """Models atmospheric PSF from observed star/satellite profiles"""
    
    def __init__(self, pixel_scale_arcsec: float = 0.5):
        """
        Args:
            pixel_scale_arcsec: Pixel scale of sensor (arcsec/pixel)
        """
        self.pixel_scale = pixel_scale_arcsec
        self.psf_cache = {}
        
    def fit_gaussian_psf(
        self,
        image: np.ndarray,
        x_center: int,
        y_center: int,
        radius: int = 10
    ) -> Tuple[np.ndarray, dict]:
        """
        Fit 2D Gaussian to star/satellite profile to characterize PSF
        
        Returns:
            psf_model: 2D PSF kernel
            params: Fitted parameters (sigma_x, sigma_y, theta)
        """
        # Extract region around center
        y_start = max(0, y_center - radius)
        y_end = min(image.shape[0], y_center + radius)
        x_start = max(0, x_center - radius)
        x_end = min(image.shape[1], x_center + radius)
        
        region = image[y_start:y_end, x_start:x_end]
        
        # Simple moment-based Gaussian fitting
        # For production, use mpdaf.obj.Image.gauss_fit
        y_indices, x_indices = np.mgrid[0:region.shape[0], 0:region.shape[1]]
        
        # Normalize
        region_norm = region - np.min(region)
        total = np.sum(region_norm)
        if total == 0:
            total = 1.0
        
        # Centroid
        x_centroid = np.sum(x_indices * region_norm) / total
        y_centroid = np.sum(y_indices * region_norm) / total
        
        # Second moments (variance)
        x_var = np.sum((x_indices - x_centroid)**2 * region_norm) / total
        y_var = np.sum((y_indices - y_centroid)**2 * region_norm) / total
        xy_cov = np.sum((x_indices - x_centroid) * (y_indices - y_centroid) * region_norm) / total
        
        # Gaussian parameters
        sigma_x = np.sqrt(x_var)
        sigma_y = np.sqrt(y_var)
        
        # Handle elongated PSFs (seeing can be anisotropic)
        if sigma_x > sigma_y:
            sigma_major, sigma_minor = sigma_x, sigma_y
        else:
            sigma_major, sigma_minor = sigma_y, sigma_x
            
        # Rotation angle from covariance
        if abs(xy_cov) > 0.001:
            theta = 0.5 * np.arctan2(2 * xy_cov, x_var - y_var)
        else:
            theta = 0.0
        
        # Generate PSF kernel
        psf_size = 2 * radius + 1
        psf = self._generate_gaussian_psf(
            psf_size, sigma_major, sigma_minor, theta
        )
        
        params = {
            'sigma_major': sigma_major,
            'sigma_minor': sigma_minor,
            'theta': theta,
            'fwhm_major_arcsec': sigma_major * 2.355 * self.pixel_scale,
            'fwhm_minor_arcsec': sigma_minor * 2.355 * self.pixel_scale,
            'ellipticity': 1.0 - sigma_minor / sigma_major if sigma_major > 0 else 0.0,
            'strehl_ratio': self._estimate_strehl(region, psf)
        }
        
        logger.info(f"PSF FWHM: {params['fwhm_major_arcsec']:.2f} x {params['fwhm_minor_arcsec']:.2f} arcsec")
        logger.info(f"Strehl ratio: {params['strehl_ratio']:.3f}")
        
        return psf, params
    
    def _generate_gaussian_psf(
        self,
        size: int,
        sigma_x: float,
        sigma_y: float,
        theta: float
    ) -> np.ndarray:
        """Generate 2D rotated Gaussian PSF kernel"""
        # Create coordinate grid
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        
        # Rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        X_rot = X * cos_theta + Y * sin_theta
        Y_rot = -X * sin_theta + Y * cos_theta
        
        # Gaussian
        psf = np.exp(-0.5 * ((X_rot / sigma_x)**2 + (Y_rot / sigma_y)**2))
        
        # Normalize
        psf /= np.sum(psf)
        
        return psf
    
    def _estimate_strehl(self, observed: np.ndarray, psf: np.ndarray) -> float:
        """
        Estimate Strehl ratio (measure of optical quality)
        Strehl = peak intensity / peak of ideal diffraction-limited PSF
        """
        # Simplified: compare peak brightness to ideal Airy pattern
        # Real implementation would require known telescope aperture
        peak_observed = np.max(observed)
        peak_psf = np.max(psf)
        
        if peak_psf == 0:
            return 0.0
        
        # Approximate Strehl (simplified)
        strehl = peak_observed / (peak_psf * np.sum(observed))
        
        return min(strehl, 1.0)
    
    def create_seeing_limited_psf(self, seeing_fwhm_arcsec: float) -> np.ndarray:
        """
        Create theoretical PSF from seeing conditions
        
        Args:
            seeing_fwhm_arcsec: Atmospheric seeing FWHM (e.g., 1.5 arcsec for good site)
        """
        # Convert FWHM to sigma
        sigma_pixels = seeing_fwhm_arcsec / (2.355 * self.pixel_scale)
        
        # Create PSF
        psf_size = int(4 * sigma_pixels) * 2 + 1
        psf = self._generate_gaussian_psf(
            psf_size, sigma_pixels, sigma_pixels, 0.0
        )
        
        return psf


class RichardsonLucyDeconvolver:
    """
    Richardson-Lucy deconvolution for image sharpening
    
    Recovers high-resolution structure from atmospherically-blurred images
    """
    
    def __init__(
        self,
        n_iterations: int = 30,
        regularization: float = 0.001,
        clip_negative: bool = True
    ):
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.clip_negative = clip_negative
        
    def deconvolve(
        self,
        blurred_image: np.ndarray,
        psf: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> DeconvolutionResult:
        """
        Apply Richardson-Lucy deconvolution
        
        Args:
            blurred_image: Observed blurry image
            psf: Point Spread Function (normalized)
            initial_guess: Starting point for iteration (default: blurred_image)
            
        Returns:
            DeconvolutionResult with deconvolved image and metrics
        """
        # Normalize inputs
        blurred = blurred_image.astype(np.float64)
        blurred = (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred) + 1e-10)
        
        psf_norm = psf / np.sum(psf)
        
        # Initial guess
        if initial_guess is None:
            deconvolved = blurred.copy()
        else:
            deconvolved = initial_guess.astype(np.float64)
        
        # Richardson-Lucy iterations
        convergence_history = []
        
        for i in range(self.n_iterations):
            # Forward convolution: current guess → blurred prediction
            convolved = convolve2d(deconvolved, psf_norm, mode='same', boundary='symm')
            
            # Prevent division by zero
            convolved = np.maximum(convolved, self.regularization)
            
            # Relative blur image
            relative_blur = blurred / convolved
            
            # Backward convolution (with flipped PSF)
            psf_flipped = np.flip(psf_norm)
            error_estimate = convolve2d(relative_blur, psf_flipped, mode='same', boundary='symm')
            
            # Update
            deconvolved = deconvolved * error_estimate
            
            # Regularization to prevent noise amplification
            deconvolved = np.maximum(deconvolved, 0)
            
            # Track convergence
            if i % 5 == 0 or i == self.n_iterations - 1:
                # Calculate residual
                forward_model = convolve2d(deconvolved, psf_norm, mode='same', boundary='symm')
                residual = np.sum((blurred - forward_model)**2)
                convergence_history.append(residual)
                logger.info(f"RL iteration {i+1}/{self.n_iterations}, residual: {residual:.6f}")
        
        # Calculate sharpness metric
        sharpness = self._calculate_sharpness(deconvolved)
        
        # Estimate recovered resolution
        recovered_res = self._estimate_resolution(deconvolved, psf_norm)
        
        return DeconvolutionResult(
            original_image=blurred_image,
            deconvolved_image=deconvolved,
            psf_model=psf_norm,
            sharpness_metric=sharpness,
            recovered_resolution=recovered_res,
            convergence_history=convergence_history
        )
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance
        Higher values indicate sharper image
        """
        from scipy.ndimage import laplace
        
        laplacian = laplace(image)
        sharpness = np.var(laplacian)
        
        return sharpness
    
    def _estimate_resolution(
        self,
        deconvolved: np.ndarray,
        psf: np.ndarray
    ) -> float:
        """
        Estimate recovered angular resolution
        
        Uses power spectrum analysis to estimate effective resolution
        """
        # FFT of deconvolved image
        fft_image = np.fft.fft2(deconvolved)
        power_spectrum = np.abs(fft_image)**2
        
        # FFT of PSF (MTF)
        fft_psf = np.fft.fft2(psf)
        mtf = np.abs(fft_psf)
        
        # Find frequency where MTF drops to 0.1 (10% contrast)
        # This gives approximate resolution limit
        mtf_1d = np.mean(mtf, axis=0)
        
        # Find 10% point
        threshold = 0.1 * np.max(mtf_1d)
        high_freq_indices = np.where(mtf_1d < threshold)[0]
        
        if len(high_freq_indices) > 0:
            # Convert to angular frequency
            nyquist_freq = 0.5 / self.pixel_scale  # cycles per arcsec
            cutoff_freq = high_freq_indices[0] / len(mtf_1d) * nyquist_freq
            
            # Resolution = 1 / cutoff frequency
            resolution = 1.0 / cutoff_freq if cutoff_freq > 0 else self.pixel_scale
        else:
            resolution = self.pixel_scale
        
        return resolution


class SuperResolutionProcessor:
    """
    Super-resolution processing for IAF optical sensors
    Combines multiple low-resolution frames to create high-resolution image
    """
    
    def __init__(self, upsampling_factor: int = 2):
        self.upsampling_factor = upsampling_factor
        
    def process_image_sequence(
        self,
        image_sequence: list,
        shifts: Optional[list] = None
    ) -> np.ndarray:
        """
        Create super-resolution image from sequence of slightly shifted images
        
        Args:
            image_sequence: List of low-resolution images
            shifts: List of (dx, dy) shifts for each image (optional, estimated if None)
            
        Returns:
            High-resolution super-resolved image
        """
        if len(image_sequence) < 2:
            logger.warning("Need at least 2 images for super-resolution")
            return image_sequence[0] if image_sequence else None
        
        # Reference image
        ref_image = image_sequence[0]
        
        # Create high-resolution grid
        hr_shape = (
            ref_image.shape[0] * self.upsampling_factor,
            ref_image.shape[1] * self.upsampling_factor
        )
        hr_image = np.zeros(hr_shape)
        hr_weights = np.zeros(hr_shape)
        
        # Estimate shifts if not provided
        if shifts is None:
            shifts = self._estimate_shifts(image_sequence)
        
        # Combine images
        for img, (dx, dy) in zip(image_sequence, shifts):
            # Upsample image
            upsampled = self._upsample_image(img)
            
            # Shift according to estimated displacement
            shifted = self._shift_image(upsampled, dx * self.upsampling_factor, dy * self.upsampling_factor)
            
            # Add to accumulation
            hr_image += shifted
            hr_weights += (shifted > 0).astype(float)
        
        # Normalize
        hr_image = np.divide(hr_image, hr_weights, where=hr_weights > 0)
        
        return hr_image
    
    def _estimate_shifts(self, image_sequence: list) -> list:
        """Estimate sub-pixel shifts between images using phase correlation"""
        from scipy.ndimage import fourier_shift
        
        shifts = [(0.0, 0.0)]  # First image is reference
        
        ref_image = image_sequence[0]
        
        for i in range(1, len(image_sequence)):
            # Phase correlation
            shift, error, diffphase = self._phase_correlation(ref_image, image_sequence[i])
            shifts.append(shift)
        
        return shifts
    
    def _phase_correlation(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[float, float]:
        """Calculate sub-pixel shift using phase correlation"""
        # FFT both images
        fft1 = np.fft.fft2(image1)
        fft2 = np.fft.fft2(image2)
        
        # Cross-power spectrum
        cross_power = (fft1 * np.conj(fft2)) / (np.abs(fft1) * np.abs(fft2) + 1e-10)
        
        # Inverse FFT to get correlation
        correlation = np.fft.ifft2(cross_power)
        correlation = np.fft.fftshift(correlation)
        
        # Find peak
        max_idx = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
        
        # Convert to shift
        dy = max_idx[0] - correlation.shape[0] // 2
        dx = max_idx[1] - correlation.shape[1] // 2
        
        return (dx, dy), 0.0, 0.0
    
    def _upsample_image(self, image: np.ndarray) -> np.ndarray:
        """Bicubic upsampling"""
        from scipy.ndimage import zoom
        return zoom(image, self.upsampling_factor, order=3)
    
    def _shift_image(self, image: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Shift image by sub-pixel amount using Fourier method"""
        from scipy.ndimage import fourier_shift
        
        shift = fourier_shift(np.fft.fftn(image), (dy, dx))
        return np.fft.ifftn(shift).real


# Example usage
def sharpen_satellite_image(
    blurry_image: np.ndarray,
    pixel_scale_arcsec: float = 0.5,
    seeing_fwhm_arcsec: float = 1.5
) -> DeconvolutionResult:
    """
    Complete pipeline: Characterize PSF and deconvolve satellite image
    
    Args:
        blurry_image: Blurry satellite image from IAF sensor
        pixel_scale_arcsec: Pixel scale (arcsec/pixel)
        seeing_fwhm_arcsec: Estimated atmospheric seeing
    """
    # Find satellite centroid
    y_center, x_center = np.array(blurry_image.shape) // 2
    
    # Model PSF
    psf_modeler = PSFModeler(pixel_scale_arcsec)
    
    # Use theoretical seeing-limited PSF (or fit to nearby star)
    psf, psf_params = psf_modeler.fit_gaussian_psf(
        blurry_image, x_center, y_center, radius=15
    )
    
    # Deconvolve
    deconvolver = RichardsonLucyDeconvolver(n_iterations=25)
    result = deconvolver.deconvolve(blurry_image, psf)
    
    logger.info(f"Resolution improved from {seeing_fwhm_arcsec:.1f} to {result.recovered_resolution:.1f} arcsec")
    logger.info(f"Sharpness increased by {result.sharpness_metric / np.var(blurry_image):.2f}x")
    
    return result


if __name__ == "__main__":
    # Create synthetic test: sharp satellite convolved with atmospheric PSF
    size = 128
    
    # Create sharp satellite image (two solar panels + body)
    sharp = np.zeros((size, size))
    center = size // 2
    
    # Satellite body
    sharp[center-3:center+4, center-8:center+8] = 1.0
    
    # Solar panels
    sharp[center-12:center-8, center-15:center+15] = 0.8
    sharp[center+8:center+12, center-15:center+15] = 0.8
    
    # Antenna
    sharp[center-15:center-12, center-1:center+2] = 0.6
    
    # Add noise
    sharp += np.random.normal(0, 0.05, sharp.shape)
    
    # Apply atmospheric blur (PSF convolution)
    psf_size = 21
    sigma = 3.0  # pixels
    x = np.arange(psf_size) - psf_size // 2
    X, Y = np.meshgrid(x, x)
    psf = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
    psf /= np.sum(psf)
    
    from scipy.signal import convolve2d
    blurry = convolve2d(sharp, psf, mode='same', boundary='symm')
    
    # Deconvolve
    result = sharpen_satellite_image(
        blurry,
        pixel_scale_arcsec=0.5,
        seeing_fwhm_arcsec=3.0 * 0.5  # sigma=3 pixels * 0.5 arcsec/pixel
    )
    
    print(f"Original resolution: 1.5 arcsec")
    print(f"Recovered resolution: {result.recovered_resolution:.2f} arcsec")
    print(f"Sharpness improvement: {result.sharpness_metric:.3f}")
