import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable

/**
 * IQNet (v2) implements a 2D-CNN architecture for Range-Doppler sensing.
 * By processing the CPI as a two-dimensional image (Pulses x Samples), 
 * the network can extract coherent Doppler innovations directly via spatial 
 * kernels, providing superior detection at low SNR (-5 dB target).
 * 
 * <plan_complexity>High</plan_complexity><total_lines_estimate>180</total_lines_estimate>
 */

class IQNet(nn.Module):
    """
    ATeena-IQNet 2.0: Physics-native 2D-CNN substrate.
    
    Architecture:
    1. 2D-CNN: Joint Range-Doppler feature extraction.
    2. ResNet Blocks: Deep coherent feature preservation.
    3. Global Average Pooling: Translation invariance across bins.
    4. Dense Head: Multi-task regression for state and detectability.
    """
    num_pulses: int = 128
    num_samples: int = 1024
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Forward pass for the 2D-CNN IQNet.
        
        @param x: (Batch, NumPulses, NumSamples, 2) [I, Q Channels]
        @param train: Training mode for Dropout/BatchNorm context
        @returns: (Batch, 6) predicted state vectors
        """
        # Entry Point: 2D-CNN Processing
        # Initial convolution to capture fast-time/slow-time correlations
        # Kernel (3, 7) captures Doppler (slow-time) and Range (fast-time)
        x = nn.Conv(features=32, kernel_size=(3, 7), strides=(1, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # ResNet Block 1
        residual = nn.Conv(features=64, kernel_size=(1, 1), strides=(2, 2))(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = x + residual
        
        # ResNet Block 2 (Deep Coherent Search)
        residual = nn.Conv(features=128, kernel_size=(1, 1), strides=(2, 2))(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = x + residual
        
        # Stage 3: Feature Aggregation
        # Flattening 2D maps into global context
        x = jnp.mean(x, axis=(1, 2)) # Global Average Pooling
        
        # Stage 4: Multi-Task Head
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.3, deterministic=not train)(x)
        
        # Final Output Head: [p_det, r, v, az, el, rcs]
        # Kendall & GalLearned Uncertainty weighting should be applied in the loss function
        out = nn.Dense(features=6)(x)
        
        # Exit Point: Prediction complete
        return out

def create_model(num_pulses=128, num_samples=1024):
    """
    Factory function for IQNet v2.
    """
    return IQNet(num_pulses=num_pulses, num_samples=num_samples)
