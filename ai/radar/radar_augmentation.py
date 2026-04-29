import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RadarAugmentation:
    """
    Sovereign Electronic Warfare Augmentation for ATeena-IQNet.
    Includes: 
    - Coherent DRFM Spoofing
    - Pulse-to-Pulse Phase Agility
    - Ground Clutter & Deceptive Jamming
    - Platform Jitter (Phase Noise)
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.noise_range = (-20, 10) # dB SNR

    def add_noise(self, iq, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(torch.sum(iq**2, dim=1))
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(iq) * torch.sqrt(noise_power / 2.0)
        return iq + noise

    def add_drfm_spoof(self, iq, alpha=0.5, delay_bins=5, doppler_shift=0.1):
        """
        Coherent Digital Radio Frequency Memory (DRFM) Spoofing.
        Replays the pulse with a delay and a systematic phase shift.
        """
        # iq: [B, 2, P, R, E]
        spoof = torch.roll(iq, shifts=delay_bins, dims=3)
        # Apply Doppler phase shift: exp(j * 2pi * fd * t)
        B, C, P, R, E = iq.shape
        t = torch.arange(P).to(self.device).float()
        phase = 2 * np.pi * doppler_shift * t
        
        rot_r = torch.cos(phase).view(1, 1, P, 1, 1)
        rot_i = torch.sin(phase).view(1, 1, P, 1, 1)
        
        spoof_r = spoof[:, 0:1] * rot_r - spoof[:, 1:2] * rot_i
        spoof_i = spoof[:, 0:1] * rot_i + spoof[:, 1:2] * rot_r
        
        return iq + alpha * torch.cat([spoof_r, spoof_i], dim=1)

    def add_phase_jitter(self, iq, jitter_deg=5.0):
        """
        Simulates platform jitter or hardware phase noise.
        """
        B, C, P, R, E = iq.shape
        jitter_rad = np.radians(jitter_deg)
        noise = torch.randn(B, 1, P, 1, E).to(self.device) * jitter_rad
        
        rot_r = torch.cos(noise)
        rot_i = torch.sin(noise)
        
        iq_r = iq[:, 0:1] * rot_r - iq[:, 1:2] * rot_i
        iq_i = iq[:, 0:1] * rot_i + iq[:, 1:2] * rot_r
        
        return torch.cat([iq_r, iq_i], dim=1)

    def add_ground_clutter(self, iq, power=0.3):
        """
        Simulates strong, low-Doppler clutter at close ranges.
        """
        B, C, P, R, E = iq.shape
        clutter_range = R // 10
        clutter = torch.randn(B, 2, P, clutter_range, E).to(self.device) * power
        
        # Apply low-pass filter across pulses (stationary ground)
        # In a real sim, we'd use a more complex Rayleigh model
        iq_aug = iq.clone().to(self.device)
        iq_aug[:, :, :, :clutter_range, :] += clutter
        return iq_aug

    def __call__(self, iq):
        # iq: [B, 128, 1024, 64, 2] -> Transform to [B, 2, P, R, E] if needed
        # We assume training iq is already in [B, 2, P, R, E]
        
        snr = random.uniform(*self.noise_range)
        iq = self.add_noise(iq, snr)
        
        if random.random() < 0.3:
            iq = self.add_drfm_spoof(iq)
        
        if random.random() < 0.5:
            iq = self.add_ground_clutter(iq)
            
        iq = self.add_phase_jitter(iq, jitter_deg=random.uniform(0.1, 2.0))
        
        return iq
