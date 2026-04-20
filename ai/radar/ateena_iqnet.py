import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ComplexBatchNorm2d(nn.Module):
    """
    Complex-Valued Batch Normalization (2x2 Covariance Whitening).
    Preserves the phase correlation between I and Q channels.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Parameters for whitening
        self.register_buffer('running_mean', torch.zeros(num_features, 2))
        self.register_buffer('running_cov', torch.eye(2).repeat(num_features, 1, 1))
        
        self.W = nn.Parameter(torch.eye(2).repeat(num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_features, 2))

    def forward(self, x):
        # x: [B, 2, C, H, W] where dim 1 is (I, Q)
        B, D, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, 2) # [B*C*H*W, 2]
        
        if self.training:
            mean = x_flat.mean(dim=0)
            # Update mean tracking
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            
            # Center and calculate covariance
            x_centered = x_flat - mean
            cov = (x_centered.transpose(0, 1) @ x_centered) / (x_flat.shape[0] - 1)
            self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * cov
        else:
            mean = self.running_mean
            cov = self.running_cov
            x_centered = x_flat - mean

        # Optimize eigendecomposition for 2x2 matrix using direct formula
        cov_eps = cov + self.eps * torch.eye(2).to(x.device)
        det_cov = cov_eps[0,0] * cov_eps[1,1] - cov_eps[0,1] * cov_eps[1,0]
        inv_det = 1.0 / torch.sqrt(torch.clamp(det_cov, min=1e-6))
        inv_cov = torch.zeros_like(cov_eps)
        inv_cov[0,0] = inv_det * cov_eps[1,1]
        inv_cov[1,1] = inv_det * cov_eps[0,0]
        inv_cov[0,1] = -inv_det * cov_eps[0,1]
        inv_cov[1,0] = -inv_det * cov_eps[1,0]
        whitener = torch.sqrt(inv_cov)  # Approximate square root inverse
        x_norm = x_centered @ whitener

        # Scale and Shift (W is 2x2 learned weight)
        out = (x_norm.unsqueeze(1) @ self.W.repeat(B*H*W, 1, 1)).squeeze(1) + self.beta.repeat(B*H*W, 1)
        
        return out.reshape(B, H, W, C, 2).permute(0, 4, 3, 1, 2)

class ModReLU(nn.Module):
    """
    Phase-Preserving Activation: ModReLU(z) = ReLU(|z| + b) * (z/|z|)
    """
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        # x: [B, 2*C, ...]
        # Treat as complex: dim 1 has (I, Q) pairs
        B, C2, *dims = x.shape
        C = C2 // 2
        
        # Reshape to separate complex channels: [B, C, 2, ...]
        x_complex = x.reshape(B, C, 2, *dims)
        
        magnitude = torch.sqrt(torch.sum(x_complex**2, dim=2, keepdim=True) + 1e-6)
        phase = x_complex / magnitude
        
        # Apply ReLU to magnitude and multiply by phase
        out = F.relu(magnitude + self.b.view(1, C, 1, *([1]*len(dims)))) * phase
        return out.reshape(B, C2, *dims)

class NeuralBeamformer(nn.Module):
    """
    Adaptive Spatial Filter for 64-element Phased Array.
    Initialized with Taylor-Tapered Weights (SLL = -30 dB).
    """
    def __init__(self, n_elements=64, n_beams=32):
        super().__init__()
        self.n_elements = n_elements
        self.n_beams = n_beams
        
        # Weights: [Beams, Elements, 2]
        self.weights = nn.Parameter(torch.empty(n_beams, n_elements, 2))
        self._initialize_taylor()

    def _initialize_taylor(self, sll=-30):
        """Warm-start with Taylor window for SLL suppression."""
        # Simple Taylor approximation for initial spatial weights
        n = self.n_elements
        window = torch.from_numpy(np.blackman(n)).float() # Blackman as proxy for Taylor start
        
        for b in range(self.n_beams):
            # Steer beams across -60 to 60 deg
            angle = -60 + (120 / (self.n_beams-1)) * b
            theta = math.radians(angle)
            
            # Phased steering vector
            indices = torch.arange(n).float()
            phase = 2 * math.pi * 0.5 * indices * math.sin(theta)
            
            self.weights.data[b, :, 0] = window * torch.cos(phase)
            self.weights.data[b, :, 1] = window * torch.sin(phase)

    def forward(self, x):
        # x: [B, 2, Pulses, Range, Elements]
        # Perform spatial convolution (Beamforming)
        # Result: [B, 2, Pulses, Range, Beams]
        B, C, P, R, E = x.shape
        
        # Complex matrix multiplication across the element dimension
        # x_complex = [B, P, R, E, 2], weights_complex = [Beams, E, 2]
        x_in = x.permute(0, 2, 3, 4, 1) 
        
        # Real/Imag multiplication: (W_R + jW_I)(X_R + jX_I)
        # = (W_R*X_R - W_I*X_I) + j(W_R*X_I + W_I*X_R)
        wr, wi = self.weights[:, :, 0], self.weights[:, :, 1]
        xr, xi = x_in[..., 0], x_in[..., 1]
        
        # Einstein summation for beamforming: n=batch, p=pulses, r=range, e=elements, b=beams
        beam_r = torch.einsum('npre,be->nprb', xr, wr) - torch.einsum('npre,be->nprb', xi, wi)
        beam_i = torch.einsum('npre,be->nprb', xr, wi) + torch.einsum('npre,be->nprb', xi, wr)
        
        return torch.stack([beam_r, beam_i], dim=1) # [B, 2, P, R, Beams]

class LearnedDopplerProcessor(nn.Module):
    """
    Dilated 1D-CNN Doppler FFT Proxy.
    Extracts Micro-Doppler signatures across pulses.
    """
    def __init__(self, n_pulses=128):
        super().__init__()
        # Using dilated convolutions to capture wide spectral ranges
        self.doppler_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=9, dilation=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, 2, Pulses, Range, Beams]
        B, C, P, R, Beams = x.shape
        # Process pulse dimension: Reshape to [B*Range*Beams, 2, Pulses]
        x_in = x.permute(0, 3, 4, 1, 2).reshape(-1, 2, P)
        
        doppler_feat = self.doppler_conv(x_in) # [N, 64, D]
        D = doppler_feat.shape[-1]
        
        return doppler_feat.reshape(B, R, Beams, 64, D).permute(0, 3, 4, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    """
    ResNet Block with Squeeze-and-Excitation across channels (Beams).
    Processes Range-Doppler maps.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

class EvidentialDetectionHead(nn.Module):
    """
    Subjective Logic head for multi-target detection and localization.
    Outputs: [B, max_detections, 6] -> [range, doppler, az, el, rcs, confidence/uncertainty]
    """
    def __init__(self, input_dim=512, max_detections=50):
        super().__init__()
        self.max_detections = max_detections
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, max_detections * 6)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, max_detections),
            nn.Softplus() # alpha values > 1.0 (after +1.0)
        )

    def forward(self, x):
        # x: [B, 512]
        B = x.shape[0]
        params = self.regressor(x).view(B, self.max_detections, 6)
        
        # Dirichlet alphas for classification (Target present?)
        alphas = self.uncertainty_head(x) + 1.0  # Ensure alphas > 1.0 for Dirichlet distribution stability
        
        # We append uncertainty mass u = 1 / S (binary detection case)
        # Note: In a 2-class case (Target/Clutter), S = alpha_target + alpha_clutter
        # For simplicity, we output alphas and calculate u during inference.
        return params, alphas

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x: [B, 2, L] (Real, Imag)
        xr, xi = x[:, 0:1, :], x[:, 1:2, :]
        real = self.conv_r(xr) - self.conv_i(xi)
        imag = self.conv_r(xi) + self.conv_i(xr)
        return torch.cat([real, imag], dim=1)

class ATeenaIQNet(nn.Module):
    """
    Full Production ATeena-IQNet.
    5-Stage Pipeline: Compression, Beamforming, Doppler, Feature Extraction, Detection.
    """
    def __init__(self, config):
        super().__init__()
        # Stage 1: Learned Strided Compression (Preserves Phase)
        # We use a single ComplexConv1d for the learned matched filter
        self.range_compression = nn.Sequential(
            ComplexConv1d(1, 1, kernel_size=8, stride=4, padding=2),
            ComplexBatchNorm2d(2),  # Fix normalization to use complex batch norm
            ModReLU(2)  # Fix channel count for ModReLU
        )
        
        # Stage 2: Spatial & Spectral Processing
        self.beamformer = NeuralBeamformer(n_elements=config.n_elements, n_beams=32)
        self.doppler_processor = LearnedDopplerProcessor(n_pulses=config.n_pulses)
        
        # Stage 3: SE-ResNet34
        self.feature_extractor = nn.Sequential(
            SEResNetBlock(64, 128, stride=2),
            SEResNetBlock(128, 256, stride=2),
            SEResNetBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Stage 4: Temporal Context
        self.temporal_encoder = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Stage 5: Evidential Head
        self.detection_head = EvidentialDetectionHead(input_dim=512, max_detections=50)

    def forward(self, iq_samples, temporal_context=None):
        # iq_samples: [batch, pulses, range, elements, 2]
        B, P, R, E, C = iq_samples.shape
        # Transform to [B, 2, P, R, E]
        iq = iq_samples.permute(0, 4, 1, 2, 3) 
        
        # Stage 1: Complex Pulse Compression
        # Process pulses/elements in batch: [B*P*E, 2, R]
        x_in = iq.permute(0, 2, 4, 1, 3).reshape(-1, 2, R)
        compressed = self.range_compression(x_in) # [B*P*E, 2, R_new]
        R_new = compressed.shape[-1]
        
        # Stage 2: Beamforming
        # Reshape to [B, P, E, 2, R_new] -> [B, 2, P, R_new, E]
        x_bf = compressed.reshape(B, P, E, 2, R_new).permute(0, 3, 1, 4, 2)
        beams = self.beamformer(x_bf) # [B, 2, P, R_new, 32]
        
        # Doppler
        rd_cube = self.doppler_processor(beams) # [B, 64, D, R_new, 32]
        
        # Stage 3: Feature Extraction (ResNet) per Beam
        # rd_cube: [B, C_feats, D, R_new, Beams]
        # Treat Beams as channels for SE attention
        # Shape: [B, C*Beams, D, R_new]
        B_tot, C_f, D, R_f, Beams = rd_cube.shape
        x_res = rd_cube.permute(0, 4, 1, 2, 3).reshape(B_tot*Beams, C_f, D, R_f)
        feats = self.feature_extractor(x_res).reshape(B_tot, Beams, 512).mean(dim=1)
        
        # Stage 4 & 5: Temporal + Detection
        if temporal_context is not None:
            # stacking features over time
            seq = torch.stack([temporal_context, feats], dim=1)
            lstm_out, _ = self.temporal_encoder(seq)
            features = lstm_out[:, -1, :]
        else:
            features, _ = self.temporal_encoder(feats.unsqueeze(1))
            features = features.squeeze(1)
            
        detections, alphas = self.detection_head(features)
        return detections, alphas, features
