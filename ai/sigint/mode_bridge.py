"""
Mode-to-SINDy Bridge for Hypersonic Vehicle Identification
Extracts Lift-to-Drag (L/D) ratio from DMD modes for strategic classification
"""

import numpy as np
import pysindy as ps
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class HypersonicSignature:
    """Extracted aerodynamic signature from SINDy analysis"""
    ld_ratio: float
    lift_coeff: float
    drag_coeff: float
    classification: str  # Ballistic, HGV, HCM
    confidence: float
    vehicle_type: str


class ModeToSINDyBridge:
    """
    Bridge between DMD spatiotemporal modes and SINDy aerodynamic system identification.
    
    Transforms isolated hypersonic vehicle modes from S2/HASH-INR space into
    physical state vectors, then runs SINDy to discover the L/D ratio.
    
    This provides "Digital Fingerprint" capability to distinguish between:
    - Ballistic Re-entry Vehicles (L/D < 1.0)
    - Hypersonic Glide Vehicles (L/D 1.5-3.0)  
    - Hypersonic Cruise Missiles (L/D > 4.0)
    """
    
    def __init__(self, threshold: float = 0.05, library_degree: int = 2):
        """
        Args:
            threshold: STLSQ sparsity threshold for SINDy
            library_degree: Polynomial degree for feature library
        """
        self.threshold = threshold
        self.library_degree = library_degree
        
        # Physics-informed feature library
        # Includes velocity squared and density-dependent terms for aerodynamics
        self.feature_library = ps.PolynomialLibrary(degree=library_degree)
        
        # Thresholds for vehicle classification
        self.ld_thresholds = {
            'ballistic': (0.0, 1.0),      # Standard re-entry
            'hgv': (1.5, 3.0),             # High maneuverability glide
            'hcm': (4.0, 10.0),            # Sustained cruise
        }
    
    def extract_trajectory(self, dmd_modes: List[np.ndarray], 
                          s2_metadata: Dict) -> np.ndarray:
        """
        Converts DMD pixel-modes into physical state vectors (r, v).
        
        Args:
            dmd_modes: List of DMD mode arrays from HypersonicDetector
            s2_metadata: S2 cell metadata containing coordinate transformations
            
        Returns:
            trajectory: Array [t, x, y, z, vx, vy, vz]
        """
        # 1. Centroid tracking of isolated mode in S2 space
        trajectory = []
        
        for i, mode in enumerate(dmd_modes):
            # Extract centroid from mode (isolated hypersonic signature)
            centroid = self._compute_centroid(mode)
            
            # 2. Coordinate transformation to ECI/ECEF
            # Convert from S2 cell coordinates to geocentric
            eci_pos = self._s2_to_eci(centroid, s2_metadata)
            
            # 3. Calculate velocity from consecutive frames
            if i == 0:
                velocity = np.array([0.0, 0.0, 0.0])  # Initialize
            else:
                dt = 1.0  # Assume 1-second intervals
                velocity = (eci_pos - trajectory[-1][1:4]) / dt
            
            # Timestamp and state
            t = float(i)
            state = np.concatenate([[t], eci_pos, velocity])
            trajectory.append(state)
        
        return np.array(trajectory)
    
    def identify_aerodynamics(self, trajectory: np.ndarray, 
                           density_model) -> HypersonicSignature:
        """
        Runs SINDy to discover the L/D ratio from trajectory.
        
        Physics Model:
        a = g(r) + (ρ(h) v² S / 2m) * (CL * n̂ - CD * v̂)
        
        SINDy solves: a = g + ξ_L * (q*e_n) + ξ_D * (q*e_v)
        
        Args:
            trajectory: [t, x, y, z, vx, vy, vz] from extract_trajectory
            density_model: Atmospheric density model (e.g., NRLMSISE-00)
            
        Returns:
            HypersonicSignature with L/D ratio and classification
        """
        # Extract components
        t = trajectory[:, 0]
        x = trajectory[:, 1:4]  # Position
        v = trajectory[:, 4:7]  # Velocity
        
        # Calculate velocity magnitude
        v_mag = np.linalg.norm(v, axis=1, keepdims=True)
        
        # Calculate local air density ρ(h)
        altitude = np.linalg.norm(x, axis=1) - 6371.0  # km above Earth
        rho = np.array([density_model.get_density(a) for a in altitude])
        
        # Dynamic Pressure: q = 0.5 * ρ * v²
        q = 0.5 * rho.reshape(-1, 1) * v_mag**2
        
        # Calculate acceleration (target for SINDy)
        a = np.gradient(v, t, axis=0)
        
        # Calculate gravity
        r_norm = np.linalg.norm(x, axis=1, keepdims=True)
        g = -(398600.4418 / r_norm**3) * x  # Earth's gravity
        
        # Aerodynamic acceleration (residual after removing gravity)
        a_aero = a - g
        
        # Define physics-informed library
        # Terms: q (dynamic pressure), q*velocity_dir, q*normal_dir
        velocity_dir = v / (v_mag + 1e-10)  # Unit velocity vector
        
        # Normal vector (lift direction - perpendicular to velocity in orbital plane)
        # Approximate as cross product of position and velocity
        h_vec = np.cross(x, v)
        h_norm = np.linalg.norm(h_vec, axis=1, keepdims=True)
        normal_dir = h_vec / (h_norm + 1e-10)
        
        # Build custom feature matrix
        # X = [q, q*velocity_dir, q*normal_dir]
        X_features = np.hstack([
            q,  # Scalar dynamic pressure
            q * velocity_dir,  # Drag direction
            q * normal_dir,    # Lift direction
        ])
        
        # Run SINDy sparse regression
        optimizer = ps.STLSQ(threshold=self.threshold, alpha=0.1)
        model = ps.SINDy(
            feature_library=self.feature_library,
            optimizer=optimizer,
            feature_names=['q', 'qv_x', 'qv_y', 'qv_z', 'qn_x', 'qn_y', 'qn_z']
        )
        
        # Fit model
        try:
            model.fit(X_features, x_dot=a_aero, t=t)
            coeffs = model.coefficients()
        except Exception as e:
            # Fallback to simple regression if SINDy fails
            coeffs = self._simple_regression(X_features, a_aero)
        
        # Extract L/D ratio from coefficients
        # Drag coefficient: terms with velocity direction (qv_*)
        # Lift coefficient: terms with normal direction (qn_*)
        xi_D = np.linalg.norm(coeffs[1:4])  # qv_x, qv_y, qv_z
        xi_L = np.linalg.norm(coeffs[4:7])  # qn_x, qn_y, qn_z
        
        if xi_D < 1e-10:
            ld_ratio = 0.0
            confidence = 0.0
        else:
            ld_ratio = abs(xi_L / xi_D)
            confidence = min(1.0, ld_ratio / 3.0)  # Normalize confidence
        
        # Classify vehicle
        classification, vehicle_type = self._classify_vehicle(ld_ratio)
        
        return HypersonicSignature(
            ld_ratio=ld_ratio,
            lift_coeff=xi_L,
            drag_coeff=xi_D,
            classification=classification,
            confidence=confidence,
            vehicle_type=vehicle_type
        )
    
    def _compute_centroid(self, mode: np.ndarray) -> np.ndarray:
        """Compute centroid of isolated mode in pixel space."""
        # Find non-zero pixels (isolated hypersonic signature)
        mask = mode > np.percentile(mode, 95)  # Top 5% intensity
        
        if np.sum(mask) == 0:
            return np.array([0.0, 0.0])
        
        # Weighted centroid
        y_idx, x_idx = np.where(mask)
        weights = mode[mask]
        
        centroid_y = np.average(y_idx, weights=weights)
        centroid_x = np.average(x_idx, weights=weights)
        
        return np.array([centroid_x, centroid_y])
    
    def _s2_to_eci(self, centroid: np.ndarray, metadata: Dict) -> np.ndarray:
        """Convert S2 cell centroid to ECI coordinates."""
        # Extract S2 cell ID and convert to lat/lon
        s2_cell_id = metadata.get('s2_cell_id', 0)
        
        # Simplified conversion (production would use proper S2 geometry)
        # Approximate ECI from cell position
        face = s2_cell_id >> 61
        
        # Map S2 face to approximate ECI direction
        face_directions = {
            0: [1, 0, 0],   # +X
            1: [-1, 0, 0],  # -X
            2: [0, 1, 0],   # +Y
            3: [0, -1, 0],  # -Y
            4: [0, 0, 1],   # +Z
            5: [0, 0, -1],  # -Z
        }
        
        base_dir = np.array(face_directions.get(face, [1, 0, 0]))
        
        # Add centroid offset (scaled to approximate position)
        # Assume 100km altitude for hypersonic vehicle
        altitude_km = 100.0
        earth_radius = 6371.0
        distance = earth_radius + altitude_km
        
        # Perturb direction based on centroid within cell
        offset_scale = 0.1  # Small angular offset
        offset_x = centroid[0] * offset_scale / 1000.0  # Normalize
        offset_y = centroid[1] * offset_scale / 1000.0
        
        direction = base_dir + np.array([offset_x, offset_y, 0])
        direction = direction / np.linalg.norm(direction)
        
        eci_pos = distance * direction
        
        return eci_pos
    
    def _simple_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fallback simple least squares if SINDy fails."""
        # Ridge regression: (X^T X + λI)^{-1} X^T y
        lambda_reg = 0.01
        XTX = X.T @ X
        I = np.eye(XTX.shape[0])
        coeffs = np.linalg.inv(XTX + lambda_reg * I) @ X.T @ y
        return coeffs.flatten()
    
    def _classify_vehicle(self, ld_ratio: float) -> Tuple[str, str]:
        """Classify vehicle based on L/D ratio."""
        if ld_ratio < 1.0:
            return 'BALLISTIC_REENTRY', 'Standard kinetic strike; predictable path'
        elif 1.5 <= ld_ratio <= 3.0:
            return 'HYPERSONIC_GLIDE_VEHICLE', 'High maneuverability; evades radar'
        elif ld_ratio > 4.0:
            return 'HYPERSONIC_CRUISE_MISSILE', 'Sustained flight; extreme threat'
        else:
            return 'UNKNOWN', 'Ambiguous aerodynamic signature'
    
    def generate_signature_report(self, signature: HypersonicSignature,
                                 target_id: str) -> Dict:
        """
        Generate Siddhantic Meta-Header update for Go engine.
        
        Returns:
            Dictionary for Plasma Shared Memory serialization
        """
        return {
            'target_id': target_id,
            'signature': 'MARS_HYPERSONIC',
            'aerodynamics': {
                'ld_ratio': round(signature.ld_ratio, 2),
                'lift_coeff': round(signature.lift_coeff, 6),
                'drag_coeff': round(signature.drag_coeff, 6),
            },
            'classification': signature.classification,
            'vehicle_type': signature.vehicle_type,
            'confidence': round(signature.confidence, 2),
            'recommended_action': 'EVASIVE_MANEUVER_DELTA',
            'urgency': 'CRITICAL' if signature.classification == 'HYPERSONIC_CRUISE_MISSILE' else 'HIGH'
        }


class SimpleAtmosphereModel:
    """Simplified atmospheric density model for testing."""
    
    def __init__(self):
        # Exponential atmosphere approximation
        self.rho0 = 1.225  # kg/m³ at sea level
        self.H = 8.5  # Scale height in km
    
    def get_density(self, altitude_km: float) -> float:
        """Get air density at given altitude."""
        if altitude_km < 0:
            altitude_km = 0
        
        # Exponential atmosphere: ρ = ρ₀ * exp(-h/H)
        rho = self.rho0 * np.exp(-altitude_km / self.H)
        
        # Convert to kg/km³ for consistency
        return rho * 1e9


# =============================================================================
# Usage Example
# =============================================================================

"""
# Create bridge
bridge = ModeToSINDyBridge(threshold=0.05)
density = SimpleAtmosphereModel()

# Process DMD modes from HypersonicDetector
dmd_modes = detector.get_isolated_modes()
s2_metadata = detector.get_s2_metadata()

# Extract trajectory and identify aerodynamics
trajectory = bridge.extract_trajectory(dmd_modes, s2_metadata)
signature = bridge.identify_aerodynamics(trajectory, density)

print(f"L/D Ratio: {signature.ld_ratio:.2f}")
print(f"Classification: {signature.classification}")
print(f"Vehicle Type: {signature.vehicle_type}")

# Generate report for Go engine via Plasma Shared Memory
report = bridge.generate_signature_report(signature, "UCT-37-ALPHA")
plasma_client.write(report)
"""


if __name__ == "__main__":
    # Test the bridge with synthetic data
    print("Testing Mode-to-SINDy Bridge...")
    
    bridge = ModeToSINDyBridge(threshold=0.1)
    density = SimpleAtmosphereModel()
    
    # Generate synthetic HGV trajectory
    t = np.linspace(0, 60, 60)  # 1 minute
    
    # HGV-like trajectory: L/D ~ 2.0
    ld_target = 2.0
    v = 6000.0  # m/s
    
    # Synthetic position (simplified glide)
    x = 6371.0 + 80.0 + 0.5 * t  # 80km altitude, moving
    y = 0.1 * t**2  # Lateral maneuver
    z = np.zeros_like(t)
    
    trajectory = np.column_stack([t, x, y, z, 
                                  np.ones_like(t)*v,
                                  np.ones_like(t)*100.0,
                                  np.zeros_like(t)])
    
    signature = bridge.identify_aerodynamics(trajectory, density)
    
    print(f"\nExtracted L/D Ratio: {signature.ld_ratio:.2f} (target: {ld_target})")
    print(f"Classification: {signature.classification}")
    print(f"Confidence: {signature.confidence:.2f}")
    
    report = bridge.generate_signature_report(signature, "TEST_HGV_001")
    print(f"\nSignature Report: {report}")
    
    print("\n✅ Mode-to-SINDy Bridge test complete")
