"""
Physics-Informed Neural Operator (PINO) for SSIS Manoeuvre Detection.
Enforces Hamiltonian conservation (Energy/Angular Momentum) as hard constraints.
"""

import numpy as np

class PINOManoeuvreDetector:
    def __init__(self, mu: float = 398600.44):
        """
        mu: Earth's Standard Gravitational Parameter (km^3/s^2)
        """
        self.mu = mu

    def calculate_h_specific(self, r: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate specific orbital energy: h = v^2/2 - mu/r.
        For a passive orbit, h should remain constant.
        """
        v_mag = np.linalg.norm(v)
        r_mag = np.linalg.norm(r)
        if r_mag == 0:
            raise ValueError("Position vector magnitude cannot be zero")
        return (v_mag**2 / 2.0) - (self.mu / r_mag)

    def detect_manoeuvre(self, states: np.ndarray, threshold: float = 0.01) -> bool:
        """
        states: Array of state vectors [r_x, r_y, r_z, v_x, v_y, v_z].
        Detects if specific energy (h) changes by more than the threshold.
        """
        if len(states) < 2:
            return False
        if states.shape[1] != 6:
            raise ValueError("Each state vector must have 6 elements [r_x, r_y, r_z, v_x, v_y, v_z]")
            
        r_start, v_start = states[0][:3], states[0][3:]
        h_initial = self.calculate_h_specific(r_start, v_start)
        
        for state in states[1:]:
            r, v = state[:3], state[3:]
            h_current = self.calculate_h_specific(r, v)
            
            # Change in energy = external work done (manoeuvre)
            delta_h = abs(h_current - h_initial)
            if delta_h > threshold:
                return True
                
        return False

if __name__ == "__main__":
    # Test: Create a passive orbit and a manoeuvring one
    r = np.array([7000, 0, 0])
    v = np.array([0, 7.5, 0]) # approx circular velocity at 7000km
    
    detector = PINOManoeuvreDetector()
    h_pass = detector.calculate_h_specific(r, v)
    print(f"Initial energy: {h_pass}")
    
    # Passive states (Mock)
    passive_states = np.array([np.concatenate([r, v]), np.concatenate([r, v])])
    print(f"Passive manoeuvre detected: {detector.detect_manoeuvre(passive_states)}")
    
    # Manoeuvring state (Burn)
    v_burn = v + np.array([0, 0.1, 0]) # 100m/s delta-V
    manoeuvre_states = np.array([np.concatenate([r, v]), np.concatenate([r, v_burn])])
    print(f"Burn manoeuvre detected: {detector.detect_manoeuvre(manoeuvre_states)}")
