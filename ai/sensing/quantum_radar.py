"""
Quantum Radar Simulation for SSIS.
Capability g: Characterisation & Cataloguing.
Ported from Substrate B (qc_protocol).
"""

import numpy as np
from scipy.special import erfc

class QuantumRadarSim:
    def __init__(self, freq_hz=1.93e14): # 1550nm default
        self.freq_hz = freq_hz
        self.h = 6.626e-34
        self.kb = 1.38e-23
        self.c = 3e8

    def calculate_noise_floor(self, temp_k=300.0):
        """
        Calculates background photons per mode (N_B).
        """
        exponent = (self.h * self.freq_hz) / (self.kb * temp_k)
        if exponent > 100:
            return 1e-12 # Quantum noise floor
        return 1.0 / (np.exp(exponent) - 1)

    def calculate_kappa(self, range_km, rcs_m2, gain_db=40.0):
        """
        Calculates round-trip transmissivity.
        """
        wavelength = self.c / self.freq_hz
        gain = 10**(gain_db / 10.0)
        range_m = range_km * 1000
        
        kappa = (gain * (wavelength**2) * rcs_m2) / (((4 * np.pi)**3) * (range_m**4))
        return min(1.0, max(0.0, kappa))

    def evaluate_advantage(self, ns=0.01, nb=0.001, kappa=1e-6, m=1e6):
        """
        Compares Classical vs. Quantum Illumination (QI) error prob.
        Returns the SNR advantage.
        """
        # Classical error bound
        exp_cs = (m * kappa * ns) / (4 * nb + 1)
        p_err_cs = 0.5 * np.exp(-exp_cs)
        
        # QI error bound (6dB advantage in ideal conditions)
        exp_qi = (m * kappa * ns) / (nb + 1e-12)
        p_err_qi = 0.5 * np.exp(-exp_qi)
        
        return p_err_cs, p_err_qi

if __name__ == "__main__":
    sim = QuantumRadarSim()
    p_cs, p_qi = sim.evaluate_advantage()
    print(f"Classical P_err: {p_cs:.2e} | Quantum P_err: {p_qi:.2e}")
    print(f"Advantage Ratio: {p_cs / p_qi:.2f}x")
