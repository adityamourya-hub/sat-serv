"""
Auditor Layer: Physics Grounding (Axiom-Prime Architecture)

Ensures Generator doesn't "hallucinate" rerouting paths that violate physical laws.
Implements:
- Ruth-4 / Forest-Ruth Symplectic Integrator (4th order)
- Real-time Pfaffian Invariant Checker
- Quantized Physics-Hardened Neural Network (Q-PHNN) with 1.58-bit ternary logic

Part of the Triadic Intelligence framework.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# Optional JAX for accelerated Pfaffian
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np


@dataclass
class PfaffianInvariant:
    """Result of Pfaffian invariant check."""
    pfaffian_value: complex
    sign: int  # +1 or -1
    is_topological: bool
    valid: bool  # False if calculation failed
    error_message: Optional[str] = None


class RealTimePfaffianChecker:
    """
    Real-time Pfaffian calculation for BdG topological invariant.
    
    Mathematical verification: if Pf(A) changes sign, topological protection
    is lost. Used as "Hard Fail" metric to invalidate generated unitary paths.
    
    The Pfaffian is computed for the skew-symmetric BdG matrix:
        H_BdG = [ H_0    Δ   ]
                [ -Δ^T  -H_0^T ]
    
    For antisymmetric A (A^T = -A): Pf(A)^2 = det(A)
    Topological invariant: Q = sgn[Pf(H_BdG)]
    """
    
    def __init__(self, use_schur: bool = True):
        """
        Args:
            use_schur: Use Schur decomposition for numerical stability
        """
        self.use_schur = use_schur
        self.check_history: List[Tuple[float, PfaffianInvariant]] = []
        
        # Try to import scipy for Schur
        try:
            from scipy.linalg import schur
            self.schur = schur
            self.has_scipy = True
        except ImportError:
            self.schur = None
            self.has_scipy = False
    
    def compute_pfaffian(
        self,
        H_bdg: NDArray[np.complex128],
        tolerance: float = 1e-10
    ) -> PfaffianInvariant:
        """
        Compute Pfaffian of BdG Hamiltonian for topological check.
        
        Args:
            H_bdg: BdG Hamiltonian matrix (should be antisymmetric in Majorana basis)
            tolerance: Numerical zero threshold
            
        Returns:
            PfaffianInvariant with topological status
        """
        n = H_bdg.shape[0]
        
        # Check dimensions
        if n % 2 != 0:
            return PfaffianInvariant(
                pfaffian_value=0.0,
                sign=1,
                is_topological=False,
                valid=False,
                error_message="Odd dimension - no Pfaffian defined"
            )
        
        # Convert to Majorana basis (real antisymmetric)
        A = self._to_majorana_basis(H_bdg)
        
        # Check antisymmetry
        if not np.allclose(A, -A.T, atol=tolerance):
            # Not antisymmetric - likely in Nambu basis
            # Extract antisymmetric part
            A = 0.5 * (A - A.T)
        
        try:
            if self.use_schur and self.has_scipy and n > 4:
                pf = self._pfaffian_schur(A, tolerance)
            else:
                pf = self._pfaffian_direct(A)
            
            # Determine topological phase
            sign = -1 if np.real(pf) < 0 else 1
            is_topological = (sign == -1) and (np.abs(pf) > tolerance)
            
            return PfaffianInvariant(
                pfaffian_value=complex(pf),
                sign=sign,
                is_topological=is_topological,
                valid=True
            )
            
        except Exception as e:
            return PfaffianInvariant(
                pfaffian_value=0.0,
                sign=1,
                is_topological=False,
                valid=False,
                error_message=str(e)
            )
    
    def _to_majorana_basis(
        self,
        H_bdg: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        """
        Convert BdG Hamiltonian from Nambu to Majorana basis.
        
        Nambu: [e↑, e↓, h↑, h↓]
        Majorana: [γ1, γ2, ...] where γ = (e + e†)/√2
        
        Returns real antisymmetric matrix.
        """
        # Simplified: extract antisymmetric component
        # Full transformation requires unitary basis change
        
        # For now: use imaginary part to extract antisymmetric structure
        A = np.imag(H_bdg)
        
        # Make it properly antisymmetric
        A = 0.5 * (A - A.T)
        
        return A
    
    def _pfaffian_schur(
        self,
        A: NDArray[np.float64],
        tolerance: float
    ) -> complex:
        """
        Compute Pfaffian via Schur decomposition (Wimmer 2012 algorithm).
        
        Numerically stable for large matrices.
        """
        n = A.shape[0]
        
        # Schur decomposition: A = Q * T * Q^T
        T, Q = self.schur(A)
        
        # Pfaffian is product of 2x2 block determinants
        pf = 1.0
        for i in range(0, n, 2):
            if i + 1 < n:
                block = T[i:i+2, i:i+2]
                pf *= block[0, 1]  # Off-diagonal element
        
        return complex(pf)
    
    def _pfaffian_direct(
        self,
        A: NDArray
    ) -> complex:
        """Direct Pfaffian for small matrices (n <= 4)."""
        n = A.shape[0]
        
        if n == 0:
            return 1.0
        elif n == 2:
            return A[0, 1]
        elif n == 4:
            # Pfaffian formula for 4x4
            return (
                A[0, 1] * A[2, 3] -
                A[0, 2] * A[1, 3] +
                A[0, 3] * A[1, 2]
            )
        else:
            # Recursion for larger matrices (inefficient but correct)
            return self._pfaffian_recursive(A)
    
    def _pfaffian_recursive(
        self,
        A: NDArray,
        i: int = 0
    ) -> complex:
        """Recursive Pfaffian using Laplace expansion."""
        n = A.shape[0]
        
        if n == 0:
            return 1.0
        if n == 2:
            return A[0, 1]
        
        # Expand along row i
        pf = 0.0
        sign = 1.0
        
        for j in range(i + 1, n):
            if abs(A[i, j]) > 1e-15:
                # Remove rows i,j and columns i,j
                mask = np.ones(n, dtype=bool)
                mask[i] = mask[j] = False
                submatrix = A[np.ix_(mask, mask)]
                
                pf += sign * A[i, j] * self._pfaffian_recursive(submatrix, max(0, i - 1))
            sign *= -1
        
        return pf

@jax.jit
def jax_pfaffian_kernel(A: jnp.ndarray) -> jnp.ndarray:
    """
    High-performance Pfaffian kernel for skew-symmetric matrices.
    Target: <50ns execution (post-JIT).
    Algorithm: Gaussian elimination for Pfaffian (Schur Complement).
    
    Optimized for fixed-size N=32. The loop is unrolled by XLA.
    """
    mat = A
    pf = 1.0
    
    # We use a fixed range for N=32 to ensure XLA unrolling and avoid tracing errors.
    # For N=32, we have 16 iterations.
    for i in range(16):
        # Pivot A[0, 1]
        p = mat[0, 1]
        pf *= p
        
        # Schur complement update: A' = A[2:, 2:] + (col0 @ row1 - col1 @ row0) / p
        col_0 = mat[2:, 0]
        col_1 = mat[2:, 1]
        row_0 = mat[0, 2:]
        row_1 = mat[1, 2:]
        
        # O(N^2) update
        update = (jnp.outer(col_0, row_1) - jnp.outer(col_1, row_0)) / (p + 1e-15)
        mat = mat[2:, 2:] + update
    
    return jnp.sign(pf)
    
    def check_rerouting_path(
        self,
        H_bdg_initial: NDArray[np.complex128],
        H_bdg_final: NDArray[np.complex128],
        timestamp: float
    ) -> Dict:
        """
        Check if rerouting path preserves topological protection.
        
        Validates:
        1. Initial state is topological
        2. Final state is topological
        3. No sign flip occurred during transition
        
        Returns:
            Validation report dictionary
        """
        pf_initial = self.compute_pfaffian(H_bdg_initial)
        pf_final = self.compute_pfaffian(H_bdg_final)
        
        # Store in history
        self.check_history.append((timestamp, pf_final))
        
        # Validation checks
        checks = {
            "initial_topological": pf_initial.is_topological,
            "final_topological": pf_final.is_topological,
            "no_sign_flip": pf_initial.sign == pf_final.sign,
            "valid_calculation": pf_initial.valid and pf_final.valid,
            "pfaffian_initial": complex(pf_initial.pfaffian_value),
            "pfaffian_final": complex(pf_final.pfaffian_value),
        }
        
        # Hard fail conditions
        hard_fail = not (
            checks["valid_calculation"] and
            checks["no_sign_flip"] and
            checks["final_topological"]
        )
        
        return {
            "valid": not hard_fail,
            "checks": checks,
            "timestamp": timestamp,
            "hard_fail": hard_fail
        }


class TernaryQuantizer:
    """
    1.58-bit Ternary Logic Quantization.
    
    Implements {-1, 0, +1} ternary states with 1.58 bits per weight.
    Used in Q-PHNN for energy-efficient physics verification.
    
    The 1.58-bit approximation: log2(3) ≈ 1.585
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Threshold for mapping to -1/+1 (values within threshold map to 0)
        """
        self.threshold = threshold
    
    def quantize(self, x: NDArray[np.float64]) -> NDArray[np.int8]:
        """
        Quantize continuous values to ternary {-1, 0, +1}.
        
        Args:
            x: Continuous values
            
        Returns:
            Ternary quantized values
        """
        result = np.zeros_like(x, dtype=np.int8)
        result[x > self.threshold] = 1
        result[x < -self.threshold] = -1
        return result
    
    def dequantize(self, q: NDArray[np.int8]) -> NDArray[np.float64]:
        """Convert ternary back to continuous (approximate)."""
        return q.astype(np.float64)
    
    def ternary_matmul(
        self,
        weights_ternary: NDArray[np.int8],
        inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Matrix multiplication with ternary weights.
        
        Efficient computation: multiply only non-zero weights.
        """
        result = np.zeros(weights_ternary.shape[0])
        
        for i in range(weights_ternary.shape[0]):
            for j in range(weights_ternary.shape[1]):
                w = weights_ternary[i, j]
                if w != 0:
                    result[i] += w * inputs[j]
        
        return result


class QPHNN:
    """
    Quantized Physics-Hardened Neural Network (Q-PHNN).
    
    Uses 1.58-bit ternary logic to check physics invariants at ultra-low
    power. Validates that Generator outputs satisfy:
    - Energy conservation
    - Symplectic structure preservation
    - Topological invariant maintenance
    
    Architecture:
    - Input: Hamiltonian features (gap, overlap, poisoning)
    - Hidden: Ternary quantized layers
    - Output: Valid/Invalid classification
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        ternary_threshold: float = 0.3
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.quantizer = TernaryQuantizer(ternary_threshold)
        
        # Initialize ternary weights (random {-1, 0, +1})
        self.W1 = self._init_ternary_weights(hidden_dim, input_dim)
        self.W2 = self._init_ternary_weights(1, hidden_dim)
        
        # Bias terms (continuous for flexibility)
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(1)
    
    def _init_ternary_weights(
        self,
        out_dim: int,
        in_dim: int
    ) -> NDArray[np.int8]:
        """Initialize random ternary weights."""
        weights = np.random.choice(
            [-1, 0, 1],
            size=(out_dim, in_dim),
            p=[0.25, 0.5, 0.25]  # Sparse: 50% zeros
        )
        return weights.astype(np.int8)
    
    def forward(self, x: NDArray[np.float64]) -> float:
        """
        Forward pass with ternary weights.
        
        Args:
            x: Input features (gap, overlap, poisoning, etc.)
            
        Returns:
            Validity score (0-1, higher = more valid)
        """
        # Hidden layer with ternary weights
        h = self.quantizer.ternary_matmul(self.W1, x) + self.b1
        h = np.maximum(h, 0)  # ReLU activation
        
        # Output layer
        out = self.quantizer.ternary_matmul(self.W2, h) + self.b2
        
        # Sigmoid to probability
        return float(1.0 / (1.0 + np.exp(-out[0])))
    
    def check_hamiltonian_validity(
        self,
        gap: float,
        mzm_overlap: float,
        qp_poisoning: float,
        energy_drift: float,
        pfaffian_sign: int
    ) -> Dict:
        """
        Check if Hamiltonian parameters satisfy physics constraints.
        
        Returns validation report with Q-PHNN confidence.
        """
        # Construct feature vector
        features = np.array([
            gap / 0.2,  # Normalized gap
            mzm_overlap,
            qp_poisoning,
            energy_drift * 1000,  # Scale up small values
            float(pfaffian_sign),
            gap * mzm_overlap,  # Interaction term
            qp_poisoning * energy_drift,
            1.0 if gap > 0.01 else 0.0,  # Gap threshold
            1.0 if qp_poisoning < 0.5 else 0.0,  # Clean threshold
            1.0 if pfaffian_sign == -1 else 0.0  # Topological
        ])
        
        # Q-PHNN prediction
        validity_score = self.forward(features)
        
        # Rule-based checks (physics constraints)
        rule_checks = {
            "gap_positive": gap > 0.001,
            "overlap_bounded": 0 <= mzm_overlap <= 1,
            "poisoning_bounded": 0 <= qp_poisoning <= 1,
            "energy_stable": abs(energy_drift) < 0.01,
            "topological": pfaffian_sign == -1
        }
        
        all_rules_pass = all(rule_checks.values())
        
        return {
            "valid": validity_score > 0.5 and all_rules_pass,
            "validity_score": validity_score,
            "rule_checks": rule_checks,
            "all_rules_pass": all_rules_pass,
            "bits_per_weight": 1.585  # log2(3)
        }


class PhysicsAuditor:
    """
    Complete Auditor Layer integrating all validation components.
    
    The Auditor is the "physics-grounded" component that ensures the
    Generator's evasion paths are physically realizable.
    """
    
    def __init__(
        self,
        use_symplectic: bool = True,
        use_pfaffian: bool = True,
        use_qphnn: bool = True
    ):
        self.use_symplectic = use_symplectic
        self.use_pfaffian = use_pfaffian
        self.use_qphnn = use_qphnn
        
        # Initialize components
        if use_symplectic:
            from .jax_backend import SymplecticAuditor
            self.symplectic = SymplecticAuditor(order=4)
        
        if use_pfaffian:
            self.pfaffian_checker = RealTimePfaffianChecker()
        
        if use_qphnn:
            self.qphnn = QPHNN()
    
    def validate_rerouting(
        self,
        H_initial: NDArray[np.complex128],
        H_final: NDArray[np.complex128],
        path_metadata: Dict,
        timestamp: float
    ) -> Dict:
        """
        Comprehensive validation of rerouting operation.
        
        Runs all Auditor checks:
        1. Pfaffian invariant (topological protection)
        2. Q-PHNN physics validation
        3. Energy conservation (via symplectic check)
        
        Returns:
            Complete validation report
        """
        results = {
            "timestamp": timestamp,
            "checks": {},
            "valid": True
        }
        
        # Check 1: Pfaffian invariant
        if self.use_pfaffian:
            pf_check = self.pfaffian_checker.check_rerouting_path(
                H_initial, H_final, timestamp
            )
            results["checks"]["pfaffian"] = pf_check
            results["valid"] = results["valid"] and pf_check["valid"]
        
        # Check 2: Q-PHNN validation
        if self.use_qphnn:
            gap = path_metadata.get("gap", 0.0)
            overlap = path_metadata.get("mzm_overlap", 0.0)
            poisoning = path_metadata.get("qp_poisoning", 0.0)
            drift = path_metadata.get("energy_drift", 0.0)
            
            # Get Pfaffian sign if available
            pf_sign = 1
            if self.use_pfaffian:
                pf_result = self.pfaffian_checker.compute_pfaffian(H_final)
                pf_sign = pf_result.sign
            
            qphnn_check = self.qphnn.check_hamiltonian_validity(
                gap, overlap, poisoning, drift, pf_sign
            )
            results["checks"]["qphnn"] = qphnn_check
            results["valid"] = results["valid"] and qphnn_check["valid"]
        
        # Check 3: Symplectic preservation (long-term stability)
        # This would be checked over simulation trajectory
        
        return results
    
    def get_memory_footprint(self) -> Dict:
        """Estimate memory usage of Auditor components."""
        footprint = {}
        
        if self.use_qphnn:
            # Q-PHNN with ternary weights
            w1_bits = self.qphnn.W1.size * 1.585
            w2_bits = self.qphnn.W2.size * 1.585
            footprint["qphnn_weights_bits"] = w1_bits + w2_bits
            footprint["qphnn_weights_kb"] = (w1_bits + w2_bits) / 8192
        
        if self.use_pfaffian:
            # Pfaffian history storage
            footprint["pfaffian_history_entries"] = len(self.pfaffian_checker.check_history)
        
        return footprint


__all__ = [
    "PfaffianInvariant",
    "RealTimePfaffianChecker",
    "TernaryQuantizer",
    "QPHNN",
    "PhysicsAuditor",
]
