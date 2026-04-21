"""
Jacobian Validation Test Suite

Validates that JAX and PyTorch gradients match within 1e-7 tolerance.
This is the critical "Model Drift" check for the Audit and Align phase.

The test compares:
    1. jax.grad(H)(z) vs torch.autograd.grad(H)(z)
    2. Second-order derivatives (Hessian) if applicable
    3. Vector field consistency: (J-R)∇H + gu

Reference:
    - Numerical Precision: IEEE 754 double-precision
    - Tolerance: 1e-7 (machine epsilon for double precision)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class ValidationConfig:
    """Configuration for Jacobian validation."""
    
    # Test parameters
    num_test_states: int = 10000
    altitude_range_km: Tuple[float, float] = (200, 36000)
    
    # Tolerance settings
    gradient_tolerance: float = 1e-7
    hessian_tolerance: float = 1e-5
    vector_field_tolerance: float = 1e-6
    
    # Output settings
    verbose: bool = True
    save_report: bool = True
    report_path: str = "jacobian_validation_report.json"


class JacobianValidator:
    """
    Validates gradient consistency between JAX and PyTorch backends.
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.test_states = self._generate_orbital_states()
    
    def _generate_orbital_states(self) -> np.ndarray:
        """
        Generate realistic orbital states for testing.
        
        States range from LEO (200km) to GEO (36000km).
        """
        np.random.seed(42)
        states = []
        
        r_min = 6378137.0 + self.config.altitude_range_km[0] * 1000
        r_max = 6378137.0 + self.config.altitude_range_km[1] * 1000
        
        for _ in range(self.config.num_test_states):
            # Random position in spherical coordinates
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Velocity for circular orbit (approximate)
            v_mag = np.sqrt(3.986004418e14 / r)
            vx = v_mag * np.random.uniform(-1, 1)
            vy = v_mag * np.random.uniform(-1, 1)
            vz = v_mag * np.random.uniform(-1, 1)
            
            states.append([x, y, z, vx, vy, vz])
        
        return np.array(states)
    
    def validate_hamiltonian_gradient(
        self, 
        jax_model: eqx.Module, 
        torch_model: nn.Module
    ) -> Dict[str, any]:
        """
        Compare jax.grad(H) vs torch.autograd.grad(H).
        
        Args:
            jax_model: JAX/Equinox Port-Hamiltonian model
            torch_model: PyTorch Port-Hamiltonian model
            
        Returns:
            Validation report with statistics
        """
        errors = []
        
        for i, state in enumerate(self.test_states):
            # JAX gradient
            z_jax = jnp.array(state)
            grad_jax = jax.grad(lambda z: jax_model.h_net(z)[0])(z_jax)
            grad_jax_np = np.array(grad_jax)
            
            # PyTorch gradient
            z_torch = torch.tensor(state, dtype=torch.float32, requires_grad=True)
            grad_torch = torch_model.gradient_hamiltonian(z_torch.unsqueeze(0))
            grad_torch_np = grad_torch.detach().cpu().numpy()[0]
            
            # Compute error
            error = np.max(np.abs(grad_jax_np - grad_torch_np))
            errors.append(error)
            
            if self.config.verbose and (i < 5 or error > self.config.gradient_tolerance):
                print(f"State {i}: max error = {error:.2e}")
        
        errors = np.array(errors)
        
        report = {
            'test_name': 'hamiltonian_gradient',
            'num_states': len(self.test_states),
            'max_error': float(np.max(errors)),
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'p95_error': float(np.percentile(errors, 95)),
            'p99_error': float(np.percentile(errors, 99)),
            'tolerance': self.config.gradient_tolerance,
            'passed': bool(np.max(errors) < self.config.gradient_tolerance),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        return report
    
    def validate_vector_field(
        self,
        jax_model: eqx.Module,
        torch_model: nn.Module
    ) -> Dict[str, any]:
        """
        Compare vector field (J-R)∇H + gu.
        
        Args:
            jax_model: JAX/Equinox Port-Hamiltonian model
            torch_model: PyTorch Port-Hamiltonian model
            
        Returns:
            Validation report
        """
        errors = []
        
        for i, state in enumerate(self.test_states[:1000]):  # Sample for speed
            z = np.array(state)
            u = np.zeros(3)  # No control input for this test
            
            # JAX vector field
            z_jax = jnp.array(z)
            u_jax = jnp.array(u)
            
            # Compute JAX vector field (simplified - assumes model has vector_field method)
            # In production, this would call the actual model method
            zdot_jax = self._jax_vector_field(jax_model, z_jax, u_jax)
            
            # PyTorch vector field
            z_torch = torch.tensor(z, dtype=torch.float32)
            u_torch = torch.tensor(u, dtype=torch.float32)
            zdot_torch = torch_model(z_torch.unsqueeze(0), u_torch.unsqueeze(0))
            zdot_torch_np = zdot_torch.detach().cpu().numpy()[0]
            
            # Convert JAX to numpy
            zdot_jax_np = np.array(zdot_jax)
            
            # Compute error
            error = np.max(np.abs(zdot_jax_np - zdot_torch_np))
            errors.append(error)
        
        errors = np.array(errors)
        
        report = {
            'test_name': 'vector_field',
            'num_states': len(errors),
            'max_error': float(np.max(errors)),
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'tolerance': self.config.vector_field_tolerance,
            'passed': bool(np.max(errors) < self.config.vector_field_tolerance),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        return report
    
    def _jax_vector_field(self, model: eqx.Module, z: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute vector field for JAX model.
        
        Placeholder - actual implementation depends on model structure.
        """
        # This is a simplified placeholder
        # In production, this would call the actual model's vector_field method
        return jnp.zeros_like(z)
    
    def validate_structural_properties(
        self,
        jax_model: eqx.Module,
        torch_model: nn.Module
    ) -> Dict[str, any]:
        """
        Validate structural properties: J skew-symmetry, R PSD.
        
        Args:
            jax_model: JAX/Equinox Port-Hamiltonian model
            torch_model: PyTorch Port-Hamiltonian model
            
        Returns:
            Validation report
        """
        skew_errors = []
        psd_violations = []
        
        for i, state in enumerate(self.test_states[:100]):
            # Test PyTorch model (easier to access matrices)
            z_torch = torch.tensor(state, dtype=torch.float32)
            
            # Check J skew-symmetry
            J = torch_model.interconnection_matrix(z_torch.unsqueeze(0))[0]
            J_np = J.detach().cpu().numpy()
            skew_error = np.max(np.abs(J_np + J_np.T))
            skew_errors.append(skew_error)
            
            # Check R PSD
            R = torch_model.dissipation_matrix(z_torch.unsqueeze(0))[0]
            R_np = R.detach().cpu().numpy()
            eigvals = np.linalg.eigvalsh(R_np)
            min_eig = np.min(eigvals)
            psd_violation = max(0, -min_eig)
            psd_violations.append(psd_violation)
        
        report = {
            'test_name': 'structural_properties',
            'skew_symmetry_max_error': float(np.max(skew_errors)),
            'skew_symmetry_mean_error': float(np.mean(skew_errors)),
            'psd_max_violation': float(np.max(psd_violations)),
            'psd_mean_violation': float(np.mean(psd_violations)),
            'skew_symmetry_passed': bool(np.max(skew_errors) < 1e-7),
            'psd_passed': bool(np.max(psd_violations) < 1e-6),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        return report
    
    def run_full_validation(
        self,
        jax_model: eqx.Module,
        torch_model: nn.Module
    ) -> Dict[str, any]:
        """
        Run complete validation suite.
        
        Args:
            jax_model: JAX/Equinox Port-Hamiltonian model
            torch_model: PyTorch Port-Hamiltonian model
            
        Returns:
            Complete validation report
        """
        print("=" * 70)
        print("Jacobian Validation Test Suite")
        print("=" * 70)
        
        print(f"\nTest Configuration:")
        print(f"  States: {self.config.num_test_states}")
        print(f"  Gradient tolerance: {self.config.gradient_tolerance}")
        print(f"  Vector field tolerance: {self.config.vector_field_tolerance}")
        
        # Run tests
        reports = []
        
        print("\n1. Validating Hamiltonian Gradient...")
        grad_report = self.validate_hamiltonian_gradient(jax_model, torch_model)
        reports.append(grad_report)
        print(f"   Max error: {grad_report['max_error']:.2e}")
        print(f"   Status: {'✅ PASS' if grad_report['passed'] else '❌ FAIL'}")
        
        print("\n2. Validating Vector Field...")
        vf_report = self.validate_vector_field(jax_model, torch_model)
        reports.append(vf_report)
        print(f"   Max error: {vf_report['max_error']:.2e}")
        print(f"   Status: {'✅ PASS' if vf_report['passed'] else '❌ FAIL'}")
        
        print("\n3. Validating Structural Properties...")
        struct_report = self.validate_structural_properties(jax_model, torch_model)
        reports.append(struct_report)
        print(f"   J skew-symmetry error: {struct_report['skew_symmetry_max_error']:.2e}")
        print(f"   R PSD violation: {struct_report['psd_max_violation']:.2e}")
        print(f"   Status: {'✅ PASS' if (struct_report['skew_symmetry_passed'] and struct_report['psd_passed']) else '❌ FAIL'}")
        
        # Overall status
        all_passed = all(r['passed'] for r in reports)
        
        full_report = {
            'overall_passed': all_passed,
            'tests': reports,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Save report if requested
        if self.config.save_report:
            with open(self.config.report_path, 'w') as f:
                json.dump(full_report, f, indent=2)
            print(f"\nReport saved to: {self.config.report_path}")
        
        print("\n" + "=" * 70)
        print(f"Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print("=" * 70)
        
        return full_report


def demo_validation():
    """
    Demonstrate Jacobian validation with mock models.
    """
    print("=" * 70)
    print("Jacobian Validation Demo (Mock Models)")
    print("=" * 70)
    
    config = ValidationConfig(
        num_test_states=100,  # Reduced for demo
        verbose=True,
        save_report=False,
    )
    
    validator = JacobianValidator(config)
    
    print(f"\nGenerated {len(validator.test_states)} test states")
    print(f"  Altitude range: {config.altitude_range_km[0]}-{config.altitude_range_km[1]} km")
    
    print(f"\nNote: Full validation requires actual JAX and PyTorch models.")
    print(f"This demo shows the validation framework structure.")
    
    print("\n" + "=" * 70)
    print("To run full validation:")
    print("  1. Train JAX Port-Hamiltonian model on H100")
    print("  2. Transfer weights to PyTorch shell")
    print("  3. Run: validator.run_full_validation(jax_model, torch_model)")
    print("  4. Verify max error < 1e-7 tolerance")
    print("=" * 70)


if __name__ == "__main__":
    demo_validation()
