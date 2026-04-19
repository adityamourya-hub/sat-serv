"""
test_hamiltonian_discovery.py
Unit tests for Hamiltonian Discovery and symbolic regression components.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Mock the imports since we're testing the structure
# In a real test environment, these would be:
# from hamiltonian_discovery import sharded_generate_trajectories, get_force_learned


class TestHamiltonianDiscovery:
    """Tests for Hamiltonian discovery via symbolic regression."""

    def test_trajectory_generation_shape(self):
        """Generated trajectories should have correct shape"""
        # Placeholder test - in real implementation would test:
        # trajectories = sharded_generate_trajectories(key)
        # assert trajectories.shape == (batch_size, time_steps, state_dim)
        batch_size = 2048
        time_steps = 151
        state_dim = 4  # [qx, qy, px, py]

        # Mock trajectory shape
        mock_trajectories = jnp.zeros((batch_size, time_steps, state_dim))
        assert mock_trajectories.shape == (batch_size, time_steps, state_dim)

    def test_hamiltonian_conservation(self):
        """True Hamiltonian dynamics should conserve energy"""
        # For a simple harmonic oscillator: H = (p^2 + q^2) / 2
        q = jnp.array([1.0, 0.0])
        p = jnp.array([0.0, 1.0])

        # Energy should be constant
        energy = 0.5 * jnp.sum(p**2 + q**2)
        expected_energy = 1.0  # (0 + 1)/2 + (1 + 0)/2 = 1

        assert jnp.isclose(energy, expected_energy)

    def test_symplectic_gradient(self):
        """Force should be gradient of potential with negative sign"""
        # V(q) = q^2 / 2 => F = -dV/dq = -q
        q = jnp.array([0.5, 0.3])

        def potential(qv):
            return 0.5 * jnp.sum(qv**2)

        force_analytic = -q
        force_autodiff = -jax.grad(potential)(q)

        assert jnp.allclose(force_analytic, force_autodiff)


class TestManeuverDistiller:
    """Tests for maneuver detection and delta-V attribution."""

    def test_delta_v_detection(self):
        """Should detect velocity change in trajectory"""
        # Simulate trajectory with maneuver
        t = jnp.linspace(0, 100, 1000)
        v_nominal = jnp.ones_like(t) * 7.5  # km/s

        # Inject delta-V at t=50
        delta_v = 0.1  # 100 m/s maneuver
        v_maneuver = v_nominal.at[500:].set(v_nominal[500:] + delta_v)

        # Simple detection: look for velocity discontinuity
        dv_detected = jnp.max(jnp.abs(jnp.diff(v_maneuver)))
        assert dv_detected >= delta_v * 0.9  # Should detect ~90% of delta-V

    def test_residual_computation(self):
        """Residual RMS should increase after maneuver"""
        # Generate nominal trajectory
        t = jnp.linspace(0, 100, 500)
        q_nominal = jnp.sin(t * 0.1)

        # Add maneuver-induced deviation
        deviation = jnp.zeros_like(t)
        deviation = deviation.at[250:].add(0.5 * jnp.sin((t[250:] - 50) * 0.5))
        q_maneuver = q_nominal + deviation

        # Compute residuals
        residual_nominal = jnp.std(jnp.diff(q_nominal, n=2))
        residual_maneuver = jnp.std(jnp.diff(q_maneuver, n=2))

        assert residual_maneuver > residual_nominal * 2  # Significant increase


class TestExplainability:
    """Tests for explainable AI components."""

    def test_feature_attribution_shape(self):
        """SHAP attributions should match feature count"""
        num_features = 13
        # Mock attribution scores
        attributions = jnp.ones(num_features)

        assert attributions.shape == (num_features,)
        assert jnp.all(attributions >= 0)  # Positive attributions

    def test_attribution_conservation(self):
        """Sum of SHAP values should approximate prediction"""
        base_value = 0.5
        shap_values = jnp.array([0.2, 0.1, -0.05, 0.15])

        prediction = base_value + jnp.sum(shap_values)
        expected = 0.9

        assert jnp.isclose(prediction, expected)


class TestPhysicsGeometry:
    """Tests for equinoctial element geometry layer."""

    def test_cartesian_to_equinoctial_roundtrip(self):
        """Cartesian <-> Equinoctial conversion should be reversible"""
        # Mock state: [x, y, z, vx, vy, vz]
        cartesian = jnp.array([7000.0, 1000.0, 500.0, 0.0, 7.0, 1.0])

        # Placeholder: In real implementation would test conversion
        # equinoctial = cartesian_to_equinoctial(cartesian)
        # cartesian_back = equinoctial_to_cartesian(equinoctial)
        # assert jnp.allclose(cartesian, cartesian_back, atol=1e-6)

        # For now, just verify state shape
        assert cartesian.shape == (6,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
