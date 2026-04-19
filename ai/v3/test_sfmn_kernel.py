"""
test_sfmn_kernel.py
Unit tests for the SFMN (Symplectic Fusion Memory Network) Kernel.
Tests the Legendre basis, rKAN forward pass, and symplectic integrators.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sfmn_kernel import (
    legendre_basis,
    rkan_forward,
    softplus,
    get_meeus_ephemeris,
    henon_heiles_force,
    symplectic_integrator_step,
    Yoshida4Integrator,
)

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestLegendreBasis:
    """Tests for Legendre polynomial basis computation."""

    def test_legendre_degree_0(self):
        """Legendre basis of degree 0 should return [1.0]"""
        x = 0.5
        result = legendre_basis(x, 0)
        assert result.shape == (1,)
        assert jnp.isclose(result[0], 1.0)

    def test_legendre_degree_1(self):
        """Legendre basis of degree 1 should return [1.0, x]"""
        x = 0.5
        result = legendre_basis(x, 1)
        assert result.shape == (2,)
        assert jnp.isclose(result[0], 1.0)
        assert jnp.isclose(result[1], x)

    def test_legendre_degree_4(self):
        """Legendre basis of degree 4 should have 5 elements"""
        x = 0.5
        result = legendre_basis(x, 4)
        assert result.shape == (5,)
        # P0 = 1
        assert jnp.isclose(result[0], 1.0)
        # P1 = x
        assert jnp.isclose(result[1], x)
        # P2 = (3x^2 - 1) / 2
        expected_p2 = (3 * x**2 - 1) / 2
        assert jnp.isclose(result[2], expected_p2, atol=1e-6)

    def test_legendre_orthogonality(self):
        """Legendre polynomials should be orthogonal on [-1, 1]"""
        # Test orthogonality by numerical integration
        x_vals = jnp.linspace(-1, 1, 1000)
        p2_vals = legendre_basis(x_vals, 2)
        p3_vals = legendre_basis(x_vals, 3)

        # Inner product of P2 and P3 should be approximately 0
        inner_product = jnp.trapezoid(p2_vals[:, 2] * p3_vals[:, 3], x_vals)
        assert jnp.abs(inner_product) < 0.1  # Approximate due to discrete sampling


class TestSoftplus:
    """Tests for softplus activation function."""

    def test_softplus_zero(self):
        """softplus(0) should be log(2)"""
        result = softplus(0.0)
        assert jnp.isclose(result, jnp.log(2.0))

    def test_softplus_positive(self):
        """softplus of positive value should be positive"""
        result = softplus(5.0)
        assert result > 5.0  # For large x, softplus(x) ≈ x
        assert result < 5.1

    def test_softplus_negative(self):
        """softplus of negative value should approach 0"""
        result = softplus(-10.0)
        assert result > 0.0
        assert result < 0.01

    def test_softplus_large_positive(self):
        """For large positive values, softplus(x) ≈ x"""
        x = 25.0
        result = softplus(x)
        assert jnp.isclose(result, x, atol=0.01)


class TestRKANForward:
    """Tests for Rational KAN forward pass."""

    def test_rkan_output_shape(self):
        """rKAN output should have correct shape"""
        x_in = jnp.array([1.0, 2.0, 3.0])
        degree = 4
        out_dim = 1

        # Create simple params
        params = {
            'num': jnp.ones((out_dim, 3, degree + 1)),
            'den': jnp.ones((out_dim, 3, degree + 1)) * 0.1
        }

        result = rkan_forward(x_in, params, degree, out_dim)
        assert result.shape == (out_dim,)

    def test_rkan_multi_output(self):
        """rKAN should support multiple outputs"""
        x_in = jnp.array([1.0, 2.0])
        degree = 3
        out_dim = 5

        params = {
            'num': jnp.ones((out_dim, 2, degree + 1)),
            'den': jnp.ones((out_dim, 2, degree + 1)) * 0.1
        }

        result = rkan_forward(x_in, params, degree, out_dim)
        assert result.shape == (out_dim,)

    def test_rkan_differentiable(self):
        """rKAN should be differentiable"""
        x_in = jnp.array([1.0, 2.0])
        degree = 3
        out_dim = 1

        params = {
            'num': jnp.ones((out_dim, 2, degree + 1)),
            'den': jnp.ones((out_dim, 2, degree + 1)) * 0.1
        }

        def loss_fn(p):
            return jnp.sum(rkan_forward(x_in, p, degree, out_dim))

        # Compute gradient - should not raise error
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        assert 'num' in grads
        assert 'den' in grads


class TestMeeusEphemeris:
    """Tests for Sun/Moon ephemeris calculations."""

    def test_meeus_returns_unit_vectors(self):
        """Sun and Moon vectors should be unit length"""
        jd = 2451545.0  # J2000 epoch
        sun_v, moon_v = get_meeus_ephemeris(jd)

        sun_norm = jnp.linalg.norm(sun_v)
        moon_norm = jnp.linalg.norm(moon_v)

        assert jnp.isclose(sun_norm, 1.0, atol=0.01)
        assert jnp.isclose(moon_norm, 1.0, atol=0.01)

    def test_meeus_sun_direction(self):
        """Sun direction should vary with time"""
        jd1 = 2451545.0  # J2000
        jd2 = 2451545.0 + 182.625  # ~6 months later

        sun_v1, _ = get_meeus_ephemeris(jd1)
        sun_v2, _ = get_meeus_ephemeris(jd2)

        # Sun direction should be significantly different
        dot_product = jnp.dot(sun_v1, sun_v2)
        assert dot_product < 0.9  # Not the same direction

    def test_meeus_consistency(self):
        """Same JD should give same result"""
        jd = 2451545.0 + 100.0
        sun_v1, moon_v1 = get_meeus_ephemeris(jd)
        sun_v2, moon_v2 = get_meeus_ephemeris(jd)

        assert jnp.allclose(sun_v1, sun_v2)
        assert jnp.allclose(moon_v1, moon_v2)


class TestSymplecticIntegrator:
    """Tests for Yoshida-4 symplectic integrator."""

    def test_integrator_preserves_energy_harmonic(self):
        """For harmonic oscillator, symplectic integrator should approximately conserve energy"""
        # Simple harmonic oscillator: H = (p^2 + q^2) / 2
        def force_harmonic(q):
            return -q  # F = -dV/dq = -q

        q0 = jnp.array([1.0])
        p0 = jnp.array([0.0])
        dt = 0.01

        # Integrate for many steps
        q, p = q0, p0
        initial_energy = 0.5 * (p[0]**2 + q[0]**2)

        for _ in range(1000):
            q, p = symplectic_integrator_step(q, p, dt, force_harmonic)

        final_energy = 0.5 * (p[0]**2 + q[0]**2)
        energy_drift = abs(final_energy - initial_energy) / initial_energy

        # Symplectic integrator should have bounded energy drift
        assert energy_drift < 0.01  # Less than 1% drift

    def test_yoshida4_integrator_shape(self):
        """Yoshida4Integrator should return correct trajectory shape"""
        q0 = jnp.array([0.1, 0.0])
        p0 = jnp.array([0.0, 0.5])
        t_grid = jnp.linspace(0, 1.0, 100)

        energy, energy_error, (qt, pt) = Yoshida4Integrator.integrate(
            q0, p0, t_grid, henon_heiles_force
        )

        assert qt.shape == (100, 2)
        assert pt.shape == (100, 2)
        assert energy.shape == ()  # Scalar
        assert isinstance(energy_error, float)

    def test_reversibility(self):
        """Symplectic integrator should be reversible"""
        q0 = jnp.array([0.5, 0.3])
        p0 = jnp.array([0.2, -0.1])
        dt = 0.1

        # Forward step
        q_fwd, p_fwd = symplectic_integrator_step(q0, p0, dt, henon_heiles_force)

        # Backward step (negative dt)
        q_back, p_back = symplectic_integrator_step(q_fwd, p_fwd, -dt, henon_heiles_force)

        assert jnp.allclose(q0, q_back, atol=1e-5)
        assert jnp.allclose(p0, p_back, atol=1e-5)


class TestHenonHeiles:
    """Tests for Hénon-Heiles system (standard test case)."""

    def test_force_computation(self):
        """Hénon-Heiles force should be computable"""
        q = jnp.array([0.1, 0.2])
        force = henon_heiles_force(q)

        assert force.shape == (2,)
        assert jnp.all(jnp.isfinite(force))

    def test_force_symmetry(self):
        """Force at origin should be zero"""
        q = jnp.array([0.0, 0.0])
        force = henon_heiles_force(q)

        assert jnp.allclose(force, jnp.zeros(2), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
