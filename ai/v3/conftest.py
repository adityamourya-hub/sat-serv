"""
conftest.py
Pytest configuration for Axiom-Zero v3 test suite.
"""

import pytest
import jax

# Enable 64-bit precision for all tests
def pytest_configure(config):
    """Configure JAX for 64-bit precision before any tests run."""
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def rng_key():
    """Provide a random number generator key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_state():
    """Provide a sample orbital state for testing."""
    return {
        'position': jnp.array([7000.0, 0.0, 0.0]),  # km
        'velocity': jnp.array([0.0, 7.5, 0.0]),      # km/s
        'epoch': 2451545.0,                           # JD
    }


@pytest.fixture
def default_params():
    """Provide default rKAN parameters for testing."""
    import jax.numpy as jnp
    return {
        'num': jnp.ones((1, 13, 5)),  # (out_dim, in_dim, degree+1)
        'den': jnp.ones((1, 13, 5)) * 0.1
    }
