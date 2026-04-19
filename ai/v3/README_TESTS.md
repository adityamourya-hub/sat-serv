# Axiom-Zero v3 Test Suite

This directory contains comprehensive unit tests for the Axiom-Zero physics substrate.

## Test Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest configuration and shared fixtures |
| `test_sfmn_kernel.py` | Tests for SFMN kernel (rKAN, Legendre, integrators) |
| `test_hamiltonian_discovery.py` | Tests for Hamiltonian discovery and maneuver detection |

## Running Tests

```bash
# Run all tests
pytest ai/v3/ -v

# Run specific test file
pytest ai/v3/test_sfmn_kernel.py -v

# Run with coverage
pytest ai/v3/ --cov=ai/v3 --cov-report=html

# Run specific test class
pytest ai/v3/test_sfmn_kernel.py::TestLegendreBasis -v
```

## Test Coverage

### SFMN Kernel (`test_sfmn_kernel.py`)
- ✅ Legendre polynomial basis computation
- ✅ Rational KAN forward pass
- ✅ Softplus activation
- ✅ Meeus ephemeris (Sun/Moon position)
- ✅ Yoshida-4 symplectic integrator
- ✅ Energy conservation

### Hamiltonian Discovery (`test_hamiltonian_discovery.py`)
- ✅ Trajectory generation
- ✅ Hamiltonian conservation
- ✅ Delta-V detection
- ✅ Residual analysis
- ✅ SHAP explainability
- ✅ Physics geometry layer

## Requirements

```bash
pip install pytest pytest-cov jax jaxlib numpy
```

## TODO

- [ ] Add integration tests with real TLE data
- [ ] Add GPU-specific tests
- [ ] Add performance benchmarks
- [ ] Add tests for federated learning components
