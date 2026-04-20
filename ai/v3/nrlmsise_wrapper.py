"""
NRLMSISE-00 C Reference Bridge for JAX Training

Bridges the 1970s C reference implementation into JAX/XLA training loop
using CFFI and jax.pure_callback.

Architecture:
    - C Library: libmsise.so (compiled from nrlmsise-00.c)
    - Bridge: CFFI (C Foreign Function Interface)
    - JAX: jax.pure_callback for XLA compatibility
    - Purpose: Ground truth oracle for HydrostaticAtmosphere training

Training Pipeline (H100):
    1. JAX trainer generates (h, φ, λ, F10.7) state coordinates
    2. jax_msise_density triggers pure_callback to CPU host
    3. CFFI executes gtd7() C function
    4. HydrostaticAtmosphere predicts ρ_nn
    5. Loss: L_atm = Σ ||ρ_nn - ρ_msis||²

M2 Air Redundancy:
    - Go-native engine calls C-code via CGO as fallback
    - ConflictResolver flags >20% divergence as atmospheric anomaly

Mathematical Cleanliness:
    - Gradients for ρ_nn are mathematically perfect
    - Model learns to mimic C-code while obeying dP/dz = -ρg
    - Result: C∞ smooth density manifold for Koopman/PHS layers

References:
    - NRLMSISE-00: Picone et al. "NRLMSISE-00 Empirical Model"
    - CFFI: Python C Foreign Function Interface
    - JAX pure_callback: https://jax.readthedocs.io/en/latest/jax.experimental.html
"""

import os
import numpy as np
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from functools import partial

# =============================================================================
# CFFI Bridge to NRLMSISE-00
# =============================================================================

try:
    from cffi import FFI
    
    ffi = FFI()
    
    # Define C structures matching nrlmsise-00.h
    ffi.cdef("""
        struct nrlmsise_input {
            int year;
            int doy;
            double sec;
            double alt;
            double g_lat;
            double g_long;
            double lst;
            double f107A;
            double f107;
            double ap;
            double *ap_a;
        };
        
        struct nrlmsise_flags {
            int switches[24];
            double sw[24];
            double swc[24];
        };
        
        struct nrlmsise_output {
            double d[9];  /* Density outputs */
            double t[2];  /* Temperature outputs */
        };
        
        void gtd7(struct nrlmsise_input *input, 
                  struct nrlmsise_flags *flags, 
                  struct nrlmsise_output *output);
        
        void gtd7d(struct nrlmsise_input *input,
                   struct nrlmsise_flags *flags,
                   struct nrlmsise_output *output);
    """)
    
    # Load shared library
    # Try multiple paths for different deployment scenarios
    _LIB_PATHS = [
        # Development paths
        os.path.abspath("pkg/physics/c/libmsise.so"),
        os.path.abspath("pkg/physics/c/libmsise.dylib"),  # macOS
        os.path.abspath("../pkg/physics/c/libmsise.so"),
        os.path.abspath("../../pkg/physics/c/libmsise.so"),
        # Production paths
        "/usr/local/lib/libmsise.so",
        "/opt/aditi/lib/libmsise.so",
        # Relative to this file
        os.path.join(os.path.dirname(__file__), "../../pkg/physics/c/libmsise.so"),
    ]
    
    _C_LIB = None
    _LIB_LOADED = False
    
    for lib_path in _LIB_PATHS:
        if os.path.exists(lib_path):
            try:
                _C_LIB = ffi.dlopen(lib_path)
                _LIB_LOADED = True
                print(f"[NRLMSISE] Loaded library: {lib_path}")
                break
            except Exception as e:
                print(f"[NRLMSISE] Failed to load {lib_path}: {e}")
                continue
    
    if not _LIB_LOADED:
        print("[NRLMSISE] WARNING: Could not load libmsise.so. C-callbacks will fail.")
        print("[NRLMSISE] Compile with: gcc -shared -o libmsise.so -fPIC -O3 nrlmsise-00.c")

except ImportError:
    print("[NRLMSISE] WARNING: cffi not installed. Install with: pip install cffi")
    _LIB_LOADED = False
    _C_LIB = None


# =============================================================================
# Host-Side C Wrapper Functions
# =============================================================================

def call_msise_gtd7(
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    f107: float,
    f107a: Optional[float] = None,
    ap: float = 15.0,
    year: int = 2024,
    doy: int = 1,
    sec: float = 43200.0,  # Noon
    lst: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Host-side Python wrapper for NRLMSISE-00 gtd7 C function.
    
    Returns total mass density and full output structure.
    
    Args:
        alt_km: Altitude in kilometers (0-1000)
        lat_deg: Geographic latitude in degrees (-90 to 90)
        lon_deg: Geographic longitude in degrees (-180 to 180)
        f107: Solar flux at 10.7 cm (SFU)
        f107a: 81-day average F10.7 (defaults to f107)
        ap: Planetary geomagnetic index
        year: Year (for doy calculation)
        doy: Day of year (1-366)
        sec: Seconds in day (0-86400)
        lst: Local solar time in hours (defaults to lon-based estimate)
        
    Returns:
        (total_density, output_dict)
        - total_density: Total mass density in kg/m³
        - output_dict: Full NRLMSISE output structure
    """
    if not _LIB_LOADED:
        raise RuntimeError("NRLMSISE C library not loaded. Cannot call gtd7.")
    
    # Set defaults
    if f107a is None:
        f107a = f107
    if lst is None:
        # Estimate LST from longitude: 12h + (lon/15)
        lst = 12.0 + lon_deg / 15.0
    
    # Allocate C structures
    inp = ffi.new("struct nrlmsise_input *")
    out = ffi.new("struct nrlmsise_output *")
    flags = ffi.new("struct nrlmsise_flags *")
    
    # Set default flags (all switches on)
    for i in range(24):
        flags.switches[i] = 1
    
    # Fill input structure
    inp.year = year
    inp.doy = doy
    inp.sec = float(sec)
    inp.alt = float(alt_km)
    inp.g_lat = float(lat_deg)
    inp.g_long = float(lon_deg)
    inp.lst = float(lst)
    inp.f107A = float(f107a)
    inp.f107 = float(f107)
    inp.ap = float(ap)
    inp.ap_a = ffi.NULL  # Simplified - use ap directly
    
    # Call C function
    _C_LIB.gtd7(inp, flags, out)
    
    # Extract outputs
    # out.d[0]: He number density (cm⁻³)
    # out.d[1]: O number density (cm⁻³)
    # out.d[2]: N2 number density (cm⁻³)
    # out.d[3]: O2 number density (cm⁻³)
    # out.d[4]: Ar number density (cm⁻³)
    # out.d[5]: Total mass density (g/cm³) → convert to kg/m³
    # out.d[6]: H number density (cm⁻³)
    # out.d[7]: N number density (cm⁻³)
    # out.d[8]: Anomalous O number density (cm⁻³)
    
    total_density_g_cm3 = float(out.d[5])
    total_density_kg_m3 = total_density_g_cm3 * 1000.0  # Convert g/cm³ → kg/m³
    
    output_dict = {
        'total_density_kg_m3': total_density_kg_m3,
        'total_density_g_cm3': total_density_g_cm3,
        'he_density': float(out.d[0]),
        'o_density': float(out.d[1]),
        'n2_density': float(out.d[2]),
        'o2_density': float(out.d[3]),
        'ar_density': float(out.d[4]),
        'h_density': float(out.d[6]),
        'n_density': float(out.d[7]),
        'anomalous_o_density': float(out.d[8]),
        'exospheric_temp': float(out.t[0]),
        'temp_at_alt': float(out.t[1]),
    }
    
    return total_density_kg_m3, output_dict


def call_msise_simple(alt_km: float, lat_deg: float, lon_deg: float, f107: float) -> float:
    """
    Simplified host-side wrapper returning just total density.
    
    Args:
        alt_km: Altitude in kilometers
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        f107: Solar flux at 10.7 cm
        
    Returns:
        density: Total mass density in kg/m³
    """
    density, _ = call_msise_gtd7(alt_km, lat_deg, lon_deg, f107)
    return np.float32(density)


# =============================================================================
# JAX-Compatible Wrappers via pure_callback
# =============================================================================

@jax.jit
def jax_msise_density(alt: float, lat: float, lon: float, f107: float) -> float:
    """
    JAX-compatible wrapper for NRLMSISE-00 using pure_callback.
    
    Enables calling C-code from inside JAX/XLA execution graph.
    
    Args:
        alt: Altitude in km (JAX scalar)
        lat: Latitude in degrees (JAX scalar)
        lon: Longitude in degrees (JAX scalar)
        f107: Solar flux at 10.7 cm (JAX scalar)
        
    Returns:
        density: Total mass density in kg/m³ (JAX scalar)
        
    Example:
        >>> density = jax_msise_density(100.0, 28.5, -80.5, 150.0)
        >>> print(density)  # At 100km, ~5e-7 kg/m³
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.float32)
    
    return jax.pure_callback(
        call_msise_simple,  # Host-side function
        result_shape,        # Output shape/dtype specification
        alt, lat, lon, f107  # JAX arguments (passed to host)
    )


@jax.jit
def jax_msise_density_vmap(alts: jnp.ndarray, lat: float, lon: float, f107: float) -> jnp.ndarray:
    """
    Vectorized JAX wrapper for multiple altitudes.
    
    Efficient for computing density profiles.
    
    Args:
        alts: [N] array of altitudes in km
        lat: Latitude in degrees (scalar)
        lon: Longitude in degrees (scalar)
        f107: Solar flux (scalar)
        
    Returns:
        densities: [N] array of densities in kg/m³
    """
    return jax.vmap(lambda alt: jax_msise_density(alt, lat, lon, f107))(alts)


def jax_msise_density_batched(
    batch_alts: jnp.ndarray,
    batch_lats: jnp.ndarray,
    batch_lons: jnp.ndarray,
    batch_f107: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched JAX wrapper for H100 training efficiency.
    
    Processes entire batch with single callback (amortizes overhead).
    
    Args:
        batch_alts: [batch_size] altitudes
        batch_lats: [batch_size] latitudes
        batch_lons: [batch_size] longitudes
        batch_f107: [batch_size] solar flux values
        
    Returns:
        densities: [batch_size] densities
    """
    def host_batch_call(alts, lats, lons, f107s):
        """Host-side batch processing."""
        results = []
        for alt, lat, lon, f107 in zip(alts, lats, lons, f107s):
            density = call_msise_simple(float(alt), float(lat), float(lon), float(f107))
            results.append(density)
        return np.array(results, dtype=np.float32)
    
    result_shape = jax.ShapeDtypeStruct(batch_alts.shape, jnp.float32)
    
    return jax.pure_callback(
        host_batch_call,
        result_shape,
        batch_alts, batch_lats, batch_lons, batch_f107
    )


# =============================================================================
# Training Integration: Ground Truth Oracle
# =============================================================================

def compute_msise_ground_truth(
    altitudes: jnp.ndarray,
    latitude: float,
    longitude: float,
    f107: float,
    batch_size: int = 64,
) -> jnp.ndarray:
    """
    Compute ground truth densities from NRLMSISE-00 for training.
    
    This is the "Oracle" function used to train HydrostaticAtmosphere.
    
    Args:
        altitudes: [N] altitude grid
        latitude: Location latitude
        longitude: Location longitude
        f107: Solar flux
        batch_size: Batch size for callback (tune for H100 efficiency)
        
    Returns:
        ground_truth: [N] densities from NRLMSISE-00 C code
    """
    # Process in batches to amortize callback overhead
    num_altitudes = len(altitudes)
    results = []
    
    for i in range(0, num_altitudes, batch_size):
        batch_alts = altitudes[i:i+batch_size]
        batch_lats = jnp.full_like(batch_alts, latitude)
        batch_lons = jnp.full_like(batch_alts, longitude)
        batch_f107 = jnp.full_like(batch_alts, f107)
        
        batch_results = jax_msise_density_batched(
            batch_alts, batch_lats, batch_lons, batch_f107
        )
        results.append(batch_results)
    
    return jnp.concatenate(results)


def create_training_data_from_msise(
    key: jax.random.PRNGKey,
    num_samples: int = 10000,
    alt_range: Tuple[float, float] = (0, 500),
    lat_range: Tuple[float, float] = (-90, 90),
    lon_range: Tuple[float, float] = (-180, 180),
    f107_range: Tuple[float, float] = (70, 250),
) -> dict:
    """
    Generate training data from NRLMSISE-00 oracle.
    
    Creates dataset for HydrostaticAtmosphere training:
        Input: [altitude, lat, lon, F10.7]
        Target: density from gtd7()
    
    Args:
        key: JAX random key
        num_samples: Number of training samples
        alt_range: Altitude range (km)
        lat_range: Latitude range (deg)
        lon_range: Longitude range (deg)
        f107_range: Solar flux range (SFU)
        
    Returns:
        Dictionary with training data
    """
    # Sample random coordinates
    key_alt, key_lat, key_lon, key_f107 = jax.random.split(key, 4)
    
    alts = jax.random.uniform(key_alt, (num_samples,), 
                               minval=alt_range[0], maxval=alt_range[1])
    lats = jax.random.uniform(key_lat, (num_samples,),
                              minval=lat_range[0], maxval=lat_range[1])
    lons = jax.random.uniform(key_lon, (num_samples,),
                              minval=lon_range[0], maxval=lon_range[1])
    f107s = jax.random.uniform(key_f107, (num_samples,),
                               minval=f107_range[0], maxval=f107_range[1])
    
    # Get ground truth densities
    print(f"[NRLMSISE] Generating {num_samples} training samples from C oracle...")
    targets = jax_msise_density_batched(alts, lats, lons, f107s)
    
    return {
        'inputs': jnp.stack([alts, lats, lons, f107s], axis=-1),  # [N, 4]
        'targets': targets,  # [N]
        'metadata': {
            'source': 'NRLMSISE-00 C reference',
            'num_samples': num_samples,
        }
    }


# =============================================================================
# Verification & Diagnostics
# =============================================================================

def verify_bridge():
    """
    Verify CFFI bridge is working correctly.
    
    Run this before H100 training to ensure pipeline is functional.
    """
    print("=" * 70)
    print("NRLMSISE-00 Bridge Verification")
    print("=" * 70)
    
    # Check library loading
    if not _LIB_LOADED:
        print("❌ FAILED: C library not loaded")
        print("   Compile with: gcc -shared -o libmsise.so -fPIC -O3 nrlmsise-00.c")
        return False
    
    print(f"✅ Library loaded: {_LIB_LOADED}")
    
    # Test host-side call
    try:
        density, info = call_msise_gtd7(100.0, 28.5, -80.5, 150.0)
        print(f"✅ Host-side C call working")
        print(f"   Density at 100km: {density:.4e} kg/m³")
        print(f"   Temperature: {info['exospheric_temp']:.1f} K")
    except Exception as e:
        print(f"❌ Host-side C call failed: {e}")
        return False
    
    # Test JAX callback
    try:
        jax_density = jax_msise_density(100.0, 28.5, -80.5, 150.0)
        print(f"✅ JAX pure_callback working")
        print(f"   JAX density: {float(jax_density):.4e} kg/m³")
    except Exception as e:
        print(f"❌ JAX callback failed: {e}")
        return False
    
    # Test vectorized call
    try:
        alts = jnp.linspace(0, 300, 10)
        densities = jax_msise_density_vmap(alts, 28.5, -80.5, 150.0)
        print(f"✅ JAX vmap working")
        print(f"   Profile range: {float(densities.min()):.4e} - {float(densities.max()):.4e} kg/m³")
    except Exception as e:
        print(f"❌ JAX vmap failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("All verifications passed! Bridge is ready for H100 training.")
    print("=" * 70)
    return True


# =============================================================================
# Demo
# =============================================================================

def demo_nrlmsise_bridge():
    """Demonstrate NRLMSISE-00 bridge capabilities."""
    print("\n" + "=" * 70)
    print("NRLMSISE-00 C Reference Bridge Demo")
    print("=" * 70)
    
    # Verify bridge
    if not verify_bridge():
        print("\n⚠️  Bridge verification failed. Skipping demo.")
        return
    
    # Generate comparison data
    print("\n2. Density Profile Comparison:")
    print(f"   {'Altitude (km)':>12} | {'NRLMSISE-00 (C)':>16} | {'Notes':>20}")
    print("   " + "-" * 55)
    
    test_alts = [0, 50, 100, 200, 300, 400, 500]
    for alt in test_alts:
        density = call_msise_simple(alt, 28.5, -80.5, 150.0)
        
        if alt < 100:
            note = "Dense atmosphere"
        elif alt < 200:
            note = "Karman line region"
        else:
            note = "Thermosphere"
        
        print(f"   {alt:>12.0f} | {density:>16.4e} | {note:>20}")
    
    # Space weather effect
    print("\n3. Space Weather Sensitivity (at 300km):")
    print(f"   {'F10.7':>8} | {'Solar Activity':>15} | {'Density':>12}")
    print("   " + "-" * 40)
    
    for f107 in [70, 150, 250]:
        density = call_msise_simple(300.0, 28.5, -80.5, f107)
        if f107 < 100:
            activity = "Solar Minimum"
        elif f107 < 200:
            activity = "Moderate"
        else:
            activity = "Solar Maximum"
        print(f"   {f107:>8.0f} | {activity:>15} | {density:>12.4e}")
    
    # Training data generation
    print("\n4. Training Data Generation:")
    key = jax.random.PRNGKey(42)
    training_data = create_training_data_from_msise(key, num_samples=100, 
                                                      alt_range=(0, 500))
    print(f"   Generated {len(training_data['targets'])} samples")
    print(f"   Input shape: {training_data['inputs'].shape}")
    print(f"   Target range: [{float(training_data['targets'].min()):.4e}, "
          f"{float(training_data['targets'].max()):.4e}] kg/m³")
    
    print("\n" + "=" * 70)
    print("Bridge Capabilities:")
    print("  - CFFI: Low-latency binary link to legacy C code")
    print("  - pure_callback: Seamless JAX/XLA integration")
    print("  - Batch processing: H100 training efficiency")
    print("  - Ground Truth: High-fidelity oracle for ρ_nn training")
    print("=" * 70)


if __name__ == "__main__":
    demo_nrlmsise_bridge()
