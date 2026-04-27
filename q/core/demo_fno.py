"""Demo script for FNO-based Hydrostatic Atmosphere Operator."""

import torch
from hno_pytorch import HydrostaticAtmosphereFNO


def demo_fno_atmosphere():
    """Demonstrate the FNO-based atmosphere operator."""
    print("FNO-based Hydrostatic Atmosphere Operator Demo")

    # Create model
    model = HydrostaticAtmosphereFNO(modes=16, width=64)
    model.eval()

    print("\n1. Model Architecture:")
    print(f"   Fourier modes: {model.fno.modes}")
    print(f"   Hidden width: {model.fno.width}")
    print(f"   Input: [h, F10.7, Ap, lat, lon, DOY]")
    print(f"   Output: density (kg/m³)")

    # Create altitude grid
    seq_len = 100
    altitude_km = torch.linspace(0, 500, seq_len)

    # Create sample input (batch_size=1)
    # Conditions: F10.7=150, Ap=15, lat=28.5, lon=-80.5, DOY=100
    f107 = 150.0
    ap = 15.0
    lat = 28.5
    lon = -80.5
    doy = 100

    x = torch.zeros(1, seq_len, 6)
    x[0, :, 0] = altitude_km
    x[0, :, 1] = f107
    x[0, :, 2] = ap
    x[0, :, 3] = lat
    x[0, :, 4] = lon
    x[0, :, 5] = doy

    print("\n2. Test Conditions:")
    print(f"   Altitude range: 0-500 km")
    print(f"   F10.7: {f107} SFU")
    print(f"   Ap: {ap}")
    print(f"   Location: {lat}°N, {lon}°E")
    print(f"   Day of year: {doy}")

    # Forward pass without projection
    print("\n3. Forward Pass (without projection):")
    with torch.no_grad():
        result_no_proj = model(x, apply_projection=False)

    rho_no_proj = result_no_proj['density'][0]
    print(f"   Surface density: {rho_no_proj[0]:.4e} kg/m³")
    print(f"   100 km density: {rho_no_proj[20]:.4e} kg/m³")
    print(f"   400 km density: {rho_no_proj[80]:.4e} kg/m³")

    # Forward pass with projection
    print("\n4. Forward Pass (with hydrostatic projection):")
    with torch.no_grad():
        result_proj = model(x, apply_projection=True)

    rho_proj = result_proj['density'][0]
    pressure = result_proj['pressure'][0]
    scale_height = result_proj['scale_height'][0]
    residual = result_proj['residual'][0]

    print(f"   Surface density: {rho_proj[0]:.4e} kg/m³")
    print(f"   100 km density: {rho_proj[20]:.4e} kg/m³")
    print(f"   400 km density: {rho_proj[80]:.4e} kg/m³")
    print(f"   100 km pressure: {pressure[20]:.4e} Pa")
    print(f"   100 km scale height: {scale_height[20]:.2f} km")
    print(f"   Mean hydrostatic residual: {residual.mean():.4e}")

    # Compare
    print("\n5. Projection Effect:")
    diff = torch.abs(rho_proj - rho_no_proj).mean()
    print(f"   Mean correction: {diff:.4e} kg/m³")

    print("\n6. Technical Advantages:")
    print("   - Zero latency: Single GPU forward pass replaces C-calls")
    print("   - Smooth gradients: C∞ continuous for symplectic integration")
    print("   - F10.7/Ap sensitivity: Learns atmospheric breathing")
    print("   - Hard constraint: Hydrostatic equilibrium enforced")

    print("\n7. PHS Integration:")
    print("   Dissipation matrix R(x) = DragCoeff · ρ_nn(x) · ||v||")
    print("   → Jacobian of Hamiltonian sees density gradient")
    print("   → Automatic drag adjustment with altitude")

    print()


if __name__ == "__main__":
    demo_fno_atmosphere()
