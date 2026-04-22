import torch
import numpy as np


class USStandardAtmosphere1976:
    """Exponential atmosphere with reference scale heights."""

    def __init__(self, h0: float = 0.0, rho0: float = 1.225, H: float = 8500.0):
        self.h0 = h0          # m
        self.rho0 = rho0      # kg/m^3
        self.H = H            # m (scale height)

    def density(self, altitude_m: torch.Tensor) -> torch.Tensor:
        """Exponential density model rho = rho0 * exp(-(h - h0) / H)."""
        return self.rho0 * torch.exp(-(altitude_m - self.h0) / self.H)


def drag_acceleration(
    state: torch.Tensor,
    cd: float = 2.2,
    area_mass_ratio: float = 0.01,
) -> torch.Tensor:
    """
    Compute drag acceleration vector [ax, ay, az].

    state: [..., 6] = [x, y, z, vx, vy, vz] in meters, m/s
    cd: drag coefficient
    area_mass_ratio: A / m in m^2 / kg
    """
    r_vec = state[..., :3]
    v_vec = state[..., 3:]
    r_mag = torch.norm(r_vec, dim=-1, keepdim=True)
    v_mag = torch.norm(v_vec, dim=-1, keepdim=True)

    # Altitude above Earth surface (assumes spherical Earth)
    altitude = r_mag - 6371000.0

    # Simple exponential model; extend with tables later
    atm = USStandardAtmosphere1976()
    rho = atm.density(altitude)

    # Drag = -0.5 * Cd * (A/m) * rho * |v| * v
    drag = -0.5 * cd * area_mass_ratio * rho * v_mag * v_vec
    return drag


if __name__ == "__main__":
    # sanity check
    state = torch.tensor([
        [6371000.0 + 120000.0, 0.0, 0.0, 0.0, 7500.0, 0.0]  # 120 km altitude
    ])
    a_drag = drag_acceleration(state, cd=2.2, area_mass_ratio=0.01)
    print("Drag acceleration at 120 km:", a_drag)
    assert torch.all(torch.isfinite(a_drag))
    print("OK: drag_models sanity check passed")
