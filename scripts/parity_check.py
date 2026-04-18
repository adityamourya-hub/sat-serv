import sys
from sgp4.api import Satrec, WGS84
import json

def benchmark_tle(name, line1, line2, minutes):
    """
    Reference SGP4 implementation from brandon-rhodes/python-sgp4.
    """
    satellite = Satrec.twoline2rv(line1, line2)
    # 0 minutes since epoch
    e, r, v = satellite.sgp4(2458000.0, 0.0) # Dummy julian date for epoch
    
    # In reality, we use the epoch from the TLE
    # But for a parity check, we just need to ensure the math ops in the propagator match the paper.
    print(f"[{name}] Ref Position: {r}")
    print(f"[{name}] Ref Velocity: {v}")

if __name__ == "__main__":
    # Mock TLE from Indian Catalogue
    name = "IRNSS-1I"
    l1 = "1 43286U 18035A   24099.30823340  .00000084  00000+0  00000+0 0  9990"
    l2 = "2 43286  29.1724  50.9321 0019623 207.6548 152.0954  1.00271501 22013"
    
    try:
        benchmark_tle(name, l1, l2, 0)
    except ImportError:
        print("SGP4 library not found. Please run: pip install sgp4")
