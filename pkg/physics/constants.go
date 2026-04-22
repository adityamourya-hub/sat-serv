package physics

import "math"

/**
 * Universal Physics Constants (IAU / WGS-84 Standards)
 * Locked for Sovereign Engine integrity.
 */
const (
	// Gravitational Parameters
	EARTH_MU_IAU = 398600.4418      // Earth's gravitational parameter (km^3/s^2)
	MOON_MU_IAU  = 4902.800066      // Moon's gravitational parameter (km^3/s^2)
	SUN_MU_IAU   = 1.32712440018e11 // Sun's gravitational parameter (km^3/s^2)

	// Earth Dimensions
	EARTH_R_EQUATORIAL = 6378.137            // Semi-major axis (km) - WGS84
	EARTH_R_POLAR      = 6356.7523142        // Semi-minor axis (km) - WGS84
	EARTH_F            = 1.0 / 298.257223563 // Flattening factor

	// Earth Rotation
	EARTH_OMEGA = 7.292115e-5 // Mean angular velocity (rad/s)

	// Time Constants
	SEC_PER_DAY = 86400.0
	JD_OFFSET   = 2440587.5 // Offset from Unix Epoch (1970-01-01) to JD

	// Orbital Harmonics (WGS-84 / IAU Standards)
	J2 = 1.082626173852220e-3
	J3 = -2.532656485332240e-6
	J4 = -1.619621591367000e-6

	// Environmental Constants
	SOLAR_PRESSURE = 4.56e-6 // N/m^2 at 1 AU
	AU_KM          = 149597870.7

	// Math Helpers
	RAD_PER_DEG = math.Pi / 180.0
	DEG_PER_RAD = 180.0 / math.Pi
)
