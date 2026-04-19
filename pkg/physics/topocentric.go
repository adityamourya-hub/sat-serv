package physics

import (
	"math"
)

/**
 * ECIToAzEl computes Azimuth/Elevation from ground station to satellite.
 * Essential for telescope slewing commands.
 * 
 * Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
 * Elevation: 0° = Horizon, 90° = Zenith
 */
func ECIToAzEl(satX, satY, satZ float64, obsLat, obsLon, obsAlt float64, jd float64) (azimuth, elevation float64) {
	// 1. Convert observer geodetic to ECEF
	obsX, obsY, obsZ := GeodeticToECEFWGS84(obsLat*RAD_PER_DEG, obsLon*RAD_PER_DEG, obsAlt)
	
	// 2. Convert satellite ECI to ECEF (account for Earth rotation)
	satECEFX, satECEFY, satECEFZ := ECIToECEF(satX, satY, satZ, jd)
	
	// 3. Relative vector in ECEF
	dx := satECEFX - obsX
	dy := satECEFY - obsY
	dz := satECEFZ - obsZ
	
	// 4. ECEF to SEZ (South-East-Zenith) rotation
	sinLat := math.Sin(obsLat * RAD_PER_DEG)
	cosLat := math.Cos(obsLat * RAD_PER_DEG)
	sinLon := math.Sin(obsLon * RAD_PER_DEG)
	cosLon := math.Cos(obsLon * RAD_PER_DEG)
	
	// Rotation matrix ECEF→SEZ
	south := -sinLat*cosLon*dx - sinLat*sinLon*dy + cosLat*dz
	east := -sinLon*dx + cosLon*dy
	zenith := cosLat*cosLon*dx + cosLat*sinLon*dy + sinLat*dz
	
	// 5. Azimuth (from North, clockwise) and Elevation
	rangeVec := math.Sqrt(south*south + east*east + zenith*zenith)
	elevation = math.Asin(zenith / rangeVec) * DEG_PER_RAD
	
	// Azimuth from South (standard convention), convert to North-based
	azimuthFromSouth := math.Atan2(east, -south) * DEG_PER_RAD
	azimuth = azimuthFromSouth + 180.0
	if azimuth >= 360.0 {
		azimuth -= 360.0
	}
	if azimuth < 0 {
		azimuth += 360.0
	}
	
	return azimuth, elevation
}

/**
 * AzElToLookVector converts Azimuth/Elevation to unit look vector in ECEF.
 * Used for line-of-sight calculations.
 */
func AzElToLookVector(azimuth, elevation float64, obsLat, obsLon float64) (lx, ly, lz float64) {
	azRad := azimuth * RAD_PER_DEG
	elRad := elevation * RAD_PER_DEG
	latRad := obsLat * RAD_PER_DEG
	lonRad := obsLon * RAD_PER_DEG
	
	// Look vector in local (SEZ) coordinates
	south := -math.Cos(elRad) * math.Cos(azRad)
	east := math.Cos(elRad) * math.Sin(azRad)
	zenith := math.Sin(elRad)
	
	// Rotate to ECEF
	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	sinLon := math.Sin(lonRad)
	cosLon := math.Cos(lonRad)
	
	lx = -sinLat*cosLon*south - sinLon*east + cosLat*cosLon*zenith
	ly = -sinLat*sinLon*south + cosLon*east + cosLat*sinLon*zenith
	lz = cosLat*south + sinLat*zenith
	
	return lx, ly, lz
}

/**
 * CalculateTrackingRate computes the angular rate of change (deg/s) for continuous tracking.
 * Used for "track mode" on high-velocity LEO objects.
 */
func CalculateTrackingRate(satState State, satVel State, obsLat, obsLon, obsAlt float64, jd float64) (azRate, elRate float64) {
	// Current Az/El
	az1, el1 := ECIToAzEl(satState.Pos[0], satState.Pos[1], satState.Pos[2], obsLat, obsLon, obsAlt, jd)
	
	// Future Az/El (1 second later)
	dt := 1.0 / 86400.0 // 1 second in days
	futureJD := jd + dt
	
	// Propagate position by 1 second
	futureX := satState.Pos[0] + satVel.Vel[0]*1.0
	futureY := satState.Pos[1] + satVel.Vel[1]*1.0
	futureZ := satState.Pos[2] + satVel.Vel[2]*1.0
	
	az2, el2 := ECIToAzEl(futureX, futureY, futureZ, obsLat, obsLon, obsAlt, futureJD)
	
	// Calculate rates (handle 360° wrap-around)
	azRate = az2 - az1
	if azRate > 180.0 {
		azRate -= 360.0
	}
	if azRate < -180.0 {
		azRate += 360.0
	}
	
	elRate = el2 - el1
	
	return azRate, elRate
}

/**
 * CheckVisibility determines if satellite is above horizon and within sensor constraints.
 */
func CheckVisibility(satX, satY, satZ float64, obsLat, obsLon, obsAlt float64, jd float64, minElevation float64) (visible bool, azimuth, elevation float64) {
	az, el := ECIToAzEl(satX, satY, satZ, obsLat, obsLon, obsAlt, jd)
	return el >= minElevation, az, el
}
