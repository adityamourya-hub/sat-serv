package od

/**
 * Observation Models for Orbit Determination
 *
 * Processes raw sensor data (radar/optical) into
 * orbit determination measurements with proper noise modeling.
 *
 * Observation Types:
 * - Radar: Range, range-rate (Doppler), angles (AZ/EL)
 * - Optical: Right Ascension, Declination (angles only)
 * - Laser: Precise range (satellite laser ranging)
 *
 * References:
 * - Vallado: Fundamentals of Astrodynamics and Applications
 * - Orekit: Observation model architecture
 * - CCSDS ODM (Orbit Data Messages) standards
 */

import (
	"math"
	"time"

	"ssis/pkg/physics"
)

// ObservationType defines the measurement type
type ObservationType int

const (
	ObsRange          ObservationType = iota // Radar/laser range (km)
	ObsRangeRate                             // Doppler range-rate (km/s)
	ObsAzimuth                               // AZ angle (degrees)
	ObsElevation                             // EL angle (degrees)
	ObsRightAscension                        // RA (hours)
	ObsDeclination                           // DEC (degrees)
)

// Observation represents a single sensor measurement
type Observation struct {
	Type     ObservationType
	Value    float64 // Measured value
	Sigma    float64 // Measurement noise (1-sigma)
	Time     time.Time
	Station  GroundStation // Sensor location
	TargetID string        // Object being tracked

	// Residual (computed during OD)
	Residual float64 // O - C (observed minus computed)
	Computed float64 // Expected value from current state estimate
}

// GroundStation represents a sensor site
type GroundStation struct {
	ID        string
	Name      string
	Latitude  float64 // degrees
	Longitude float64 // degrees
	Altitude  float64 // meters above ellipsoid

	// ECI position (computed from lat/lon/alt)
	PositionECI [3]float64 // km

	// Sensor characteristics
	MinElevation float64 // degrees (horizon mask)
	RangeNoise   float64 // meters (1-sigma)
	AngleNoise   float64 // arcseconds (1-sigma)
}

// ComputeECI computes ground station ECI position at given time
func (gs *GroundStation) ComputeECI(jd float64) [3]float64 {
	// Use existing Greenwich sidereal time calculation
	gst := GreenwichSiderealTime(jd)

	// Convert lat/lon to ECI
	latRad := gs.Latitude * math.Pi / 180.0
	lonRad := gs.Longitude * math.Pi / 180.0

	// Earth radius at latitude (WGS84 approximation)
	Re := 6378.137         // km (equatorial)
	e2 := 0.00669437999014 // eccentricity squared

	// Radius of curvature in prime vertical
	N := Re / math.Sqrt(1-e2*math.Sin(latRad)*math.Sin(latRad))

	// ECEF coordinates
	ecefX := (N + gs.Altitude/1000.0) * math.Cos(latRad) * math.Cos(lonRad)
	ecefY := (N + gs.Altitude/1000.0) * math.Cos(latRad) * math.Sin(lonRad)
	ecefZ := (N*(1-e2) + gs.Altitude/1000.0) * math.Sin(latRad)

	// Rotate to ECI using GST
	cosGST := math.Cos(gst)
	sinGST := math.Sin(gst)

	eciX := ecefX*cosGST - ecefY*sinGST
	eciY := ecefX*sinGST + ecefY*cosGST
	eciZ := ecefZ

	gs.PositionECI = [3]float64{eciX, eciY, eciZ}
	return gs.PositionECI
}

// GreenwichSiderealTime computes GMST in radians
func GreenwichSiderealTime(jd float64) float64 {
	// GMST at 0h UT (seconds)
	gmst0 := 6.697374558 + 0.06570982441908*float64(int(jd-2451545.0)) +
		1.00273790935*float64(jd-float64(int(jd))-0.5)*24.0

	// Convert to radians
	gmst := gmst0 * 15.0 * math.Pi / 180.0 // 15° per hour

	// Normalize to [0, 2π)
	for gmst < 0 {
		gmst += 2 * math.Pi
	}
	for gmst >= 2*math.Pi {
		gmst -= 2 * math.Pi
	}

	return gmst
}

// ObservationModel computes expected observation from state
type ObservationModel struct {
	ObsType ObservationType
	Station GroundStation
}

// ComputeExpected calculates what the observation should be given satellite state
func (om *ObservationModel) ComputeExpected(state physics.State, jd float64) float64 {
	// Update station ECI position
	stationECI := om.Station.ComputeECI(jd)

	// Vector from station to satellite
	dx := state.Pos[0] - stationECI[0]
	dy := state.Pos[1] - stationECI[1]
	dz := state.Pos[2] - stationECI[2]

	range_km := math.Sqrt(dx*dx + dy*dy + dz*dz)

	switch om.ObsType {
	case ObsRange:
		return range_km

	case ObsRangeRate:
		// Range rate = relative velocity dot LOS unit vector
		// Compute station velocity from Earth rotation (ω = 7.2921e-5 rad/s)
		omega := 7.2921159e-5                 // Earth rotation rate (rad/s)
		stationVelX := -omega * stationECI[1] // ω × r in ECI
		stationVelY := omega * stationECI[0]
		stationVelZ := 0.0 // No Z rotation in ECI frame

		dvx := state.Vel[0] - stationVelX
		dvy := state.Vel[1] - stationVelY
		dvz := state.Vel[2] - stationVelZ

		return (dx*dvx + dy*dvy + dz*dvz) / range_km

	case ObsAzimuth, ObsElevation:
		// Convert to topocentric coordinates (East-North-Up frame)
		// Compute unit vectors for ENU frame at station location
		latRad := om.Station.Latitude * math.Pi / 180.0
		lonRad := om.Station.Longitude * math.Pi / 180.0

		// ENU basis vectors in ECI
		// East: [-sin(lon), cos(lon), 0]
		// North: [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
		// Up: [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]
		east := [3]float64{-math.Sin(lonRad), math.Cos(lonRad), 0}
		north := [3]float64{-math.Sin(latRad) * math.Cos(lonRad), -math.Sin(latRad) * math.Sin(lonRad), math.Cos(latRad)}
		up := [3]float64{math.Cos(latRad) * math.Cos(lonRad), math.Cos(latRad) * math.Sin(lonRad), math.Sin(latRad)}

		// LOS unit vector in ECI
		los := [3]float64{dx / range_km, dy / range_km, dz / range_km}

		// Project LOS onto ENU frame
		losEast := los[0]*east[0] + los[1]*east[1] + los[2]*east[2]
		losNorth := los[0]*north[0] + los[1]*north[1] + los[2]*north[2]
		losUp := los[0]*up[0] + los[1]*up[1] + los[2]*up[2]

		// Elevation = arcsin(losUp)
		elevation := math.Asin(losUp) * 180.0 / math.Pi

		if om.ObsType == ObsElevation {
			return elevation
		}

		// Azimuth = atan2(losEast, losNorth)
		azimuth := math.Atan2(losEast, losNorth) * 180.0 / math.Pi
		if azimuth < 0 {
			azimuth += 360.0
		}
		return azimuth

	case ObsRightAscension:
		// RA = atan2(y, x) in equatorial coordinates
		ra := math.Atan2(dy, dx) * 12.0 / math.Pi // Convert to hours (0-24)
		if ra < 0 {
			ra += 24.0
		}
		return ra

	case ObsDeclination:
		// DEC = atan2(z, sqrt(x²+y²))
		dec := math.Atan2(dz, math.Sqrt(dx*dx+dy*dy)) * 180.0 / math.Pi
		return dec

	default:
		return 0.0
	}
}

// ComputeResidual calculates O - C (observed minus computed)
func (om *ObservationModel) ComputeResidual(obs Observation, state physics.State, jd float64) float64 {
	computed := om.ComputeExpected(state, jd)
	return obs.Value - computed
}

// Predefined Ground Stations (IAF/ISRO network)

// NewNETRARadar creates NETRA radar station (Sriharikota area)
func NewNETRARadar() GroundStation {
	return GroundStation{
		ID:           "NETRA-R1",
		Name:         "NETRA Radar (Sriharikota)",
		Latitude:     13.7331, // degrees N
		Longitude:    80.2016, // degrees E
		Altitude:     10.0,    // meters
		MinElevation: 5.0,     // degrees
		RangeNoise:   10.0,    // meters (10m range accuracy)
		AngleNoise:   100.0,   // arcseconds
	}
}

// NewIAFOptical creates IAF optical tracking station
func NewIAFOptical() GroundStation {
	return GroundStation{
		ID:           "IAF-OPT1",
		Name:         "IAF Optical Tracking (Pune)",
		Latitude:     18.5204,
		Longitude:    73.8567,
		Altitude:     560.0,
		MinElevation: 20.0,
		RangeNoise:   0.0, // No range measurement
		AngleNoise:   5.0, // 5 arcsec (good optical)
	}
}

// NewIRSOTracking creates ISRO tracking station
func NewISROTracking() GroundStation {
	return GroundStation{
		ID:           "ISRO-TRK",
		Name:         "ISRO Master Control Facility (Hassan)",
		Latitude:     13.005,
		Longitude:    76.099,
		Altitude:     950.0,
		MinElevation: 5.0,
		RangeNoise:   5.0,
		AngleNoise:   50.0,
	}
}

// ObservationBatch represents a set of observations for OD
type ObservationBatch struct {
	Observations []Observation
	TargetID     string
}

// NewObservationBatch creates batch from observations
func NewObservationBatch(obs []Observation) *ObservationBatch {
	return &ObservationBatch{
		Observations: obs,
		TargetID:     obs[0].TargetID,
	}
}

// TimeSpan returns JD range of observations
func (ob *ObservationBatch) TimeSpan() (startJD, endJD float64) {
	if len(ob.Observations) == 0 {
		return 0, 0
	}

	start := ob.Observations[0].Time
	end := ob.Observations[0].Time

	for _, obs := range ob.Observations {
		if obs.Time.Before(start) {
			start = obs.Time
		}
		if obs.Time.After(end) {
			end = obs.Time
		}
	}

	return JulianDateFromTime(start), JulianDateFromTime(end)
}

// JulianDateFromTime converts time.Time to Julian Date
func JulianDateFromTime(t time.Time) float64 {
	// Simple conversion (use existing TimeToJulianDate from orbital if available)
	unixSec := float64(t.Unix())
	jd := 2440587.5 + unixSec/86400.0
	return jd
}
