// Package analytics provides blind spot detection for ground site protection
// Calculates when adversary satellites cannot see strategic IAF locations
package analytics

import (
	"math"
	"sort"
	"time"

	"ssis/pkg/orbital"
)

// GroundSite represents a strategic IAF location
type GroundSite struct {
	Name         string
	Lat          float64 // degrees
	Lon          float64 // degrees
	Alt          float64 // km
	MinElevation float64 // degrees (usually 10-15°)
}

// AccessWindow defines a period when a satellite can see a ground site
type AccessWindow struct {
	SatelliteID   int32
	SatelliteName string
	StartTime     time.Time
	EndTime       time.Time
	MaxElevation  float64 // degrees
	SensorType    string  // "OPTICAL", "SAR", "ALL"
}

// BlindSpot represents a period when no adversary can see the site
type BlindSpot struct {
	StartTime     time.Time
	EndTime       time.Time
	Duration      time.Duration
	ProtectedFrom string // "OPTICAL", "RADAR", "ALL"
}

// Default IAF Strategic Sites
func IAFStrategicSites() []GroundSite {
	return []GroundSite{
		{
			Name:         "Sriharikota_Launch",
			Lat:          13.7331,
			Lon:          80.2358,
			Alt:          0.01,
			MinElevation: 10.0,
		},
		{
			Name:         "Abdul_Kalam_Test_Range",
			Lat:          20.7557,
			Lon:          87.0926,
			Alt:          0.01,
			MinElevation: 15.0,
		},
		{
			Name:         "Bengaluru_ISRO_HQ",
			Lat:          12.9716,
			Lon:          77.5946,
			Alt:          0.92,
			MinElevation: 10.0,
		},
		{
			Name:         "Siliguri_Corridor",
			Lat:          26.7271,
			Lon:          88.3953,
			Alt:          0.10,
			MinElevation: 10.0,
		},
		{
			Name:         "Leh_LAD",
			Lat:          34.1526,
			Lon:          77.5771,
			Alt:          3.52,
			MinElevation: 15.0, // High altitude needs higher elevation
		},
	}
}

// LookAngles represents topocentric coordinates
type LookAngles struct {
	Azimuth   float64 // degrees, 0=North, 90=East
	Elevation float64 // degrees, 0=horizon, 90=zenith
	Range     float64 // km
}

// CalculateLookAngles computes azimuth, elevation, and range from site to satellite
// Satellite position should be in ECEF coordinates
func CalculateLookAngles(site GroundSite, satX, satY, satZ float64) LookAngles {
	// Convert site to ECEF
	siteX, siteY, siteZ := GeodeticToECEF(site.Lat, site.Lon, site.Alt)

	// Vector from site to satellite
	dx := satX - siteX
	dy := satY - siteY
	dz := satZ - siteZ

	// Range
	rangeKm := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Convert site lat/lon to radians
	latRad := site.Lat * math.Pi / 180.0
	lonRad := site.Lon * math.Pi / 180.0

	// Transform to local ENU (East-North-Up) frame
	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	sinLon := math.Sin(lonRad)
	cosLon := math.Cos(lonRad)

	// ENU transformation matrix applied to dx,dy,dz
	east := -sinLon*dx + cosLon*dy
	north := -sinLat*cosLon*dx - sinLat*sinLon*dy + cosLat*dz
	up := cosLat*cosLon*dx + cosLat*sinLon*dy + sinLat*dz

	// Elevation angle
	elevation := math.Atan2(up, math.Sqrt(east*east+north*north))

	// Azimuth (from North)
	azimuth := math.Atan2(east, north)

	// Normalize azimuth to 0-360
	azimuth = azimuth * 180.0 / math.Pi
	if azimuth < 0 {
		azimuth += 360.0
	}

	return LookAngles{
		Azimuth:   azimuth,
		Elevation: elevation * 180.0 / math.Pi,
		Range:     rangeKm,
	}
}

// CalculateAccessWindow predicts when a satellite can see a ground site
func CalculateAccessWindow(tleLine1, tleLine2 string, site GroundSite, duration time.Duration, step time.Duration) AccessWindow {
	var window AccessWindow
	inView := false

	for t := time.Duration(0); t < duration; t += step {
		targetTime := time.Now().Add(t)

		// Propagate satellite (simplified - would use full SGP4 in production)
		satX, satY, satZ := PropagateToECEF(tleLine1, tleLine2, targetTime)

		// Calculate look angles
		angles := CalculateLookAngles(site, satX, satY, satZ)

		if angles.Elevation >= site.MinElevation {
			if !inView {
				inView = true
				window.StartTime = targetTime
				window.MaxElevation = angles.Elevation
			}
			if angles.Elevation > window.MaxElevation {
				window.MaxElevation = angles.Elevation
			}
		} else {
			if inView {
				inView = false
				window.EndTime = targetTime
				return window
			}
		}
	}

	// If still in view at end of period
	if inView {
		window.EndTime = time.Now().Add(duration)
	}

	return window
}

// FindBlindSpots calculates time periods when no adversary can see the site
func FindBlindSpots(windows []AccessWindow, site GroundSite, duration time.Duration, sensorType string) []BlindSpot {
	if len(windows) == 0 {
		// No windows means always blind (ideal case)
		return []BlindSpot{
			{
				StartTime:     time.Now(),
				EndTime:       time.Now().Add(duration),
				Duration:      duration,
				ProtectedFrom: sensorType,
			},
		}
	}

	// Sort windows by start time
	SortAccessWindows(windows)

	// Merge overlapping windows
	merged := MergeAccessWindows(windows)

	// Check if merged is empty after merging
	if len(merged) == 0 {
		return []BlindSpot{
			{
				StartTime:     time.Now(),
				EndTime:       time.Now().Add(duration),
				Duration:      duration,
				ProtectedFrom: sensorType,
			},
		}
	}

	// Find gaps between merged windows
	var blindSpots []BlindSpot
	now := time.Now()
	endTime := now.Add(duration)

	// Gap from start to first window
	if merged[0].StartTime.After(now) {
		gap := merged[0].StartTime.Sub(now)
		if gap > time.Minute {
			blindSpots = append(blindSpots, BlindSpot{
				StartTime:     now,
				EndTime:       merged[0].StartTime,
				Duration:      gap,
				ProtectedFrom: sensorType,
			})
		}
	}

	// Gaps between windows
	for i := 0; i < len(merged)-1; i++ {
		gap := merged[i+1].StartTime.Sub(merged[i].EndTime)
		if gap > time.Minute {
			blindSpots = append(blindSpots, BlindSpot{
				StartTime:     merged[i].EndTime,
				EndTime:       merged[i+1].StartTime,
				Duration:      gap,
				ProtectedFrom: sensorType,
			})
		}
	}

	// Gap from last window to end
	if merged[len(merged)-1].EndTime.Before(endTime) {
		gap := endTime.Sub(merged[len(merged)-1].EndTime)
		if gap > time.Minute {
			blindSpots = append(blindSpots, BlindSpot{
				StartTime:     merged[len(merged)-1].EndTime,
				EndTime:       endTime,
				Duration:      gap,
				ProtectedFrom: sensorType,
			})
		}
	}

	return blindSpots
}

// GeodeticToECEF converts lat/lon/alt to ECEF coordinates
func GeodeticToECEF(lat, lon, alt float64) (x, y, z float64) {
	// WGS84 parameters
	const a = 6378.137            // Semi-major axis (km)
	const f = 1.0 / 298.257223563 // Flattening
	const e2 = 2*f - f*f          // First eccentricity squared

	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	// Prevent division by zero at poles by bounding denominator
	denominator := 1.0 - e2*math.Sin(latRad)*math.Sin(latRad)
	if denominator < 1e-10 {
		denominator = 1e-10
	}
	N := a / math.Sqrt(denominator)

	x = (N + alt) * math.Cos(latRad) * math.Cos(lonRad)
	y = (N + alt) * math.Cos(latRad) * math.Sin(lonRad)
	z = (N*(1.0-e2) + alt) * math.Sin(latRad)

	return x, y, z
}

// PropagateToECEF propagates TLE to ECEF coordinates at given time
// Uses SGP4 for TEME propagation, then converts to ECEF using GMST
func PropagateToECEF(tleLine1, tleLine2 string, t time.Time) (x, y, z float64) {
	// Parse TLE lines
	tle, err := orbital.ParseTLE(tleLine1 + "\n" + tleLine2)
	if err != nil {
		// Return error indicator (invalid coordinates)
		return 0, 0, 0
	}

	// Use SGP4 to propagate to TEME frame
	prop := orbital.NewSGP4Propagator(tle)
	if prop == nil {
		return 0, 0, 0
	}

	// Calculate minutes since TLE epoch
	epoch := time.Date(tle.EpochYear, 1, 1, 0, 0, 0, 0, time.UTC)
	epoch = epoch.Add(time.Duration((tle.EpochDay - 1.0) * 24.0 * float64(time.Hour)))
	deltaMin := t.Sub(epoch).Minutes()
	state, err := prop.Propagate(deltaMin)
	if err != nil {
		return 0, 0, 0
	}

	// Convert TEME to ECEF using GMST (Greenwich Mean Sidereal Time)
	temeX, temeY, temeZ := state.Pos[0], state.Pos[1], state.Pos[2]
	gmst := calculateGMST(t)

	// Rotation from TEME to ECEF (rotation around Z-axis by GMST)
	cosGMST := math.Cos(gmst)
	sinGMST := math.Sin(gmst)

	x = cosGMST*temeX + sinGMST*temeY
	y = -sinGMST*temeX + cosGMST*temeY
	z = temeZ

	return x, y, z
}

// calculateGMST computes Greenwich Mean Sidereal Time in radians
func calculateGMST(t time.Time) float64 {
	// Julian Date calculation
	// Simplified formula: GMST = 280.46061837 + 360.98564736629*(JD - 2451545.0)
	// Result in degrees, convert to radians

	unixSec := t.Unix()
	jd := float64(unixSec)/86400.0 + 2440587.5
	d := jd - 2451545.0

	gmstDeg := 280.46061837 + 360.98564736629*d
	gmstDeg = math.Mod(gmstDeg, 360.0)
	if gmstDeg < 0 {
		gmstDeg += 360.0
	}

	return gmstDeg * math.Pi / 180.0
}

// SortAccessWindows sorts windows by start time using O(n log n) sort
func SortAccessWindows(windows []AccessWindow) {
	sort.Slice(windows, func(i, j int) bool {
		return windows[i].StartTime.Before(windows[j].StartTime)
	})
}

// MergeAccessWindows merges overlapping windows
func MergeAccessWindows(windows []AccessWindow) []AccessWindow {
	if len(windows) == 0 {
		return nil
	}

	merged := []AccessWindow{windows[0]}

	for i := 1; i < len(windows); i++ {
		last := &merged[len(merged)-1]
		current := windows[i]

		// Check for overlap or adjacency (within 1 minute)
		if current.StartTime.Before(last.EndTime.Add(time.Minute)) {
			// Merge windows
			if current.EndTime.After(last.EndTime) {
				last.EndTime = current.EndTime
			}
			if current.MaxElevation > last.MaxElevation {
				last.MaxElevation = current.MaxElevation
			}
		} else {
			merged = append(merged, current)
		}
	}

	return merged
}
