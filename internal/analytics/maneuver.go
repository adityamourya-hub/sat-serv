// Package analytics provides maneuver detection and orbital analysis
// IAF-grade detection of non-natural orbital changes using Z-score analysis
package analytics

import (
	"math"
	"time"
)

// ManeuverReport details a detected orbital maneuver
type ManeuverReport struct {
	NoradCatID    int32
	ObjectName    string
	Metric        string  // "Inclination", "MeanMotion", "Eccentricity", "RAAN", "BStar"
	CurrentValue  float64
	Mean          float64
	StdDev        float64
	Delta         float64
	ZScore        float64 // Sigma deviance
	AlertLevel    string  // "YELLOW", "RED"
	DetectedAt    time.Time
}

// MetricThresholds define Z-score thresholds for each orbital element
type MetricThresholds struct {
	Inclination float64 // Degrees
	MeanMotion  float64 // rev/day
	Eccentricity float64
	RAAN        float64 // Degrees
	BStar       float64 // 1/ER
}

// DefaultThresholds for IAF operations
func DefaultThresholds() MetricThresholds {
	return MetricThresholds{
		Inclination:  3.0,  // 3 sigma for inclination (plane changes are rare)
		MeanMotion:   2.5,  // 2.5 sigma for mean motion (station-keeping is common)
		Eccentricity: 3.0,
		RAAN:         2.0,  // RAAN drifts naturally due to J2
		BStar:        4.0,  // High threshold - drag varies with solar activity
	}
}

// OrbitalHistory represents historical data for a satellite
type OrbitalHistory struct {
	NoradCatID   int32
	ObjectName   string
	Epochs       []time.Time
	Inclinations []float64
	MeanMotions  []float64
	Eccentricities []float64
	RAANs        []float64
	BStars       []float64
}

// DetectManeuvers analyzes orbital history and returns any detected maneuvers
func DetectManeuvers(history *OrbitalHistory, thresholds MetricThresholds) []ManeuverReport {
	var reports []ManeuverReport

	// Need at least 5 data points for statistical significance
	if len(history.Inclinations) < 5 {
		return reports
	}

	// Check each orbital element
	if report := checkMetric(history.NoradCatID, history.ObjectName, "Inclination",
		history.Inclinations, thresholds.Inclination); report != nil {
		reports = append(reports, *report)
	}

	if report := checkMetric(history.NoradCatID, history.ObjectName, "MeanMotion",
		history.MeanMotions, thresholds.MeanMotion); report != nil {
		reports = append(reports, *report)
	}

	if report := checkMetric(history.NoradCatID, history.ObjectName, "Eccentricity",
		history.Eccentricities, thresholds.Eccentricity); report != nil {
		reports = append(reports, *report)
	}

	if report := checkMetric(history.NoradCatID, history.ObjectName, "RAAN",
		history.RAANs, thresholds.RAAN); report != nil {
		reports = append(reports, *report)
	}

	if report := checkMetric(history.NoradCatID, history.ObjectName, "BStar",
		history.BStars, thresholds.BStar); report != nil {
		reports = append(reports, *report)
	}

	return reports
}

// checkMetric performs Z-score analysis on a single orbital element
func checkMetric(noradID int32, objectName, metric string, values []float64, threshold float64) *ManeuverReport {
	if len(values) < 5 {
		return nil
	}

	// Current value is the most recent (index 0 from DESC query)
	currentValue := values[0]

	// Calculate mean and std dev of historical window (excluding current)
	historical := values[1:]
	mean, stdDev := calculateStats(historical)

	if stdDev == 0 {
		return nil // No variation - can't detect anomaly
	}

	// Calculate Z-score
	delta := math.Abs(currentValue - mean)
	zScore := delta / stdDev

	// Check threshold
	if zScore >= threshold {
		level := "YELLOW"
		if zScore >= threshold*2.0 || zScore > 6.0 {
			level = "RED" // Extreme deviation
		}

		return &ManeuverReport{
			NoradCatID:   noradID,
			ObjectName:   objectName,
			Metric:       metric,
			CurrentValue: currentValue,
			Mean:         mean,
			StdDev:       stdDev,
			Delta:        delta,
			ZScore:       zScore,
			AlertLevel:   level,
			DetectedAt:   time.Now(),
		}
	}

	return nil
}

// calculateStats computes mean and standard deviation
func calculateStats(values []float64) (mean, stdDev float64) {
	if len(values) == 0 {
		return 0, 0
	}

	var sum, sumSq float64
	for _, v := range values {
		sum += v
		sumSq += v * v
	}

	n := float64(len(values))
	mean = sum / n

	// Population standard deviation
	variance := (sumSq / n) - (mean * mean)
	if variance < 0 {
		variance = 0 // Handle floating point errors
	}
	stdDev = math.Sqrt(variance)

	return mean, stdDev
}

// ClassifyManeuver provides tactical interpretation of detected change
func ClassifyManeuver(report ManeuverReport) string {
	switch report.Metric {
	case "Inclination":
		if report.Delta > 0.5 {
			return "TACTICAL_PLANE_CHANGE"
		} else if report.Delta > 0.1 {
			return "PHASING_MANEUVER"
		}
		return "STATION_KEEPING"

	case "MeanMotion":
		if math.Abs(report.Delta) > 0.5 {
			return "ALTITUDE_ADJUSTMENT"
		}
		return "ORBIT_MAINTENANCE"

	case "RAAN":
		if report.ZScore > 3.0 {
			return "INTENTIONAL_PRECESSION_SHIFT"
		}
		return "J2_DRIFT"

	case "Eccentricity":
		if report.Delta > 0.01 {
			return "ORBIT_CIRCULARIZATION"
		}
		return "PERTURBATION_RESPONSE"

	case "BStar":
		return "ATMOSPHERIC_INTERACTION"

	default:
		return "UNKNOWN_MANEUVER"
	}
}
