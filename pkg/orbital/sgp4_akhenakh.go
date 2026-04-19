// Package orbital provides alternative SGP4 implementation using akhenakh/sgp4
// For OMM/JSON format support and modern API compatibility
package orbital

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/physics"
	// TODO: Re-enable when dependency available
	// "github.com/akhenakh/sgp4"
)

// AkhenakhSGP4Propagator wraps the akhenakh/sgp4 library
// This provides an alternative to the go-satellite implementation
// with support for OMM (Orbit Mean-elements Message) JSON format
type AkhenakhSGP4Propagator struct {
	// tle     *sgp4.TLE
	epoch   time.Time
	epochJD float64
}

// NewAkhenakhSGP4PropagatorFromLines creates propagator from TLE lines
func NewAkhenakhSGP4PropagatorFromLines(line1, line2 string) (*AkhenakhSGP4Propagator, error) {
	// tle, err := sgp4.ParseTLE(line1, line2, "")
	// if err != nil {
	// 	return nil, fmt.Errorf("parse TLE with akhenakh/sgp4: %w", err)
	// }

	// Extract epoch from TLE
	// Note: proper TLE epoch parsing would go here
	epoch := time.Now() // Placeholder - proper parsing needed

	epochJD := float64(epoch.Unix())/86400.0 + 2440587.5

	return &AkhenakhSGP4Propagator{
		// tle:     tle,
		epoch:   epoch,
		epochJD: epochJD,
	}, nil
}

// NewAkhenakhSGP4PropagatorFromOMM creates propagator from OMM JSON
// OMM is the modern standard replacing TLE for data exchange
func NewAkhenakhSGP4PropagatorFromOMM(ommData []byte) (*AkhenakhSGP4Propagator, error) {
	// OMM (Orbit Mean-elements Message) JSON format
	// Example OMM fields: EPOCH, MEAN_MOTION, ECCENTRICITY, etc.

	// Note: akhenakh/sgp4 supports OMM parsing - implementation would
	// use sgp4.ParseOMM or similar function

	return nil, fmt.Errorf("OMM parsing not yet implemented - placeholder for future OMM support")
}

// Propagate computes satellite state at given time
func (p *AkhenakhSGP4Propagator) Propagate(targetTime time.Time) (*physics.State, error) {
	// Calculate minutes since epoch
	deltaMin := targetTime.Sub(p.epoch).Minutes()

	// Validate propagation limit (SGP4 accuracy degrades beyond ~10 days)
	if math.Abs(deltaMin) > 10*24*60 {
		return nil, fmt.Errorf("propagation exceeds 10-day SGP4 accuracy limit: %.1f minutes", deltaMin)
	}

	// Use akhenakh/sgp4 to find position
	// TODO: Re-enable when dependency available
	// pos, err := p.tle.FindPosition(targetTime)
	// if err != nil {
	// 	return nil, fmt.Errorf("sgp4 propagation failed: %w", err)
	// }

	// Convert to geodetic (lat/lon/alt)
	// lat, lon, alt := pos.ToGeodetic()
	// _ = lat
	// _ = lon
	// _ = alt

	// akhenakh/sgp4 returns ECI position
	// Note: Velocity requires separate calculation or extraction from library
	// TODO: Return actual propagated position when dependency available
	state := &physics.State{
		Pos: physics.Vector3{0, 0, 0}, // Placeholder
		// Velocity would need to be calculated from derivative or library-specific method
		Vel: physics.Vector3{0, 0, 0}, // Placeholder
	}

	return state, nil
}

// GetTLEEpoch returns the TLE epoch time
func (p *AkhenakhSGP4Propagator) GetTLEEpoch() time.Time {
	return p.epoch
}

// CompareWithGoSatellite validates against existing implementation
// This function performs side-by-side comparison for verification
func CompareWithGoSatellite(line1, line2 string, targetTime time.Time) (*ComparisonResult, error) {
	// Create both propagators
	akhenakhProp, err := NewAkhenakhSGP4PropagatorFromLines(line1, line2)
	if err != nil {
		return nil, fmt.Errorf("akhenakh propagator failed: %w", err)
	}

	// Use existing go-satellite propagator
	goSatTLE, err := physics.ParseTLE(line1 + "\n" + line2)
	if err != nil {
		return nil, fmt.Errorf("TLE parsing failed: %w", err)
	}
	goSatProp := NewSGP4Propagator(goSatTLE)

	// Propagate with both
	akhenakhState, err := akhenakhProp.Propagate(targetTime)
	if err != nil {
		return nil, fmt.Errorf("akhenakh propagation failed: %w", err)
	}

	deltaMin := targetTime.Sub(akhenakhProp.epoch).Minutes()
	goSatState, err := goSatProp.Propagate(deltaMin)
	if err != nil {
		return nil, fmt.Errorf("go-satellite propagation failed: %w", err)
	}

	// Calculate position difference
	dx := akhenakhState.Pos[0] - goSatState.Pos[0]
	dy := akhenakhState.Pos[1] - goSatState.Pos[1]
	dz := akhenakhState.Pos[2] - goSatState.Pos[2]
	positionDiff := math.Sqrt(dx*dx + dy*dy + dz*dz)

	return &ComparisonResult{
		AkhenakhPosition:     akhenakhState.Pos,
		GoSatellitePosition:  goSatState.Pos,
		PositionDifferenceKm: positionDiff,
		Within1mm:            positionDiff < 0.000001, // 1mm = 0.000001 km
	}, nil
}

// ComparisonResult holds validation comparison data
type ComparisonResult struct {
	AkhenakhPosition     physics.Vector3
	GoSatellitePosition  physics.Vector3
	PositionDifferenceKm float64
	Within1mm            bool
}

// Usage Example:
//
// // For legacy TLE sources (existing pipelines):
// legacyProp := NewSGP4Propagator(tle) // go-satellite
//
// // For modern OMM/JSON sources (new data exchange):
// modernProp, err := NewAkhenakhSGP4PropagatorFromOMM(ommJSON)
// if err != nil {
//     // Fall back to legacy
//     modernProp = NewSGP4Propagator(tle)
// }
//
// // Validation during transition:
// result, err := CompareWithGoSatellite(line1, line2, targetTime)
// if result.Within1mm {
//     log.Println("✓ Propagators agree within 1mm - safe to migrate")
// }
