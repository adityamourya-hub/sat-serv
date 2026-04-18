// Package types provides shared data structures for the SSIS engine.
// These types are independent of protobuf and can be used across all packages
// without causing import cycles.
package types

import "time"

// MeasurementType identifies the sensor source of an observation.
type MeasurementType int

const (
	MeasurementTypeUnknown MeasurementType = iota
	MeasurementTypeRadar
	MeasurementTypeOptical
	MeasurementTypeSAR
	MeasurementTypeNavIC
	MeasurementTypeTLE
)

// RadarMeasurement represents a radar sensor reading.
type RadarMeasurement struct {
	TargetID      string
	RangeKm       float64
	RangeRateKms  float64
	AzimuthDeg    float64
	ElevationDeg  float64
	RcsDbm        float64
	SnrDb         float64
	CovarianceDiag []float64 // [range, range_rate, azimuth, elevation]
}

// OpticalMeasurement represents an optical/telescope observation.
type OpticalMeasurement struct {
	TargetID   string
	RaDeg      float64
	DecDeg     float64
	Magnitude  float64
	CovarianceDiag []float64 // [ra, dec]
}

// TLEMeasurement represents a Two-Line Element set measurement.
type TLEMeasurement struct {
	Line1     string
	Line2     string
	Epoch     time.Time
	Position  []float64 // [x, y, z] km
	Velocity  []float64 // [vx, vy, vz] km/s
}

// Observation represents a raw sensor packet ingested into the system.
type Observation struct {
	ObservationID string
	SensorID      string
	Timestamp     time.Time
	Type          MeasurementType
	TargetID      string // Resolved target (empty for new detections)

	// Oneof measurement fields
	Radar   *RadarMeasurement
	Optical *OpticalMeasurement
	TLE     *TLEMeasurement

	// Signature for integrity verification (Ed25519)
	Signature []byte
}

// Reset clears the observation for reuse from a sync pool.
func (o *Observation) Reset() {
	o.ObservationID = ""
	o.SensorID = ""
	o.Timestamp = time.Time{}
	o.Type = MeasurementTypeUnknown
	o.TargetID = ""
	o.Radar = nil
	o.Optical = nil
	o.TLE = nil
	o.Signature = nil
}

// GetMeasurementVector returns the measurement as a state vector if applicable.
func (o *Observation) GetMeasurementVector() ([]float64, bool) {
	switch o.Type {
	case MeasurementTypeTLE:
		if o.TLE != nil {
			return append(o.TLE.Position, o.TLE.Velocity...), true
		}
	}
	return nil, false
}
