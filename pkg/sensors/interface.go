// pkg/sensors/interface.go - Sensor Agnosticism Interface
//
// Implements strictly gated modularity for sensor extensibility.
// The core reflex kernel never needs to know what hardware is feeding it;
// it only cares about the residual geometry through the SensorObserver interface.

package sensors

import (
	"ssis/pkg/physics"
)

// Observation represents a single sensor observation
// Designed for zero-copy operations and cache locality
type Observation struct {
	// Position and velocity (ECI frame)
	Position physics.Vector3 // [km]
	Velocity physics.Vector3 // [km/s]

	// Measurement metadata
	TimestampNs int64         // UnixNano for deterministic time
	Wavelength   float64      // Sensor wavelength [m] (differentiates Optical vs RF)
	DeviceID     uint32       // Integer ID (avoids string allocations)
	Quality      float64      // Measurement quality [0-1]

	// Flattened covariance (cache-friendly, SIMD-ready)
	Covariance [36]float64 // 6x6 matrix flattened
}

// SensorObserver defines the contract for all sensor types
// Any sensor (optical, radar, RF) must implement this interface
// to be compatible with the core reflex kernel
type SensorObserver interface {
	// GetObservation returns the latest observation from the sensor
	// This is called by the ingest pipeline at the sensor's native rate
	GetObservation() (Observation, error)

	// GetCovariance returns the measurement uncertainty matrix
	// Flattened [36]float64 format for cache locality
	GetCovariance() [36]float64

	// GetWavelength returns the sensor's operating wavelength
	// Used to differentiate Optical (~500nm) from RF (~0.1m)
	GetWavelength() float64

	// GetDeviceID returns the sensor's integer identifier
	// Used for provenance tracking and multi-sensor fusion
	GetDeviceID() uint32

	// GetSampleRate returns the sensor's native sample rate in Hz
	// Used to synchronize with the 100Hz reflex loop
	GetSampleRate() float64
}
