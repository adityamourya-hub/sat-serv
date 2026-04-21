package db

import "time"

// Stub file to satisfy package import requirements
// The S2 conjunction functionality has been temporarily disabled due to API compatibility

// OrbitalObject represents a catalog object (stub for compatibility)
type OrbitalObject struct {
	ID         string
	NoradID    int
	S2CellID   uint64
	Position   Point
	Lat        float64
	Lng        float64
	AltitudeKm float64
	Timestamp  time.Time
	Metadata   []byte
}

// Point represents a geographic point (stub for compatibility)
type Point struct {
	Lat, Lng float64
}
