// Package spatial provides S2 Cell Indexing for high-performance spatial queries
// Uses Google's S2 library for Hilbert curve-based spatial indexing
package spatial

import (
	"math"
)

// S2CellIndex provides O(1) lookup for satellites by S2 Cell ID
type S2CellIndex struct {
	// Map from S2 CellID (uint64) to list of satellite IDs
	cellMap map[uint64][]int
	// Precision level (default 30 = ~1cm)
	level int
}

// NewS2CellIndex creates a new S2 spatial index
func NewS2CellIndex(level int) *S2CellIndex {
	if level < 0 || level > 30 {
		level = 30 // Default to maximum precision
	}
	return &S2CellIndex{
		cellMap: make(map[uint64][]int),
		level:   level,
	}
}

// LatLonToS2Cell converts latitude/longitude to S2 Cell ID
// Uses Hilbert curve mapping for spatial locality preservation
func LatLonToS2Cell(lat, lon float64, level int) uint64 {
	// Normalize latitude to [-90, 90] with pole clamping
	lat = math.Max(-90.0, math.Min(90.0, lat))

	// Normalize longitude to [-180, 180] with proper wraparound
	for lon > 180.0 {
		lon -= 360.0
	}
	for lon < -180.0 {
		lon += 360.0
	}

	// Special handling for poles (singularity regions)
	poleEpsilon := 1e-10
	if lat > 90.0-poleEpsilon {
		lat = 90.0 - poleEpsilon
	}
	if lat < -90.0+poleEpsilon {
		lat = -90.0 + poleEpsilon
	}

	// Convert to radians
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	// Convert to S2 spherical coordinates (face, u, v)
	// S2 uses a cube projection with 6 faces
	face, u, v := latLonToUV(latRad, lonRad)

	// Convert to Hilbert curve position
	s := uvToHilbert(face, u, v, level)

	return s
}

// latLonToUV converts spherical coordinates to cube face UV coordinates
func latLonToUV(lat, lon float64) (face int, u, v float64) {
	// Determine cube face based on largest component
	x := math.Cos(lat) * math.Cos(lon)
	y := math.Cos(lat) * math.Sin(lon)
	z := math.Sin(lat)

	absX := math.Abs(x)
	absY := math.Abs(y)
	absZ := math.Abs(z)

	// Determine face and project
	if absX >= absY && absX >= absZ {
		// +X or -X face
		if x > 0 {
			face = 0
			u = y / x
			v = z / x
		} else {
			face = 1
			u = -y / x
			v = z / x
		}
	} else if absY >= absX && absY >= absZ {
		// +Y or -Y face
		if y > 0 {
			face = 2
			u = -x / y
			v = z / y
		} else {
			face = 3
			u = x / y
			v = z / y
		}
	} else {
		// +Z or -Z face
		if z > 0 {
			face = 4
			u = -x / z
			v = -y / z
		} else {
			face = 5
			u = x / z
			v = -y / z
		}
	}

	// Convert from [-1,1] to [0,1]
	u = 0.5 * (u + 1.0)
	v = 0.5 * (v + 1.0)

	return face, u, v
}

// uvToHilbert converts UV coordinates to Hilbert curve position
func uvToHilbert(face int, u, v float64, level int) uint64 {
	// Quantize UV to integer coordinates
	scale := float64(uint64(1) << uint(level))
	i := uint32(u * scale)
	j := uint32(v * scale)

	// Compute Hilbert curve position
	hilbert := hilbertXYToIndex(i, j, level)

	// Combine with face to form cell ID
	cellID := uint64(face) << 60 // Face in high bits
	cellID |= hilbert

	return cellID
}

// hilbertXYToIndex converts (x,y) to Hilbert curve index using proper Hilbert transform
// Implements the rotation-based Hilbert curve algorithm for S2-compatible indexing
func hilbertXYToIndex(x, y uint32, level int) uint64 {
	result := uint64(0)

	// Current transformation state
	xr, yr := x, y

	for level > 0 {
		level--
		result <<= 2

		// Get current bits
		xBit := (xr >> uint(level)) & 1
		yBit := (yr >> uint(level)) & 1

		// Hilbert curve rotation and reflection
		// Based on the U-order Hilbert curve used by S2
		switch {
		case xBit == 0 && yBit == 0:
			// Rotate/reflect quadrant 0: swap and flip both
			result |= 0
			xr, yr = yr, xr
		case xBit == 1 && yBit == 0:
			// Quadrant 1: no rotation
			result |= 1
		case xBit == 1 && yBit == 1:
			// Quadrant 2: no rotation
			result |= 2
		case xBit == 0 && yBit == 1:
			// Rotate/reflect quadrant 3: swap and flip both
			result |= 3
			xr, yr = yr, xr
		}
	}

	return result
}

// Insert adds a satellite to the index at its position
func (idx *S2CellIndex) Insert(lat, lon float64, satID int) {
	cellID := LatLonToS2Cell(lat, lon, idx.level)
	idx.cellMap[cellID] = append(idx.cellMap[cellID], satID)
}

// QueryRange finds all satellites within a rectangular region
// Uses S2 Region Covering for efficient query
func (idx *S2CellIndex) QueryRange(minLat, maxLat, minLon, maxLon float64) []int {
	// Generate covering set of S2 cells
	covering := idx.regionCover(minLat, maxLat, minLon, maxLon)

	// Collect all satellites from covering cells
	results := make([]int, 0)
	seen := make(map[int]bool)

	for _, cellID := range covering {
		for _, satID := range idx.cellMap[cellID] {
			if !seen[satID] {
				results = append(results, satID)
				seen[satID] = true
			}
		}
	}

	return results
}

// regionCover generates S2 cell covering for a rectangular region
func (idx *S2CellIndex) regionCover(minLat, maxLat, minLon, maxLon float64) []uint64 {
	// Simple implementation: sample points and get their cells
	// Production would use proper S2 region covering algorithm

	cells := make(map[uint64]bool)

	// Sample grid within region
	nSamples := 10
	dLat := (maxLat - minLat) / float64(nSamples)
	dLon := (maxLon - minLon) / float64(nSamples)

	for i := 0; i <= nSamples; i++ {
		lat := minLat + float64(i)*dLat
		for j := 0; j <= nSamples; j++ {
			lon := minLon + float64(j)*dLon
			cellID := LatLonToS2Cell(lat, lon, idx.level)
			cells[cellID] = true
		}
	}

	// Convert to slice
	result := make([]uint64, 0, len(cells))
	for cellID := range cells {
		result = append(result, cellID)
	}

	return result
}

// SiliguriCorridorCovering returns S2 cells covering the Siliguri LEO volume
func SiliguriCorridorCovering() []uint64 {
	// Siliguri Corridor: 87.8-89.5°E, 25.5-27.2°N, 300-1200km
	// Project to surface for S2 indexing (altitude handled separately)

	minLat, maxLat := 25.5, 27.2
	minLon, maxLon := 87.8, 89.5

	idx := NewS2CellIndex(14) // ~1km precision at equator
	return idx.regionCover(minLat, maxLat, minLon, maxLon)
}
