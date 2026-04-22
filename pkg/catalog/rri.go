package catalog

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

/**
 * Reachability Risk Index (RRI)
 *
 * Polynomial approximation of reachability sets for fast threat assessment.
 * Stores Taylor series coefficients for O(1) reachability queries.
 *
 * Theory: Instead of propagating full reachable sets, approximate them
 * as polynomials in initial state space. RRI evaluates whether a target
 * is "optimizing its EFE" toward strategic assets.
 *
 * Storage: 104 bytes per target (fits in L2 cache line)
 */

// TaylorDegree is the polynomial degree for reachability approximation
const TaylorDegree = 3 // Cubic approximation

// NumTaylorCoeffs is the number of coefficients for 6D state, degree 3
// (6+3 choose 3) = 84, but we store only dominant modes = 24
const NumTaylorCoeffs = 24

// ReachabilityIndex stores polynomial coefficients for reachability envelope
type ReachabilityIndex struct {
	// Target identification
	TargetID uint32
	_        uint32 // Padding

	// Reference state (center of Taylor expansion)
	CenterState [6]float64 // [q, p] canonical coordinates

	// Taylor coefficients for reachability polynomial
	// R(x) ≈ Σ c_i * (x - x0)^i  where i is multi-index
	TaylorCoeff [NumTaylorCoeffs]float64

	// Pre-computed risk metrics
	RiskScore      float32 // Normalized 0-1 risk
	ConflictMetric float32 // EFE-based conflict potential
	TimeHorizon    float32 // Seconds of valid prediction
	_              uint32  // Padding

	// Metadata
	LastUpdate int64  // Unix nanoseconds
	Flags      uint32 // Status flags
}

// Verify struct size
const ReachabilityIndexSize = 8 + 48 + 192 + 16 + 16 // 280 bytes

// RRIFlags
type RRIFlags uint32

const (
	RRIFlagActive      RRIFlags = 1 << 0
	RRIFlagManeuvering RRIFlags = 1 << 1
	RRIFlagHGV         RRIFlags = 1 << 2 // Hypersonic Glide Vehicle
	RRIFlagDebris      RRIFlags = 1 << 3
)

// RRIStore manages reachability indices for threat assessment
type RRIStore struct {
	indices map[uint32]*ReachabilityIndex
	mu      sync.RWMutex
}

// NewRRIStore creates a new RRI storage
func NewRRIStore() *RRIStore {
	return &RRIStore{
		indices: make(map[uint32]*ReachabilityIndex, 1000),
	}
}

// Get retrieves an RRI by target ID
func (s *RRIStore) Get(targetID uint32) (*ReachabilityIndex, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	idx, ok := s.indices[targetID]
	return idx, ok
}

// Put updates or inserts an RRI
func (s *RRIStore) Put(idx *ReachabilityIndex) {
	s.mu.Lock()
	defer s.mu.Unlock()
	idx.LastUpdate = time.Now().UnixNano()
	s.indices[idx.TargetID] = idx
}

// Delete removes an RRI
func (s *RRIStore) Delete(targetID uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.indices, targetID)
}

// EvaluateReachability computes if target can reach query point within time horizon
func (idx *ReachabilityIndex) EvaluateReachability(queryState [6]float64, timeHorizon float64) float64 {
	// Check if prediction is still valid
	age := float64(time.Now().UnixNano()-idx.LastUpdate) / 1e9
	if age > float64(idx.TimeHorizon) {
		return -1 // Stale data
	}

	// Normalize to center
	dx := [6]float64{}
	for i := 0; i < 6; i++ {
		dx[i] = queryState[i] - idx.CenterState[i]
	}

	// Evaluate Taylor polynomial
	reachability := idx.taylorEvaluate(dx)

	// Time-based decay (reachability decreases with time)
	timeFactor := math.Exp(-age / float64(idx.TimeHorizon))

	return reachability * timeFactor
}

// EvaluateRisk computes collision risk between two reachability indices
func (idx *ReachabilityIndex) EvaluateRisk(other *ReachabilityIndex) float64 {
	// Compute distance between centers
	distSq := 0.0
	for i := 0; i < 6; i++ {
		d := idx.CenterState[i] - other.CenterState[i]
		distSq += d * d
	}
	dist := math.Sqrt(distSq)

	// Get reachability radii
	r1 := idx.getReachabilityRadius()
	r2 := other.getReachabilityRadius()

	// Risk is high if reachability envelopes overlap
	if dist < r1+r2 {
		overlap := (r1 + r2 - dist) / (r1 + r2)
		return float64(idx.RiskScore) * float64(other.RiskScore) * overlap
	}

	return 0.0
}

// CheckIntentChange detects if SINDy residual indicates maneuver
func (idx *ReachabilityIndex) CheckIntentChange(sindyResidual float64, threshold float64) bool {
	// If residual exceeds threshold, object is deviating from predicted trajectory
	if sindyResidual > threshold {
		// Compare new trajectory against RRI polynomial
		// If optimizing toward strategic asset, escalate
		return idx.isOptimizingTowardTarget()
	}
	return false
}

// Private methods

func (idx *ReachabilityIndex) taylorEvaluate(dx [6]float64) float64 {
	// Simplified: use first few coefficients for speed
	// Full polynomial would evaluate all NumTaylorCoeffs terms

	result := 0.0

	// Zeroth order (constant)
	result += idx.TaylorCoeff[0]

	// First order (linear)
	for i := 0; i < 6; i++ {
		result += idx.TaylorCoeff[1+i] * dx[i]
	}

	// Second order (quadratic) - simplified diagonal terms only
	for i := 0; i < 6; i++ {
		result += idx.TaylorCoeff[7+i] * dx[i] * dx[i]
	}

	return result
}

func (idx *ReachabilityIndex) getReachabilityRadius() float64 {
	// Approximate reachability radius from Taylor coefficients
	// This is a simplified metric for fast comparison
	radius := 0.0
	for i := 0; i < 6; i++ {
		radius += idx.TaylorCoeff[1+i] * idx.TaylorCoeff[1+i]
	}
	return math.Sqrt(radius)
}

func (idx *ReachabilityIndex) isOptimizingTowardTarget() bool {
	// Simplified intent detection
	// In production: would check if trajectory reduces EFE to strategic assets
	return idx.Flags&uint32(RRIFlagManeuvering) != 0 && idx.RiskScore > 0.7
}

// MarshalBinary serializes RRI to binary (for WAL/storage)
func (idx *ReachabilityIndex) MarshalBinary() ([]byte, error) {
	buf := make([]byte, ReachabilityIndexSize)

	// Use little-endian encoding
	binary.LittleEndian.PutUint32(buf[0:4], idx.TargetID)
	offset := 8

	// CenterState (6 float64)
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint64(buf[offset:offset+8],
			math.Float64bits(idx.CenterState[i]))
		offset += 8
	}

	// TaylorCoeff (24 float64)
	for i := 0; i < NumTaylorCoeffs; i++ {
		binary.LittleEndian.PutUint64(buf[offset:offset+8],
			math.Float64bits(idx.TaylorCoeff[i]))
		offset += 8
	}

	// Risk metrics (3 float32)
	binary.LittleEndian.PutUint32(buf[offset:offset+4],
		math.Float32bits(idx.RiskScore))
	binary.LittleEndian.PutUint32(buf[offset+4:offset+8],
		math.Float32bits(idx.ConflictMetric))
	binary.LittleEndian.PutUint32(buf[offset+8:offset+12],
		math.Float32bits(idx.TimeHorizon))
	offset += 16

	// Metadata
	binary.LittleEndian.PutUint64(buf[offset:offset+8], uint64(idx.LastUpdate))
	binary.LittleEndian.PutUint32(buf[offset+8:offset+12], idx.Flags)

	return buf, nil
}

// UnmarshalBinary deserializes RRI from binary
func (idx *ReachabilityIndex) UnmarshalBinary(data []byte) error {
	if len(data) < ReachabilityIndexSize {
		return fmt.Errorf("insufficient data for RRI")
	}

	idx.TargetID = binary.LittleEndian.Uint32(data[0:4])
	offset := 8

	// CenterState
	for i := 0; i < 6; i++ {
		idx.CenterState[i] = math.Float64frombits(
			binary.LittleEndian.Uint64(data[offset : offset+8]))
		offset += 8
	}

	// TaylorCoeff
	for i := 0; i < NumTaylorCoeffs; i++ {
		idx.TaylorCoeff[i] = math.Float64frombits(
			binary.LittleEndian.Uint64(data[offset : offset+8]))
		offset += 8
	}

	// Risk metrics
	idx.RiskScore = math.Float32frombits(
		binary.LittleEndian.Uint32(data[offset : offset+4]))
	idx.ConflictMetric = math.Float32frombits(
		binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
	idx.TimeHorizon = math.Float32frombits(
		binary.LittleEndian.Uint32(data[offset+8 : offset+12]))
	offset += 16

	// Metadata
	idx.LastUpdate = int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
	idx.Flags = binary.LittleEndian.Uint32(data[offset+8 : offset+12])

	return nil
}

// RRIComputeTaylor fits Taylor polynomial to trajectory samples
func RRIComputeTaylor(samples [][6]float64, center [6]float64) [NumTaylorCoeffs]float64 {
	// Simplified: compute coefficients via finite differences
	// In production: would use proper polynomial regression

	var coeffs [NumTaylorCoeffs]float64

	// Zeroth order (mean)
	for _, s := range samples {
		for i := 0; i < 6; i++ {
			coeffs[0] += s[i]
		}
	}
	coeffs[0] /= float64(len(samples) * 6)

	// First order (linear trend)
	if len(samples) > 1 {
		for d := 0; d < 6; d++ {
			delta := samples[len(samples)-1][d] - samples[0][d]
			coeffs[1+d] = delta / float64(len(samples))
		}
	}

	// Higher orders would require more samples
	return coeffs
}
