package catalog

import (
	"log/slog"
	"math"
	"sort"
	"sync"
	"time"

	"ssis/pkg/spatial"
	"ssis/pkg/types"
)

type CoordinateFrame string

const (
	FrameGCRF    CoordinateFrame = "GCRF"    // High-fidelity GCRF (IAU 2000/2006)
	FrameEME2000 CoordinateFrame = "EME2000" // Mean J2000
	FrameECEF    CoordinateFrame = "ECEF"    // Earth-Centered Fixed
)

type ObjectStatus string

const (
	StatusActive  ObjectStatus = "ACTIVE"
	StatusDecayed ObjectStatus = "DECAYED"
	StatusLost    ObjectStatus = "LOST"
)

/**
 * SDAObject defines the mission-ready persistent state and analytical metadata
 * of a tracked space object, compliant with the iDEX industry hardening.
 */
type SDAObject struct {
	ID          int          `json:"object_id"`
	Name        string       `json:"name"`
	CountryCode string       `json:"country_code"`
	AssetType   string       `json:"type"` // e.g., PAYLOAD, ROCKET_BODY, DEBRIS
	Status      ObjectStatus `json:"status"`

	// State & Uncertainty
	Pos        types.Vector3   `json:"position"` // ECI km
	Vel        types.Vector3   `json:"velocity"` // ECI km/s
	Covariance [6][6]float64   `json:"covariance"`
	EpochTT    float64         `json:"epoch_tt"` // Terrestrial Time Julian Date
	Frame      CoordinateFrame `json:"frame"`

	// Physical Characteristics (Environment Interactions)
	MassKg   float64 `json:"mass_kg"`
	DragArea float64 `json:"drag_area_m2"`
	SRPArea  float64 `json:"srp_area_m2"`
	Cd       float64 `json:"drag_coeff"`
	Cr       float64 `json:"reflectivity_coeff"`
	RCS      float64 `json:"rcs_m2"`

	// Provenance & Quality
	LastUpdateUnix int64   `json:"last_updated"`
	Version        int     `json:"version"`
	ODQualityScore float64 `json:"od_quality"`
	SensorID       string  `json:"sensor_id"`

	// Uncertainty (State Sigma)
	PosSigmaKm  float64 `json:"pos_sigma_km"`
	VelSigmaKmS float64 `json:"vel_sigma_kms"`

	// Legacy Support
	TLELine1 string `json:"tle_l1,omitempty"`
	TLELine2 string `json:"tle_l2,omitempty"`

	// Alias for BVH alignment
	Timestamp float64 `json:"-"`
}

// ObjectRecord is a legacy alias for SDAObject to support BVH indexing
// and internal structural consistency during the Phase 1.7 migration.
type ObjectRecord = SDAObject

func (r SDAObject) GetPosition() types.Vector3 { return r.Pos }
func (r SDAObject) GetID() int                 { return r.ID }
func (r SDAObject) GetTimestamp() float64      { return r.EpochTT }

type TrackObservation struct {
	TrackID       int
	Pos           types.Vector3
	SigmaKm       float64
	TimestampUnix int64
}

type TrackAssociationCandidate struct {
	Record           SDAObject
	DistanceKm       float64
	AssociationScore float64
	Confidence       float64
}

// FusedTrackObservation represents a fused multi-sensor track observation.
type FusedTrackObservation struct {
	TrackID          int
	Pos              types.Vector3
	Vel              types.Vector3
	Covariance       [][]float64
	TimestampUnix    int64
	SensorIDs        []string
	FusionConfidence float64
}

// TrackAssociationInference contains the ML model's track association prediction.
type TrackAssociationInference struct {
	MatchedObjectID int
	Confidence      float64
	AssociationType string // "NEW", "EXISTING", "MANEUVER"
	Scores          map[int]float64
	Attention       map[int]float64
}

/**
 * Store represents the high-fidelity catalog substrate.
 * Refactored in Phase 1.7 to use a Bounding Volume Hierarchy (BVH)
 * for superior spatial-temporal query performance.
 *
 * <plan_complexity>High</plan_complexity><total_lines_estimate>650</total_lines_estimate>
 */
type Store struct {
	mu      sync.RWMutex
	version int
	records map[int]SDAObject
	grid    *spatial.HashGrid
	index   *BVHIndex // Production Index: Bounding Volume Hierarchy

	PersistFunc func(Delta)
}

type Delta struct {
	Upserts []SDAObject
	Deletes []int
}

/**
 * NewStore initializes the catalog with the production BVH substrate.
 */
func NewStore(cellSize float64) *Store {
	return &Store{
		records: make(map[int]SDAObject),
		grid:    spatial.NewHashGrid(cellSize),
		index:   &BVHIndex{Root: nil},
	}
}

/**
 * ApplyDeltaWithRebuild integrates updates and triggers a BVH reconstruction.
 * Uses Centroid Median splitting to optimize for O(log N) collision pruning.
 */
func (s *Store) ApplyDeltaWithRebuild(delta Delta, fullRebuild bool) int {
	// Entry Point: Delta Application
	slog.Debug("Applying Delta to Store", "size", len(delta.Upserts), "time", time.Now())

	s.mu.Lock()
	defer s.mu.Unlock()

	s.version++
	now := time.Now().Unix()

	for _, rec := range delta.Upserts {
		rec.Version = s.version
		if rec.LastUpdateUnix == 0 {
			rec.LastUpdateUnix = now
		}
		s.records[rec.ID] = rec
		s.grid.Upsert(spatial.Point{ID: rec.ID, Pos: rec.Pos})
	}

	for _, id := range delta.Deletes {
		s.grid.Delete(id)
		delete(s.records, id)
	}

	// Rebuild BVH Index (Current requirement: Broad-phase rebuild)
	allRecords := make([]SDAObject, 0, len(s.records))
	for _, rec := range s.records {
		allRecords = append(allRecords, rec)
	}
	s.index = BuildBVH(allRecords)

	if s.PersistFunc != nil {
		s.PersistFunc(delta)
	}

	slog.Debug("Catalog updated", "version", s.version)
	// Exit Point: Update Complete
	return s.version
}

/**
 * QueryRange performs a localized volume scan using the BVH tree.
 */
func (s *Store) QueryRange(min, max types.Vector3, tMin, tMax float64) []SDAObject {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.index.QueryRange(min, max, tMin, tMax)
}

/**
 * Correlate finds candidates for track association using HashGrid broad-phase.
 */
func (s *Store) Correlate(obs TrackObservation, gateKm float64, maxCandidates int) []TrackAssociationCandidate {
	if gateKm <= 0 {
		gateKm = 50
	}
	if maxCandidates <= 0 {
		maxCandidates = 5
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	ids := s.grid.QueryRadius(obs.Pos, gateKm)
	results := make([]TrackAssociationCandidate, 0, len(ids))

	for _, id := range ids {
		rec, ok := s.records[id]
		if !ok {
			continue
		}

		dist := math.Sqrt(sqDist(rec.Pos, obs.Pos))
		if dist > gateKm {
			continue
		}

		score := math.Exp(-0.5 * math.Pow(dist/math.Max(obs.SigmaKm, 0.5), 2.0))
		results = append(results, TrackAssociationCandidate{
			Record:           rec,
			DistanceKm:       dist,
			AssociationScore: score,
			Confidence:       score,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].AssociationScore > results[j].AssociationScore
	})

	if len(results) > maxCandidates {
		results = results[:maxCandidates]
	}
	return results
}

func sqDist(a, b types.Vector3) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	dz := a.Z - b.Z
	return dx*dx + dy*dy + dz*dz
}

func (s *Store) Snapshot() []ObjectRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]ObjectRecord, 0, len(s.records))
	for _, rec := range s.records {
		out = append(out, rec)
	}
	return out
}

func (s *Store) GetRecord(id int) (ObjectRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	rec, ok := s.records[id]
	return rec, ok
}

// Version returns the current catalog version number.
func (s *Store) Version() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.version
}
