package catalog

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"ssis/pkg/octree"
	"ssis/pkg/spatial"
)

/**
 * ObjectRecord defines the persistent state and analytical metadata of a tracked space object.
 */
type ObjectRecord struct {
	ID             int
	Name           string
	CountryCode    string
	AssetType      string
	Pos            octree.Vector3
	Vel            octree.Vector3
	PosSigmaKm     float64
	Covariance     [6][6]float64
	LastUpdateUnix int64
	Version        int
	Timestamp      float64
	TLELine1       string
	TLELine2       string
	IsSGP4         bool
	Cd             float64
	AMR            float64
}

func (r ObjectRecord) GetPosition() octree.Vector3 { return r.Pos }
func (r ObjectRecord) GetID() int                  { return r.ID }
func (r ObjectRecord) GetTimestamp() float64       { return r.Timestamp }

type TrackObservation struct {
	TrackID       int
	Pos           octree.Vector3
	SigmaKm       float64
	TimestampUnix int64
}

type TrackAssociationCandidate struct {
	Record           ObjectRecord
	DistanceKm       float64
	AssociationScore float64
	Confidence       float64
}

// FusedTrackObservation represents a fused multi-sensor track observation.
type FusedTrackObservation struct {
	TrackID          int
	Pos              octree.Vector3
	Vel              octree.Vector3
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
	records map[int]ObjectRecord
	grid    *spatial.HashGrid
	index   *BVHIndex // Production Index: Bounding Volume Hierarchy

	PersistFunc func(Delta)
}

type Delta struct {
	Upserts []ObjectRecord
	Deletes []int
}

/**
 * NewStore initializes the catalog with the production BVH substrate.
 */
func NewStore(cellSize float64) *Store {
	return &Store{
		records: make(map[int]ObjectRecord),
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
	fmt.Printf("DEBUG: Applying Delta to Store (Size=%d) at %v\n", len(delta.Upserts), time.Now())

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
	allRecords := make([]ObjectRecord, 0, len(s.records))
	for _, rec := range s.records {
		allRecords = append(allRecords, rec)
	}
	s.index = BuildBVH(allRecords)

	if s.PersistFunc != nil {
		s.PersistFunc(delta)
	}

	fmt.Printf("DEBUG: Catalog updated to v%d\n", s.version)
	// Exit Point: Update Complete
	return s.version
}

/**
 * QueryRange performs a localized volume scan using the BVH tree.
 */
func (s *Store) QueryRange(min, max octree.Vector3, tMin, tMax float64) []ObjectRecord {
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

func sqDist(a, b octree.Vector3) float64 {
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
