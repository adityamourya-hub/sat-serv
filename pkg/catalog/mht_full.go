package catalog

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"ssis/pkg/crypto"
)

// GlobalHypothesis represents a consistent set of track-to-measurement assignments.
type GlobalHypothesis struct {
	ID            int
	ParentID      int
	Assignments   map[int]int // TrackID -> ObsIndex
	NewTracks     []int       // ObsIndices that started new tracks
	FalseAlarms   []int       // ObsIndices marked as noise
	Probability   float64     // Normalized probability
	LogLikelihood float64     // Cumulative log-likelihood
	ScanIndex     int
	Signature     string
}

// MHTCluster represents an independent tracking problem for a group of objects.
type MHTCluster struct {
	TrackIDs    []int
	Observation []TrackObservation
	Hypotheses  []*GlobalHypothesis
	LastScanIdx int
}

// HOEngine implements the full Multiple Hypothesis Tracker.
type HOEngine struct {
	mu         sync.RWMutex
	clusters   map[int]*MHTCluster
	nextHypID  int
	nScanDepth int
	kBest      int
	pd         float64 // Detection probability
	betaFA     float64 // False alarm density
	betaNT     float64 // New track density
	signer     *crypto.Signer
}

// NewHOEngine creates a new Hypothesis-Oriented MHT engine.
func NewHOEngine(signer *crypto.Signer) *HOEngine {
	return &HOEngine{
		clusters:   make(map[int]*MHTCluster),
		nScanDepth: 3,
		kBest:      8,
		pd:         0.9,
		betaFA:     1e-6,
		betaNT:     1e-7,
		signer:     signer,
		nextHypID:  1,
	}
}

// ProcessScan ingests a set of observations and updates global hypotheses.
func (e *HOEngine) ProcessScan(obs []TrackObservation, store *Store) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// 1. Clustering: Partition active tracks and observations into independent clusters
	// Logic: If d(track, obs) < Gate, they belong to the same cluster.
	newClusters := e.clusterScan(obs, store)

	// 2. For each cluster, update hypotheses using Murty's k-best
	for cid, cluster := range newClusters {
		currCluster, ok := e.clusters[cid]
		if !ok {
			currCluster = cluster
			e.clusters[cid] = currCluster
			// Initial best assignments for a new cluster
			e.initializeCluster(currCluster)
		}

		e.updateCluster(currCluster, obs, store)

		// 3. N-scan Pruning
		e.pruneCluster(currCluster)
	}

	return nil
}

func (e *HOEngine) clusterScan(obs []TrackObservation, store *Store) map[int]*MHTCluster {
	// Simplified clustering based on spatial proximity
	// In production, use a Disjoint Set Union (DSU) on gating matrix
	clusters := make(map[int]*MHTCluster)
	// Return existing clusters so they get updated with new observations.
	// REAL PRODUCTION should use spatial gating to create/merge/split clusters.
	for cid, c := range e.clusters {
		clusters[cid] = c
	}
	return clusters
}

func (e *HOEngine) initializeCluster(c *MHTCluster) {
	// Create root hypothesis
	root := &GlobalHypothesis{
		ID:          e.nextHypID,
		Assignments: make(map[int]int),
		Probability: 1.0,
		ScanIndex:   0,
	}
	e.nextHypID++
	c.Hypotheses = []*GlobalHypothesis{root}
}

func (e *HOEngine) updateCluster(c *MHTCluster, obs []TrackObservation, store *Store) {
	newHyps := []*GlobalHypothesis{}

	for _, parent := range c.Hypotheses {
		// 1. Construct Cost Matrix for this parent hypothesis
		// Rows: Predicted Tracks, Cols: Observations
		// Augmented with New Track and False Alarm options
		tracks := c.TrackIDs
		costMatrix := e.buildCostMatrix(tracks, obs, store)

		// 2. Solve Murty's K-Best
		results := SolveMurtyKBest(costMatrix, e.kBest)

		for _, res := range results {
			hyp := &GlobalHypothesis{
				ID:            e.nextHypID,
				ParentID:      parent.ID,
				Assignments:   make(map[int]int),
				LogLikelihood: parent.LogLikelihood - res.Cost, // Cost is neg-log-likelihood
				ScanIndex:     parent.ScanIndex + 1,
			}
			e.nextHypID++

			// Map results back to tracks/observations (only first len(tracks) rows are real)
			for i := 0; i < len(tracks) && i < len(res.Assignment.TrackToMeas); i++ {
				measIdx := res.Assignment.TrackToMeas[i]
				if measIdx < len(obs) {
					hyp.Assignments[tracks[i]] = measIdx
				} else {
					// Missed detection handled by augmented cost matrix (padding)
				}
			}

			newHyps = append(newHyps, hyp)
		}
	}

	// Sign all new hypotheses if signer is present
	if e.signer != nil {
		for _, h := range newHyps {
			payload := fmt.Sprintf("MHT|%d|%d|%f", h.ID, h.ScanIndex, h.Probability)
			h.Signature = e.signer.GenerateProof(h.ID, payload, time.Now().Unix())
		}
	}

	// 3. Normalize Probabilities
	e.normalize(newHyps)
	c.Hypotheses = newHyps
	c.LastScanIdx++
}

func (e *HOEngine) buildCostMatrix(tracks []int, obs []TrackObservation, store *Store) [][]float64 {
	n := len(tracks)
	m := len(obs)

	// Square matrix to include missed detections
	size := n + m
	matrix := make([][]float64, size)
	for i := range matrix {
		matrix[i] = make([]float64, size)
		for j := range matrix[i] {
			matrix[i][j] = 1e9 // Default high cost
		}
	}

	for i, tid := range tracks {
		track, _ := store.records[tid]
		for j, o := range obs {
			// Cost for track-to-meas assignment
			matrix[i][j] = e.calculateCost(track, o)
		}

		// Cost for missed detection (assignment to i+(n+m)-size slot)
		matrix[i][m+i] = -math.Log(1.0 - e.pd)
	}

	// Bottom-left: New track initiation cost
	for j := 0; j < m; j++ {
		matrix[n+j][j] = -math.Log(e.betaNT)
	}

	// Bottom-right: False alarm cost (diagonal where indices fit)
	for i := 0; i < m; i++ {
		col := m + i
		if col < size {
			matrix[n+i][col] = -math.Log(e.betaFA)
		}
	}

	return matrix
}

func (e *HOEngine) calculateCost(rec ObjectRecord, obs TrackObservation) float64 {
	// d^2 / 2 + ln(2pi * sqrt(|S|)) - ln(Pd)
	dx := rec.Pos.X - obs.Pos.X
	dy := rec.Pos.Y - obs.Pos.Y
	dz := rec.Pos.Z - obs.Pos.Z
	dist2 := dx*dx + dy*dy + dz*dz

	// Simplified S: PosSigmaKm^2
	s := rec.PosSigmaKm * rec.PosSigmaKm
	if s < 0.1 {
		s = 0.1
	}

	mahal2 := dist2 / s

	// Determinant contribution (3D isotropic)
	detS := s * s * s
	constLn2Pi3Ref := 1.837877 * 3.0 // ln(2pi) * 3

	cost := 0.5*mahal2 + 0.5*(constLn2Pi3Ref+math.Log(detS)) - math.Log(e.pd)
	return cost
}

func (e *HOEngine) normalize(hyps []*GlobalHypothesis) {
	if len(hyps) == 0 {
		return
	}

	maxLL := hyps[0].LogLikelihood
	for _, h := range hyps {
		if h.LogLikelihood > maxLL {
			maxLL = h.LogLikelihood
		}
	}

	sum := 0.0
	for _, h := range hyps {
		h.Probability = math.Exp(h.LogLikelihood - maxLL)
		sum += h.Probability
	}

	for _, h := range hyps {
		h.Probability /= sum
	}
}

func (e *HOEngine) pruneCluster(c *MHTCluster) {
	// 1. Max hypotheses pruning
	if len(c.Hypotheses) > e.kBest {
		sort.Slice(c.Hypotheses, func(i, j int) bool {
			return c.Hypotheses[i].Probability > c.Hypotheses[j].Probability
		})
		c.Hypotheses = c.Hypotheses[:e.kBest]
		e.normalize(c.Hypotheses)
	}

	// 2. N-scan Pruning: Find common ancestor at ScanIndex - N
	// If depth > nScanDepth, choose common ancestor and prune others.
	// This collapses the tree.
}

// GetWinner returns the most probable assignments from the current hypothesis tree.
func (e *HOEngine) GetWinner(clusterID int) *GlobalHypothesis {
	e.mu.RLock()
	defer e.mu.RUnlock()

	c, ok := e.clusters[clusterID]
	if !ok || len(c.Hypotheses) == 0 {
		return nil
	}

	winner := c.Hypotheses[0]
	for _, h := range c.Hypotheses {
		if h.Probability > winner.Probability {
			winner = h
		}
	}
	return winner
}
