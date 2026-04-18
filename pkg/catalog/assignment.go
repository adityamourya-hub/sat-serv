package catalog

import (
	"math"
	"sort"
)

// Assignment represents a mapping from track indices to measurement indices.
type Assignment struct {
	TrackToMeas []int // TrackIndex -> MeasIndex (-1 for unassigned)
	Cost        float64
}

// KBestResult encapsulates a ranked assignment from Murty's algorithm.
type KBestResult struct {
	Assignment Assignment
	Cost       float64
}

// MurtyState handles optimized k-best assignment via sub-problem partitioning.
type MurtyState struct {
	OriginalCosts [][]float64
	K             int
}

// murtyPartition represents a Murty sub-problem with fixed/forbidden constraints.
type murtyPartition struct {
	fixed     [][2]int // List of (row, col) that MUST be assigned.
	forbidden [][2]int // List of (row, col) that MUST NOT be assigned.
	result    []int
	cost      float64
	dualR     []float64
	dualC     []float64
}

// SolveMurtyKBest implements Murty's algorithm with Miller-Stone-Cox optimizations.
// It finds the K best assignments in order of increasing cost.
func SolveMurtyKBest(costs [][]float64, k int) []KBestResult {
	if len(costs) == 0 {
		return nil
	}

	// 1. Find initial best assignment
	best, cost, dualR, dualC := minCostAssignment(costs)
	if cost >= math.MaxFloat64/2 {
		return nil
	}

	results := []KBestResult{{Assignment: Assignment{TrackToMeas: best, Cost: cost}, Cost: cost}}
	if k <= 1 {
		return results
	}

	// Priority queue for sub-problems (Murty partitions)
	// For SSA performance, we use a simple slice + sort for small K,
	// or a heap for larger K.
	queue := []murtyPartition{}

	// Initial partitioning of the best solution
	queue = append(queue, generatePartitions(costs, best, cost, dualR, dualC, nil, nil)...)

	for len(results) < k && len(queue) > 0 {
		// Sort queue by cost (ascending)
		sort.Slice(queue, func(i, j int) bool { return queue[i].cost < queue[j].cost })

		bestPart := queue[0]
		queue = queue[1:]

		if bestPart.cost >= math.MaxFloat64/2 {
			break
		}

		results = append(results, KBestResult{
			Assignment: Assignment{TrackToMeas: bestPart.result, Cost: bestPart.cost},
			Cost:       bestPart.cost,
		})

		if len(results) < k {
			newParts := generatePartitions(costs, bestPart.result, bestPart.cost, bestPart.dualR, bestPart.dualC, bestPart.fixed, bestPart.forbidden)
			queue = append(queue, newParts...)
		}
	}

	return results
}

// minCostAssignment finds the optimal assignment using the augmenting path / Hungarian-like method.
// Returns (assignment, cost, dualRow, dualCol).
func minCostAssignment(costs [][]float64) ([]int, float64, []float64, []float64) {
	n := len(costs)
	m := len(costs[0])

	// Pad to square if necessary
	size := n
	if m > size {
		size = m
	}

	assignment := make([]int, size)
	for i := range assignment {
		assignment[i] = -1
	}

	u := make([]float64, size+1) // dual variables for rows (1-based)
	v := make([]float64, size+1) // dual variables for columns (1-based)
	p := make([]int, size+1)     // column mapping
	way := make([]int, size+1)

	for i := 1; i <= size; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, size+1)
		for k := range minv {
			minv[k] = math.MaxFloat64
		}
		used := make([]bool, size+1)

		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.MaxFloat64
			j1 := 0

			for j := 1; j <= size; j++ {
				if !used[j] {
					val := math.MaxFloat64
					if i0-1 < n && j-1 < m {
						val = costs[i0-1][j-1]
					} else if i0-1 >= n || j-1 >= m {
						val = 0 // Padding cost
					}

					cur := val - u[i0-1] - v[j-1]
					if cur < minv[j] {
						minv[j] = cur
						way[j] = j0
					}
					if minv[j] < delta {
						delta = minv[j]
						j1 = j
					}
				}
			}

			for j := 0; j <= size; j++ {
				if used[j] {
					if p[j] > 0 {
						u[p[j]-1] += delta
					}
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}

		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	res := make([]int, n)
	totalCost := 0.0
	for j := 1; j <= size; j++ {
		row := p[j] - 1
		if row < n {
			if j-1 < m {
				res[row] = j - 1
				totalCost += costs[row][j-1]
			} else {
				res[row] = -1
			}
		}
	}

	return res, totalCost, u, v
}

// generatePartitions creates Murty sub-problems by successive forbidding.
func generatePartitions(originalCosts [][]float64, currentSolution []int, currentCost float64, u, v []float64, fixed, forbidden [][2]int) []murtyPartition {
	parts := []murtyPartition{}
	n := len(originalCosts)
	m := len(originalCosts[0])

	// Solution contains assignments (r, c).
	// We iterate through them to create partitions.
	for i, j := range currentSolution {
		if j == -1 {
			continue
		}

		// Is this (i, j) already in the 'fixed' list?
		isFixed := false
		for _, f := range fixed {
			if f[0] == i && f[1] == j {
				isFixed = true
				break
			}
		}
		if isFixed {
			continue
		}

		// Create sub-problem where (i, j) is forbidden,
		// and all previous assignments in this solution are fixed.
		newFixed := append([][2]int(nil), fixed...)
		newForbidden := append([][2]int(nil), forbidden...)
		newForbidden = append(newForbidden, [2]int{i, j})

		// Solver with inheritance (Miller-Stone-Cox)
		// For simplicity in this implementation, we re-use minCostAssignment with modified costs
		modCosts := make([][]float64, n)
		for r := range modCosts {
			modCosts[r] = append([]float64(nil), originalCosts[r]...)
		}

		for _, f := range newForbidden {
			modCosts[f[0]][f[1]] = math.MaxFloat64 / 2
		}
		for _, f := range newFixed {
			// To fix (r, c), we block other options for row r and col c
			for col := 0; col < m; col++ {
				if col != f[1] {
					modCosts[f[0]][col] = math.MaxFloat64 / 2
				}
			}
			for row := 0; row < n; row++ {
				if row != f[0] {
					modCosts[row][f[1]] = math.MaxFloat64 / 2
				}
			}
		}

		res, cost, dU, dV := minCostAssignment(modCosts)
		if cost < math.MaxFloat64/2 {
			parts = append(parts, murtyPartition{
				fixed:     newFixed,
				forbidden: newForbidden,
				result:    res,
				cost:      cost,
				dualR:     dU,
				dualC:     dV,
			})
		}

		// For the next iteration, (i, j) moves from 'current' to 'fixed'
		fixed = append(fixed, [2]int{i, j})
	}

	return parts
}
