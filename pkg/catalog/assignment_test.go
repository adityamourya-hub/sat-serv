package catalog

import (
	"math"
	"testing"
)

func TestMinCostAssignment(t *testing.T) {
	costs := [][]float64{
		{10, 20, 30},
		{30, 10, 20},
		{20, 30, 10},
	}
	
	assign, cost, _, _ := minCostAssignment(costs)
	
	if math.Abs(cost-30.0) > 1e-9 {
		t.Errorf("Expected cost 30, got %f", cost)
	}
	
	expected := []int{0, 1, 2}
	for i, v := range assign {
		if v != expected[i] {
			t.Errorf("Expected assignment %v, got %v", expected, assign)
			break
		}
	}
}

func TestSolveMurtyKBest(t *testing.T) {
	costs := [][]float64{
		{1, 2, 3},
		{2, 1, 3},
		{3, 2, 1},
	}
	
	// Best: (0,0), (1,1), (2,2) -> cost 3
	// 2nd: (0,1), (1,0), (2,2) -> cost 2+2+1 = 5
	// OR (0,0), (1,2), (2,1) -> cost 1+3+2 = 6
	
	results := SolveMurtyKBest(costs, 3)
	
	if len(results) < 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}
	
	if results[0].Cost != 3 {
		t.Errorf("Expected rank 1 cost 3, got %f", results[0].Cost)
	}
	
	if results[1].Cost < 3 {
		t.Errorf("Rank 2 cost %f cannot be less than rank 1 cost %f", results[1].Cost, results[0].Cost)
	}
	
	if results[2].Cost < results[1].Cost {
		t.Errorf("Rank 3 cost %f cannot be less than rank 2 cost %f", results[2].Cost, results[1].Cost)
	}

	t.Logf("Rank 1: %v Cost: %f", results[0].Assignment.TrackToMeas, results[0].Cost)
	t.Logf("Rank 2: %v Cost: %f", results[1].Assignment.TrackToMeas, results[1].Cost)
	t.Logf("Rank 3: %v Cost: %f", results[2].Assignment.TrackToMeas, results[2].Cost)
}

func TestSolveMurtyRectangular(t *testing.T) {
	// More observations than tracks
	costs := [][]float64{
		{10, 5, 20, 30},
		{20, 10, 5, 25},
	}
	
	results := SolveMurtyKBest(costs, 5)
	
	for i, res := range results {
		t.Logf("Rank %d: %v Cost: %f", i+1, res.Assignment.TrackToMeas, res.Cost)
		if len(res.Assignment.TrackToMeas) != 2 {
			t.Errorf("Rank %d: expected 2 assignments, got %d", i+1, len(res.Assignment.TrackToMeas))
		}
	}
}
