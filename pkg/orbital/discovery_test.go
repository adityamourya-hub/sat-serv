package orbital

import (
	"os"
	"testing"
)

func TestDiscoveryBridge(t *testing.T) {
	// 1. Create a dummy weights file for testing
	weightsPath := "test_axiom_weights.json"
	
	// Mock weights for a 13-feature manifold
	// num/den shape: [1][13][5] (outDim, inDim, degree+1)
	mockWeights := `{
		"degree": 4,
		"num": [[[0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
		"den": [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
	}`
	
	err := os.WriteFile(weightsPath, []byte(mockWeights), 0644)
	if err != nil {
		t.Fatalf("failed to write mock weights: %v", err)
	}
	defer os.Remove(weightsPath)

	// 2. Run the bridge
	t.Logf("Running Discovery Bridge verification...")
	law, err := DistillSovereignLaw(weightsPath)
	if err != nil {
		t.Fatalf("DiscoveryBridge failed: %v", err)
	}

	if law == "" {
		t.Errorf("expected LaTeX law result, got empty string")
	}

	t.Logf("Distilled Law: %s", law)
}
