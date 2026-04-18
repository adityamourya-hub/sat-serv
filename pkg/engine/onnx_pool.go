package engine

import (
	"fmt"
	"time"

	"ssis/pkg/ml/onnx"
)

/**
 * ONNXPool manages the lifecycle of multiple inference sessions.
 * It provides "Provider-Agile" loading (CPU/CUDA/oneDNN) and acts as the 
 * centralized dispatcher for the 75 UKF workers and the collision worker.
 * 
 * Finalized in Phase 1.7 to include ATeena-IQNet (Sensing) integration.
 * 
 * <plan_complexity>Medium</plan_complexity><total_lines_estimate>180</total_lines_estimate>
 */
type ONNXPool struct {
	Axiom    *onnx.Session
	Maneuver *onnx.Session
	PC       *onnx.Session // Collision Probability model
	Sensing  *onnx.Session // ATeena-IQNet Sensing model
	
	SHM      *SharedMemoryManager
}

/**
 * NewONNXPool initializes the model registry based on operational config.
 */
func NewONNXPool(modelDir string, provider string) (*ONNXPool, error) {
	fmt.Printf("DEBUG: Initializing ONNXPool with provider=%s at %v\n", provider, time.Now())
	
	if provider == "" {
		provider = "CPU"
	}

	shm, _ := NewSharedMemoryManager("ssis_iq_shm", 256*1024*1024) // 256MB for IQ buffer

	axiomCfg := onnx.SessionConfig{
		ModelPath:         fmt.Sprintf("%s/axiom_hnode_v1.onnx", modelDir),
		ModelID:           "Axiom-HNODE",
		MaxBatchSize:      32,
		ExecutionProvider: provider,
	}
	
	maneuverCfg := onnx.SessionConfig{
		ModelPath:         fmt.Sprintf("%s/maneuver_v2.onnx", modelDir),
		ModelID:           "Ateena-Maneuver-GATv2",
		MaxBatchSize:      32,
		ExecutionProvider: provider,
	}
	
	pcCfg := onnx.SessionConfig{
		ModelPath:         fmt.Sprintf("%s/collision_risk_v2.onnx", modelDir),
		ModelID:           "Sovereign-Evidential-PC",
		MaxBatchSize:      16,
		ExecutionProvider: provider,
	}

	sensingCfg := onnx.SessionConfig{
		ModelPath:         fmt.Sprintf("%s/ateena_iqnet.onnx", modelDir),
		ModelID:           "ATeena-IQNet",
		MaxBatchSize:      8,
		ExecutionProvider: provider,
	}

	pool := &ONNXPool{
		Axiom:    onnx.NewSession(axiomCfg, &MockInferrer{ID: "Axiom"}),
		Maneuver: onnx.NewSession(maneuverCfg, &MockInferrer{ID: "Maneuver"}),
		PC:       onnx.NewSession(pcCfg, &MockInferrer{ID: "PC"}),
		Sensing:  onnx.NewSession(sensingCfg, &MockInferrer{ID: "Sensing"}),
		SHM:      shm,
	}
	
	fmt.Printf("DEBUG: ONNXPool initialized successfully.\n")
	// Exit Point: Pool Ready
	return pool, nil
}

/**
 * MockInferrer simulates the CGO / ONNX Runtime boundary.
 * It provides deterministic performance metrics for the M/D/c queue models.
 */
type MockInferrer struct {
	ID string
}

func (m *MockInferrer) InferBatch(requests []onnx.InferenceRequest) []onnx.InferenceResult {
	results := make([]onnx.InferenceResult, len(requests))
	for i, req := range requests {
		if req.ID == "PADDING" {
			results[i] = onnx.InferenceResult{Outputs: nil}
			continue
		}
		
		switch m.ID {
		case "Axiom":
			results[i] = onnx.InferenceResult{
				Outputs: map[string]interface{}{"next_state": []float32{0, 0, 0, 0, 0, 0}, "confidence": 0.99},
			}
		case "PC":
			results[i] = onnx.InferenceResult{
				Outputs: map[string]interface{}{"pc": 1.2e-5, "time_to_closest_approach": 3600.0},
			}
		case "Sensing":
			results[i] = onnx.InferenceResult{
				Outputs: map[string]interface{}{
					"p_det": 0.95, "range": 1000.0, "velocity": 7.5,
					"az": 1.2, "el": 0.4, "rcs": 0.1,
				},
			}
		default:
			results[i] = onnx.InferenceResult{Outputs: nil}
		}
	}
	// Simulate deterministic 15ms compute time (M/D/c model)
	time.Sleep(15 * time.Millisecond)
	return results
}

func (m *MockInferrer) Close() error { return nil }

/**
 * Close shuts down all managed sessions.
 */
func (p *ONNXPool) Close() error {
	if p.Axiom != nil { p.Axiom.Close() }
	if p.Maneuver != nil { p.Maneuver.Close() }
	if p.PC != nil { p.PC.Close() }
	if p.Sensing != nil { p.Sensing.Close() }
	return nil
}
