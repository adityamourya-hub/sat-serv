// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.
package report

import "fmt"

// QuantizedLLMStub is a default local-model stub for 4-bit Llama/sovereign checkpoints.
// Deployments can replace this with a real local runtime (GGUF, llama.cpp, vLLM, etc.).
type QuantizedLLMStub struct {
	ModelID string
}

func NewQuantizedLLMStub(modelID string) *QuantizedLLMStub {
	if modelID == "" {
		modelID = "llama3-8b-4bit-stub"
	}
	return &QuantizedLLMStub{ModelID: modelID}
}

func (m *QuantizedLLMStub) Name() string {
	return m.ModelID
}

func (m *QuantizedLLMStub) GenerateEnglish(input SSAReportInput) (string, string, float64, error) {
	return "", "", 0, fmt.Errorf("quantized LLM runtime not configured for model %s", m.ModelID)
}
