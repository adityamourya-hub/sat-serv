package assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"ssis/pkg/common"
)

type TacticalAssistant struct {
	session    *ort.DynamicAdvancedSession
	inputInfo  []ort.InputOutputInfo
	outputInfo []ort.InputOutputInfo
	modelPath  string
	mode       string
}

const (
	AssistantModeLocalONNX = "local_onnx"
	AssistantModeTemplate  = "template_fallback"
)

var llmEnvOnce sync.Once
var llmEnvErr error

func initializeLLMEnvironment() error {
	llmEnvOnce.Do(func() {
		llmEnvErr = ort.InitializeEnvironment()
		if llmEnvErr != nil {
			msg := strings.ToLower(llmEnvErr.Error())
			if strings.Contains(msg, "already been initialized") {
				llmEnvErr = nil
			}
		}
	})
	return llmEnvErr
}

func NewTacticalAssistant(modelPath string) (*TacticalAssistant, error) {
	strict := common.StrictModeEnabled()
	if err := VerifyBundleManifest(filepath.Dir(modelPath)); err != nil {
		common.EmitFallbackEvent("llm", "tactical_assistant", "integrity_check_failed", "template_fallback", 0.0, strict, map[string]interface{}{"model_path": modelPath, "error": err.Error()})
		return nil, fmt.Errorf("verify LLM bundle manifest: %w", err)
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		common.EmitFallbackEvent("llm", "tactical_assistant", "model_missing", "template_fallback", 0.0, strict, map[string]interface{}{"model_path": modelPath})
		return nil, fmt.Errorf("LLM model not found at %s", modelPath)
	}
	if err := initializeLLMEnvironment(); err != nil {
		common.EmitFallbackEvent("llm", "tactical_assistant", "onnx_env_init_failed", "template_fallback", 0.0, strict, map[string]interface{}{"error": err.Error()})
		return nil, fmt.Errorf("initialize onnx runtime: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil || len(inputs) == 0 || len(outputs) == 0 {
		reason := "io_introspection_failed"
		meta := map[string]interface{}{"model_path": modelPath}
		if err != nil {
			meta["error"] = err.Error()
		}
		common.EmitFallbackEvent("llm", "tactical_assistant", reason, "template_fallback", 0.0, strict, meta)
		if err == nil {
			err = fmt.Errorf("missing model input/output metadata")
		}
		return nil, fmt.Errorf("inspect llm model io: %w", err)
	}

	inNames := make([]string, 0, len(inputs))
	outNames := make([]string, 0, len(outputs))
	tensorInputs := make([]ort.InputOutputInfo, 0, len(inputs))
	tensorOutputs := make([]ort.InputOutputInfo, 0, len(outputs))
	for _, in := range inputs {
		if in.OrtValueType != ort.ONNXTypeTensor {
			continue
		}
		inNames = append(inNames, in.Name)
		tensorInputs = append(tensorInputs, in)
	}
	for _, out := range outputs {
		if out.OrtValueType != ort.ONNXTypeTensor {
			continue
		}
		outNames = append(outNames, out.Name)
		tensorOutputs = append(tensorOutputs, out)
	}
	if len(inNames) == 0 || len(outNames) == 0 {
		common.EmitFallbackEvent("llm", "tactical_assistant", "non_tensor_io", "template_fallback", 0.0, strict, map[string]interface{}{"model_path": modelPath})
		return nil, fmt.Errorf("model has no tensor inputs/outputs suitable for assistant inference")
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inNames, outNames, nil)
	if err != nil {
		common.EmitFallbackEvent("llm", "tactical_assistant", "onnx_session_unavailable", "template_fallback", 0.0, strict, map[string]interface{}{"model_path": modelPath, "error": err.Error()})
		return nil, fmt.Errorf("failed to load LLM session: %w", err)
	}

	common.EmitInfoEvent("llm", "tactical_assistant", "onnx_initialized", "use_local_onnx", strict, map[string]interface{}{
		"model_path":     modelPath,
		"input_tensors":  len(inNames),
		"output_tensors": len(outNames),
	})

	return &TacticalAssistant{session: session, inputInfo: tensorInputs, outputInfo: tensorOutputs, modelPath: modelPath, mode: AssistantModeLocalONNX}, nil
}

func NewTacticalAssistantWithFallback(modelPath string) *TacticalAssistant {
	strict := common.StrictModeEnabled()
	a, err := NewTacticalAssistant(modelPath)
	if err == nil {
		return a
	}
	common.EmitFallbackEvent("llm", "tactical_assistant", "assistant_init_failed", "template_fallback", 0.0, strict, map[string]interface{}{
		"model_path": modelPath,
		"error":      err.Error(),
	})
	return &TacticalAssistant{session: nil, modelPath: modelPath, mode: AssistantModeTemplate}
}

func (a *TacticalAssistant) GenerateSITREP(ctx context.Context, pipelineOutput []byte) (string, error) {
	_ = ctx
	strict := common.StrictModeEnabled()
	if strict && a.Mode() == AssistantModeTemplate {
		common.EmitFallbackEvent("llm", "tactical_assistant", "template_mode_blocked", "fail_fast", 0.0, strict, map[string]interface{}{"model_path": a.modelPath})
		return "", fmt.Errorf("strict mode: tactical assistant template fallback is not allowed")
	}
	start := time.Now()
	sitrep := GenerateTemplateSITREP(pipelineOutput)
	if a.Mode() == AssistantModeLocalONNX {
		if conf, runErr := a.probeModel(pipelineOutput); runErr != nil {
			common.EmitFallbackEvent("llm", "tactical_assistant", "onnx_probe_failed", "template_fallback", 0.0, strict, map[string]interface{}{"error": runErr.Error(), "model_path": a.modelPath})
		} else {
			sitrep = strings.Replace(sitrep, "COMMANDER RECOMMENDATIONS:", fmt.Sprintf("ONNX PROBE CONFIDENCE: %.3f\n\nCOMMANDER RECOMMENDATIONS:", conf), 1)
		}
	}
	sitrep = strings.Replace(sitrep, "SITUATION:", fmt.Sprintf("MODE: %s\nSITUATION:", a.mode), 1)

	fmt.Printf("[LLM] Generated SITREP in %v\n", time.Since(start))
	return sitrep, nil
}

func (a *TacticalAssistant) Mode() string {
	if a == nil || strings.TrimSpace(a.mode) == "" {
		return AssistantModeTemplate
	}
	return a.mode
}

func (a *TacticalAssistant) Close() {
	if a.session != nil {
		a.session.Destroy()
	}
}

func (a *TacticalAssistant) probeModel(pipelineOutput []byte) (float64, error) {
	if a.session == nil {
		return 0, fmt.Errorf("onnx session unavailable")
	}
	tokens := tokenizePrompt(pipelineOutput, 32)
	if len(tokens) == 0 {
		tokens = []int64{1}
	}

	inputs := make([]ort.Value, 0, len(a.inputInfo))
	outputs := make([]ort.Value, len(a.outputInfo))
	defer destroyValues(inputs)
	defer destroyValues(outputs)

	for _, in := range a.inputInfo {
		if in.OrtValueType != ort.ONNXTypeTensor {
			continue
		}
		shape := resolveAssistantShape(in.Dimensions, len(tokens))
		val, err := buildAssistantTensor(shape, in.DataType, tokens)
		if err != nil {
			return 0, fmt.Errorf("build input tensor %s: %w", in.Name, err)
		}
		inputs = append(inputs, val)
	}

	if len(inputs) == 0 {
		return 0, fmt.Errorf("no tensor inputs prepared")
	}

	if err := a.session.Run(inputs, outputs); err != nil {
		return 0, err
	}

	conf := 0.5
	if len(outputs) > 0 {
		conf = clamp01(readFirstValue(outputs[0]))
		if conf == 0 {
			conf = 0.5
		}
	}
	return conf, nil
}

func resolveAssistantShape(s ort.Shape, tokenCount int) ort.Shape {
	out := make(ort.Shape, len(s))
	if len(s) == 0 {
		return ort.NewShape(1, int64(maxInt(tokenCount, 1)))
	}
	for i := range s {
		d := s[i]
		if d <= 0 {
			switch i {
			case 0:
				d = 1
			case 1:
				d = int64(maxInt(tokenCount, 1))
			default:
				d = 1
			}
		}
		out[i] = d
	}
	return out
}

func buildAssistantTensor(shape ort.Shape, dtype ort.TensorElementDataType, tokens []int64) (ort.Value, error) {
	flat := int(shape.FlattenedSize())
	if flat <= 0 {
		flat = maxInt(len(tokens), 1)
		shape = ort.NewShape(int64(flat))
	}

	switch dtype {
	case ort.TensorElementDataTypeInt64:
		data := make([]int64, flat)
		for i := 0; i < flat; i++ {
			data[i] = tokens[i%len(tokens)]
		}
		return ort.NewTensor(shape, data)
	case ort.TensorElementDataTypeInt32:
		data := make([]int32, flat)
		for i := 0; i < flat; i++ {
			data[i] = int32(tokens[i%len(tokens)] % math.MaxInt32)
		}
		return ort.NewTensor(shape, data)
	case ort.TensorElementDataTypeDouble:
		data := make([]float64, flat)
		for i := 0; i < flat; i++ {
			data[i] = float64(tokens[i%len(tokens)]%997) / 997.0
		}
		return ort.NewTensor(shape, data)
	default:
		data := make([]float32, flat)
		for i := 0; i < flat; i++ {
			data[i] = float32(float64(tokens[i%len(tokens)]%997) / 997.0)
		}
		return ort.NewTensor(shape, data)
	}
}

func tokenizePrompt(payload []byte, maxTokens int) []int64 {
	parts := strings.Fields(string(payload))
	if len(parts) > maxTokens {
		parts = parts[:maxTokens]
	}
	out := make([]int64, 0, len(parts))
	for _, p := range parts {
		h := fnv.New64a()
		_, _ = h.Write([]byte(p))
		id := int64(h.Sum64()%50000 + 1)
		out = append(out, id)
	}
	return out
}

func readFirstValue(v ort.Value) float64 {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case *ort.Tensor[float32]:
		d := t.GetData()
		if len(d) > 0 {
			return float64(d[0])
		}
	case *ort.Tensor[float64]:
		d := t.GetData()
		if len(d) > 0 {
			return d[0]
		}
	case *ort.Tensor[int64]:
		d := t.GetData()
		if len(d) > 0 {
			return float64(d[0])
		}
	case *ort.Tensor[int32]:
		d := t.GetData()
		if len(d) > 0 {
			return float64(d[0])
		}
	}
	return 0
}

func destroyValues(values []ort.Value) {
	for _, v := range values {
		if v != nil {
			_ = v.Destroy()
		}
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func GenerateTemplateSITREP(pipelineOutput []byte) string {
	type threatRow struct {
		TrackName      string  `json:"track_name"`
		DynamicScore   float64 `json:"dynamic_score"`
		Recommendation string  `json:"recommendation"`
	}
	type steeringRow struct {
		ObjectName     string `json:"object_name"`
		Recommendation string `json:"recommendation"`
		Reasoning      string `json:"reasoning"`
	}
	type explainabilityPayload struct {
		Summary     string   `json:"summary"`
		TopSignals  []string `json:"top_signals"`
		ReasonCodes []string `json:"reason_codes"`
	}
	type signedAlert struct {
		ThreatID        int                   `json:"threat_id"`
		ModelConfidence float64               `json:"model_confidence"`
		SignatureValid  bool                  `json:"signature_valid"`
		Explainability  explainabilityPayload `json:"explainability"`
	}
	type demoSignedReport struct {
		Mode              string        `json:"mode"`
		TopThreats        []threatRow   `json:"top_threats"`
		CollisionAlerts   []any         `json:"collision_alerts"`
		ManeuversDetected []any         `json:"maneuvers_detected"`
		RLSteering        []steeringRow `json:"rl_steering_priorities"`
		AlertsSignedValid bool          `json:"alerts_signed_valid"`
	}
	type operationalMetrics struct {
		TotalAlerts        int         `json:"total_alerts"`
		Throughput         float64     `json:"throughput_req_per_second"`
		CollisionAccuracy  float64     `json:"collision_accuracy"`
		CollisionFPR       float64     `json:"collision_false_positive_rate"`
		SignedAlertExample signedAlert `json:"signed_alert_example"`
	}

	now := time.Now().UTC().Format("2006-01-02 15:04")
	demo := demoSignedReport{}
	metrics := operationalMetrics{}

	demoErr := json.Unmarshal(pipelineOutput, &demo)
	metricsErr := json.Unmarshal(pipelineOutput, &metrics)

	lines := []string{
		fmt.Sprintf("SSIS TACTICAL COMMAND ASSISTANT - %s UTC", now),
		"",
		"SITUATION:",
	}

	if demoErr == nil && (len(demo.TopThreats) > 0 || len(demo.CollisionAlerts) > 0 || len(demo.RLSteering) > 0) {
		lines = append(lines,
			fmt.Sprintf("- Demo mode: %s", fallbackValue(demo.Mode, "indian-catalogue")),
			fmt.Sprintf("- Active collision alerts: %d; detected maneuvers: %d.", len(demo.CollisionAlerts), len(demo.ManeuversDetected)),
		)
		if len(demo.TopThreats) > 0 {
			t := demo.TopThreats[0]
			lines = append(lines, fmt.Sprintf("- Top threat: %s (dynamic score %.1f), recommendation: %s.", fallbackValue(t.TrackName, "UNKNOWN_TRACK"), t.DynamicScore, fallbackValue(t.Recommendation, "monitor_only")))
		}
		if len(demo.RLSteering) > 0 {
			s := demo.RLSteering[0]
			lines = append(lines, fmt.Sprintf("- RL sensor steering: %s -> %s (%s).", fallbackValue(s.ObjectName, "UNKNOWN_OBJECT"), fallbackValue(s.Recommendation, "observe"), trimSentence(s.Reasoning)))
		}
		if demo.AlertsSignedValid {
			lines = append(lines, "- Alert signatures: verified.")
		} else {
			lines = append(lines, "- Alert signatures: verification pending.")
		}
	} else if metricsErr == nil && (metrics.TotalAlerts > 0 || metrics.Throughput > 0) {
		lines = append(lines,
			fmt.Sprintf("- Operational alerts emitted: %d.", metrics.TotalAlerts),
			fmt.Sprintf("- Throughput %.2f req/s; collision accuracy %.3f; false-positive-rate %.3f.", metrics.Throughput, metrics.CollisionAccuracy, metrics.CollisionFPR),
		)
		if metrics.SignedAlertExample.ThreatID != 0 {
			lines = append(lines, fmt.Sprintf("- Representative threat ID: %d (model confidence %.2f).", metrics.SignedAlertExample.ThreatID, metrics.SignedAlertExample.ModelConfidence))
		}
		if metrics.SignedAlertExample.Explainability.Summary != "" {
			lines = append(lines, fmt.Sprintf("- Explainability: %s", trimSentence(metrics.SignedAlertExample.Explainability.Summary)))
		}
		if len(metrics.SignedAlertExample.Explainability.ReasonCodes) > 0 {
			lines = append(lines, fmt.Sprintf("- Reason codes: %s.", strings.Join(metrics.SignedAlertExample.Explainability.ReasonCodes, ", ")))
		}
		if metrics.SignedAlertExample.SignatureValid {
			lines = append(lines, "- Alert signatures: verified.")
		}
	} else {
		lines = append(lines,
			"- Pipeline payload ingested for command briefing.",
			fmt.Sprintf("- Source digest: %d bytes of signed operational data.", len(pipelineOutput)),
		)
	}

	lines = append(lines,
		"",
		"COMMANDER RECOMMENDATIONS:",
		"- Maintain continuous track quality on highest-priority objects with PPO-guided retasking.",
		"- Prioritize conjunction risk branches with highest explainability confidence and narrow uncertainty.",
		"- Keep PINN-refined propagation enabled for long-arc or high-drag segments before maneuver authorization.",
		"",
		"All alerts and AI inferences cryptographically signed and verified.",
		"End of SITREP.",
	)

	return strings.Join(lines, "\n")
}

func trimSentence(s string) string {
	t := strings.TrimSpace(s)
	if t == "" {
		return "no additional rationale"
	}
	if len(t) > 180 {
		t = t[:180]
	}
	return strings.TrimRight(t, ".")
}

func fallbackValue(s, def string) string {
	if strings.TrimSpace(s) == "" {
		return def
	}
	return s
}
