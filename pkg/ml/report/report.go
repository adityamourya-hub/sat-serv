package report

import (
	"encoding/json"
	"fmt"
	"ssis/pkg/cryptography"
	"strings"
	"time"
)

type AlertSeverity string

const (
	SeverityLow      AlertSeverity = "low"
	SeverityMedium   AlertSeverity = "medium"
	SeverityHigh     AlertSeverity = "high"
	SeverityCritical AlertSeverity = "critical"
)

type SSAReportInput struct {
	EventID              string
	EventType            string
	Severity             AlertSeverity
	AssetID              int
	AssetName            string
	ThreatID             int
	ThreatName           string
	TimestampUnix        int64
	MissDistanceKm       float64
	CollisionProbability float64
	TcaMinutes           float64
	PriorityScore        float64
	RecommendedAction    string
	Confidence           float64
}

type SSAReport struct {
	EventID       string        `json:"event_id"`
	TimestampUnix int64         `json:"timestamp_unix"`
	Language      string        `json:"language"`
	Severity      AlertSeverity `json:"severity"`
	Summary       string        `json:"summary"`
	Action        string        `json:"action"`
	Confidence    float64       `json:"confidence"`
	ModelName     string        `json:"model_name"`
	SignedProof   string        `json:"signed_proof"`
	PublicKeyHex  string        `json:"public_key_hex,omitempty"`
}

type NarrativeModel interface {
	Name() string
	GenerateEnglish(input SSAReportInput) (summary string, action string, confidence float64, err error)
}

type Generator struct {
	Signer *cryptography.Signer
	Model  NarrativeModel
}

func NewGenerator(signer *cryptography.Signer, model NarrativeModel) *Generator {
	return &Generator{Signer: signer, Model: model}
}

func (g *Generator) GenerateEnglishSignedReport(input SSAReportInput) (SSAReport, error) {
	if input.TimestampUnix <= 0 {
		input.TimestampUnix = time.Now().Unix()
	}
	if strings.TrimSpace(input.EventID) == "" {
		input.EventID = fmt.Sprintf("EV-%d-%d", input.AssetID, input.TimestampUnix)
	}
	if strings.TrimSpace(input.EventType) == "" {
		input.EventType = "SSA_ALERT"
	}

	summary, action, conf, modelName := fallbackNarrative(input)
	if g.Model != nil {
		mSummary, mAction, mConf, err := g.Model.GenerateEnglish(input)
		if err == nil && strings.TrimSpace(mSummary) != "" {
			summary = strings.TrimSpace(mSummary)
			if strings.TrimSpace(mAction) != "" {
				action = strings.TrimSpace(mAction)
			}
			if mConf > 0 {
				conf = clamp01(mConf)
			}
			modelName = g.Model.Name()
		}
	}

	report := SSAReport{
		EventID:       input.EventID,
		TimestampUnix: input.TimestampUnix,
		Language:      "en",
		Severity:      input.Severity,
		Summary:       summary,
		Action:        action,
		Confidence:    conf,
		ModelName:     modelName,
	}

	if g.Signer != nil {
		proof := g.Signer.GenerateProof(input.AssetID, input.EventType, input.TimestampUnix)
		report.SignedProof = proof
		report.PublicKeyHex = g.Signer.PublicKeyHex()
	}

	return report, nil
}

func (g *Generator) GenerateEnglishSignedReportJSON(input SSAReportInput) (string, error) {
	rep, err := g.GenerateEnglishSignedReport(input)
	if err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func fallbackNarrative(input SSAReportInput) (summary string, action string, confidence float64, modelName string) {
	sev := input.Severity
	if sev == "" {
		sev = SeverityMedium
	}

	conf := input.Confidence
	if conf <= 0 {
		conf = 0.72
	}

	summary = fmt.Sprintf(
		"SSA alert %s: asset %s (%d) reports a %s threat %s (%d). Miss distance is %.3f km, estimated collision probability is %.2e, and time-to-closest-approach is %.1f minutes.",
		input.EventID,
		defaultStr(input.AssetName, "UNKNOWN_ASSET"),
		input.AssetID,
		sev,
		defaultStr(input.ThreatName, "UNKNOWN_THREAT"),
		input.ThreatID,
		input.MissDistanceKm,
		input.CollisionProbability,
		input.TcaMinutes,
	)

	action = strings.TrimSpace(input.RecommendedAction)
	if action == "" {
		switch {
		case input.CollisionProbability >= 1e-3 || input.PriorityScore >= 140:
			action = "Initiate immediate collision-avoidance protocol and retask sensors for continuous tracking."
		case input.CollisionProbability >= 1e-5 || input.PriorityScore >= 90:
			action = "Prepare evasive maneuver options and maintain high-frequency tracking updates."
		default:
			action = "Continue monitoring and maintain standard conjunction screening cadence."
		}
	}

	return summary, action, clamp01(conf), "template_fallback_en"
}

func defaultStr(v, d string) string {
	if strings.TrimSpace(v) == "" {
		return d
	}
	return v
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
