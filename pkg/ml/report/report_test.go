package report

import (
	"strings"
	"testing"

	engcrypto "ssis/pkg/crypto"
)

func TestGenerateEnglishSignedReportFallback(t *testing.T) {
	signer := engcrypto.NewSigner("test-key")
	gen := NewGenerator(signer, nil)

	in := SSAReportInput{
		EventID:              "EV-123",
		EventType:            "CONJUNCTION_ALERT",
		Severity:             SeverityHigh,
		AssetID:              43286,
		AssetName:            "IRNSS-1I",
		ThreatID:             99999,
		ThreatName:           "DEBRIS_UNKNOWN",
		TimestampUnix:        1712736000,
		MissDistanceKm:       0.028,
		CollisionProbability: 2.4e-5,
		TcaMinutes:           11.2,
		PriorityScore:        112,
	}

	rep, err := gen.GenerateEnglishSignedReport(in)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if rep.Language != "en" {
		t.Fatalf("expected english language output")
	}
	if rep.SignedProof == "" {
		t.Fatalf("expected signed proof")
	}
	if !strings.Contains(rep.Summary, "IRNSS-1I") {
		t.Fatalf("expected human-readable summary with asset name")
	}
}

func TestGenerateEnglishSignedReportJSON(t *testing.T) {
	signer := engcrypto.NewSigner("another-key")
	gen := NewGenerator(signer, NewQuantizedLLMStub("llama3-8b-4bit-stub"))

	jsonOut, err := gen.GenerateEnglishSignedReportJSON(SSAReportInput{
		EventID:              "EV-987",
		EventType:            "RAPID_THREAT_ALERT",
		Severity:             SeverityCritical,
		AssetID:              45034,
		ThreatID:             77777,
		MissDistanceKm:       0.012,
		CollisionProbability: 8.1e-4,
		TcaMinutes:           3.8,
		PriorityScore:        170,
		TimestampUnix:        1712737000,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !strings.Contains(jsonOut, "\"signed_proof\"") {
		t.Fatalf("expected signed_proof field in JSON output")
	}
}
