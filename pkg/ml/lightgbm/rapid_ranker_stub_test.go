package lightgbm

import "testing"

func TestNewRapidRankerStub(t *testing.T) {
	m, err := NewRapidRanker("rapid_ranker.txt")
	if err != nil {
		t.Fatalf("expected stub constructor to succeed, got err=%v", err)
	}
	if m.Name() == "" {
		t.Fatalf("expected non-empty model name")
	}
}
