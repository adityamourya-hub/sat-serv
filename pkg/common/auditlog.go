package common

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

var fallbackEventCount uint64

type AuditEvent struct {
	TimestampUTC string                 `json:"timestamp_utc"`
	Type         string                 `json:"type"`
	Domain       string                 `json:"domain"`
	Component    string                 `json:"component"`
	Reason       string                 `json:"reason"`
	Action       string                 `json:"action"`
	Confidence   float64                `json:"confidence,omitempty"`
	Strict       bool                   `json:"strict"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

func EmitFallbackEvent(domain, component, reason, action string, confidence float64, strict bool, metadata map[string]interface{}) {
	e := AuditEvent{
		TimestampUTC: time.Now().UTC().Format(time.RFC3339),
		Type:         "fallback",
		Domain:       domain,
		Component:    component,
		Reason:       reason,
		Action:       action,
		Confidence:   confidence,
		Strict:       strict,
		Metadata:     metadata,
	}
	emitEvent(e)
	atomic.AddUint64(&fallbackEventCount, 1)
}

func FallbackEventCount() uint64 {
	return atomic.LoadUint64(&fallbackEventCount)
}

func ResetFallbackEventCount() {
	atomic.StoreUint64(&fallbackEventCount, 0)
}

func EmitInfoEvent(domain, component, reason, action string, strict bool, metadata map[string]interface{}) {
	e := AuditEvent{
		TimestampUTC: time.Now().UTC().Format(time.RFC3339),
		Type:         "info",
		Domain:       domain,
		Component:    component,
		Reason:       reason,
		Action:       action,
		Strict:       strict,
		Metadata:     metadata,
	}
	emitEvent(e)
}

func StrictModeEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("SSIS_STRICT_MODE")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func emitEvent(e AuditEvent) {
	b, err := json.Marshal(e)
	if err != nil {
		fmt.Fprintf(os.Stderr, "{\"type\":\"audit_error\",\"error\":%q}\n", err.Error())
		return
	}
	fmt.Fprintln(os.Stderr, string(b))
}
