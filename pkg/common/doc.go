// Package common contains shared cross-cutting types and utilities used across SSIS packages.
package common

import (
	"sync"
	"time"
)

// AuditLog provides structured logging for audit trails.
type AuditLog struct {
	entries []AuditEntry
	mu      sync.RWMutex
}

// AuditEntry represents a single audit log entry.
type AuditEntry struct {
	Timestamp time.Time
	ClientID  string
	Action    string
	Details   string
	Status    string
	LatencyMs float64
}

// NewAuditLog creates a new audit log.
func NewAuditLog() *AuditLog {
	return &AuditLog{
		entries: make([]AuditEntry, 0),
	}
}

// LogRPC logs an RPC call for audit purposes.
func (a *AuditLog) LogRPC(method string, latency time.Duration, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	status := "success"
	if err != nil {
		status = "error"
	}

	entry := AuditEntry{
		Timestamp: time.Now(),
		Action:    method,
		Details:   "",
		Status:    status,
		LatencyMs: float64(latency.Milliseconds()),
	}
	a.entries = append(a.entries, entry)
}

// LogError logs an error event.
func (a *AuditLog) LogError(clientID, action string, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	entry := AuditEntry{
		Timestamp: time.Now(),
		ClientID:  clientID,
		Action:    action,
		Details:   err.Error(),
		Status:    "error",
	}
	a.entries = append(a.entries, entry)
}

// LogSuccess logs a successful event.
func (a *AuditLog) LogSuccess(clientID, action string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	entry := AuditEntry{
		Timestamp: time.Now(),
		ClientID:  clientID,
		Action:    action,
		Status:    "success",
	}
	a.entries = append(a.entries, entry)
}

// LogCommand logs a command event.
func (a *AuditLog) LogCommand(clientID, command string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	entry := AuditEntry{
		Timestamp: time.Now(),
		ClientID:  clientID,
		Action:    command,
		Status:    "command",
	}
	a.entries = append(a.entries, entry)
}
