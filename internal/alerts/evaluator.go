// Package alerts provides the alert evaluator goroutine for processing engine events.
package alerts

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"ssis/pkg/cryptography"
	"ssis/pkg/types"
)

// Evaluator processes alerts from the EventBus, matches rules, enriches, and persists.
type Evaluator struct {
	rules      *RulesEngine
	repository *Repository
	sseHub     *SSEHub
	signer     *cryptography.Signer
	alertChan  chan *types.Alert
	stopChan   chan struct{}
}

// NewEvaluator creates a new alert evaluator.
func NewEvaluator(rulesConfigPath string, repository *Repository, sseHub *SSEHub, signer *cryptography.Signer) (*Evaluator, error) {
	rules, err := NewRulesEngine(rulesConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create rules engine: %w", err)
	}

	return &Evaluator{
		rules:      rules,
		repository: repository,
		sseHub:     sseHub,
		signer:     signer,
		alertChan:  make(chan *types.Alert, 1000),
		stopChan:   make(chan struct{}),
	}, nil
}

// Start begins the evaluator goroutine.
func (e *Evaluator) Start(eventBus *types.EventBus) {
	// Subscribe to EventBus
	alertCh := make(chan *types.Alert, 100)
	eventBus.Subscribe(alertCh)

	// Process alerts in goroutine
	go e.processAlerts(alertCh)

	slog.Info("Alert evaluator started")
}

// Stop gracefully shuts down the evaluator.
func (e *Evaluator) Stop(ctx context.Context) error {
	close(e.stopChan)

	// Wait for alert channel to drain or timeout
	select {
	case <-time.After(5 * time.Second):
		slog.Warn("Alert evaluator shutdown timeout")
	case <-ctx.Done():
		return ctx.Err()
	}

	slog.Info("Alert evaluator stopped")
	return nil
}

// processAlerts handles incoming alerts from the EventBus.
func (e *Evaluator) processAlerts(alertCh <-chan *types.Alert) {
	for {
		select {
		case <-e.stopChan:
			return
		case alert := <-alertCh:
			if err := e.evaluateAndEnrich(alert); err != nil {
				slog.Error("Failed to evaluate alert", "error", err, "alert_id", alert.AlertID)
			}
		}
	}
}

// evaluateAndEnrich matches the alert against rules and enriches it.
func (e *Evaluator) evaluateAndEnrich(engineAlert *types.Alert) error {
	// Convert types.Alert to our Category
	category := e.mapAlertTypeToCategory(engineAlert.Type)

	// Extract metrics from engine alert
	metrics := e.extractMetrics(engineAlert)

	// Convert to map[string]interface{} for EnrichedAlert
	metricsInterface := make(map[string]interface{})
	for k, v := range metrics {
		metricsInterface[k] = v
	}

	// Match against rules
	rule, matched := e.rules.Evaluate(category, metrics)
	if !matched {
		// No rule matched, skip enrichment
		return nil
	}

	// Enrich alert
	enriched := &EnrichedAlert{
		CreatedAt:         engineAlert.Timestamp,
		Category:          category,
		Severity:          rule.Severity,
		Status:            StatusOpen,
		PrimaryNoradID:    e.parseNoradID(engineAlert.TargetID),
		SecondaryNoradID:  e.parseNoradID(engineAlert.SecondaryID),
		Title:             e.generateTitle(category, engineAlert),
		Message:           engineAlert.Message,
		RecommendedAction: e.generateAction(rule.Severity, category),
		Metrics:           metricsInterface,
		KLDivergence:      engineAlert.KLDivergence,
		Confidence:        engineAlert.Probability,
		TriggeredRule:     rule.ID,
	}

	// Calculate escalation deadline
	if rule.Escalation.TimeoutMin > 0 {
		deadline := enriched.CreatedAt.Add(time.Duration(rule.Escalation.TimeoutMin) * time.Minute)
		enriched.EscalationDeadline = &deadline
	}

	// Sign the alert
	if e.signer != nil {
		sig := e.signAlert(enriched)
		enriched.Signature = sig
	}

	// Persist to database
	ctx := context.Background()
	if err := e.repository.InsertAlert(ctx, enriched); err != nil {
		return fmt.Errorf("insert alert: %w", err)
	}

	// Broadcast to SSE hub
	e.sseHub.Broadcast(enriched)

	// Log audit entry
	settings := e.rules.GetSettings()
	if settings.AuditEveryAlert {
		auditEntry := &AuditEntry{
			LoggedAt:   time.Now(),
			ActorID:    "SYSTEM",
			ActorRole:  RoleSystem,
			Action:     "ALERT_CREATED",
			EntityType: "alert",
			EntityID:   enriched.AlertID,
			Details:    map[string]interface{}{"rule_id": rule.ID, "severity": rule.Severity},
			NewState:   enriched,
		}
		if e.signer != nil {
			auditEntry.Signature = e.signAudit(auditEntry)
		}
		if err := e.repository.InsertAuditLog(ctx, auditEntry); err != nil {
			slog.Error("Failed to insert audit log", "error", err)
		}
	}

	return nil
}

// mapAlertTypeToCategory converts types.AlertType to Category.
func (e *Evaluator) mapAlertTypeToCategory(alertType types.AlertType) Category {
	switch alertType {
	case types.AlertTypeConjunction:
		return CategoryConjunction
	case types.AlertTypeManeuver:
		return CategoryManeuver
	case types.AlertTypeReentry:
		return CategoryReentry
	case types.AlertTypeCollisionRisk:
		return CategoryCollisionRisk
	case types.AlertTypeAnomaly:
		return CategoryAnomaly
	default:
		return CategoryAnomaly
	}
}

// extractMetrics extracts numeric metrics from the engine alert.
func (e *Evaluator) extractMetrics(alert *types.Alert) map[string]float64 {
	metrics := make(map[string]float64)

	// Add probability
	if alert.Probability > 0 {
		metrics["probability"] = alert.Probability
	}

	// Add KL divergence
	if alert.KLDivergence > 0 {
		metrics["kl_divergence"] = alert.KLDivergence
	}

	// Add SHAP scores if available
	if alert.Explanation != nil {
		for feature, score := range alert.Explanation.FeatureScores {
			metrics[feature] = score
		}
	}

	return metrics
}

// parseNoradID extracts NORAD ID from string.
func (e *Evaluator) parseNoradID(id string) int {
	var noradID int
	fmt.Sscanf(id, "%d", &noradID)
	return noradID
}

// generateTitle creates a human-readable title.
func (e *Evaluator) generateTitle(category Category, alert *types.Alert) string {
	switch category {
	case CategoryConjunction:
		return fmt.Sprintf("Conjunction Risk: %s vs %s", alert.TargetID, alert.SecondaryID)
	case CategoryManeuver:
		return fmt.Sprintf("Maneuver Detected: %s", alert.TargetID)
	case CategoryReentry:
		return fmt.Sprintf("Re-entry Risk: %s", alert.TargetID)
	case CategoryAnomaly:
		return fmt.Sprintf("Anomaly Detected: %s", alert.TargetID)
	default:
		return "Alert Generated"
	}
}

// generateAction provides recommended action based on severity.
func (e *Evaluator) generateAction(severity Severity, category Category) string {
	switch severity {
	case SeverityInfo:
		return "Monitor situation"
	case SeverityWarning:
		return "Increase observation frequency"
	case SeverityCritical:
		return "Prepare for potential action"
	case SeverityEmergency:
		return "Immediate action required"
	default:
		return "Review alert"
	}
}

// signAlert creates a signature for the alert.
func (e *Evaluator) signAlert(alert *EnrichedAlert) string {
	payload := fmt.Sprintf("%s:%s:%s:%d", alert.AlertID, alert.Category, alert.Severity, alert.CreatedAt.Unix())
	return e.signer.Sign([]byte(payload))
}

// signAudit creates a signature for the audit entry.
func (e *Evaluator) signAudit(entry *AuditEntry) string {
	payload := fmt.Sprintf("%d:%s:%s:%s", entry.LoggedAt.Unix(), entry.ActorID, entry.Action, entry.EntityID)
	return e.signer.Sign([]byte(payload))
}

// GetAlertChannel returns the alert channel for external submission.
func (e *Evaluator) GetAlertChannel() chan<- *types.Alert {
	return e.alertChan
}
