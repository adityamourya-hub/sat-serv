// Package alerts provides PostgreSQL data access for the sovereign alerting system.
package alerts

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Repository provides data access for alerts.
type Repository struct {
	pool *pgxpool.Pool
}

// NewRepository creates a new repository with the given connection pool.
func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// InsertAlert stores an alert in the alerts hypertable.
func (r *Repository) InsertAlert(ctx context.Context, alert *EnrichedAlert) error {
	if alert.AlertID == "" {
		alert.AlertID = uuid.New().String()
	}

	query := `
		INSERT INTO alerts (
			alert_id, created_at, category, severity, status,
			primary_norad_id, primary_object_name, secondary_norad_id, secondary_object_name,
			title, message, recommended_action, metrics, kl_divergence, confidence,
			latitude, longitude, altitude_km, sovereign_volume,
			triggered_rule, signature, escalation_deadline
		) VALUES (
			$1, $2, $3, $4, $5,
			$6, $7, $8, $9,
			$10, $11, $12, $13, $14, $15,
			$16, $17, $18, $19,
			$20, $21, $22
		) ON CONFLICT (alert_id) DO NOTHING
	`

	metricsJSON, _ := json.Marshal(alert.Metrics)

	_, err := r.pool.Exec(ctx, query,
		alert.AlertID,
		alert.CreatedAt,
		alert.Category,
		alert.Severity,
		alert.Status,
		alert.PrimaryNoradID,
		alert.PrimaryObjectName,
		alert.SecondaryNoradID,
		alert.SecondaryObjectName,
		alert.Title,
		alert.Message,
		alert.RecommendedAction,
		metricsJSON,
		alert.KLDivergence,
		alert.Confidence,
		alert.Latitude,
		alert.Longitude,
		alert.AltitudeKm,
		alert.SovereignVolume,
		alert.TriggeredRule,
		alert.Signature,
		alert.EscalationDeadline,
	)

	if err != nil {
		return fmt.Errorf("insert alert: %w", err)
	}

	return nil
}

// QueryAlerts retrieves alerts with pagination and filtering.
func (r *Repository) QueryAlerts(ctx context.Context, filter AlertFilter) ([]*EnrichedAlert, int, error) {
	// Build dynamic query
	whereClauses := []string{"1=1"}
	args := []interface{}{}
	argIdx := 1

	if filter.Category != "" {
		whereClauses = append(whereClauses, fmt.Sprintf("category = $%d", argIdx))
		args = append(args, filter.Category)
		argIdx++
	}

	if filter.Severity != "" {
		whereClauses = append(whereClauses, fmt.Sprintf("severity = $%d", argIdx))
		args = append(args, filter.Severity)
		argIdx++
	}

	if filter.Status != "" {
		whereClauses = append(whereClauses, fmt.Sprintf("status = $%d", argIdx))
		args = append(args, filter.Status)
		argIdx++
	}

	if filter.Since != nil {
		whereClauses = append(whereClauses, fmt.Sprintf("created_at >= $%d", argIdx))
		args = append(args, filter.Since)
		argIdx++
	}

	whereSQL := ""
	if len(whereClauses) > 1 {
		whereSQL = "WHERE " + whereClauses[1]
		for i := 2; i < len(whereClauses); i++ {
			whereSQL += " AND " + whereClauses[i]
		}
	}

	// Get total count
	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM alerts %s", whereSQL)
	var total int
	err := r.pool.QueryRow(ctx, countQuery, args...).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("count alerts: %w", err)
	}

	// Get paginated results
	query := fmt.Sprintf(`
		SELECT alert_id, created_at, category, severity, status,
		       primary_norad_id, primary_object_name, secondary_norad_id, secondary_object_name,
		       title, message, recommended_action, metrics, kl_divergence, confidence,
		       latitude, longitude, altitude_km, sovereign_volume,
		       triggered_rule, acknowledged_by, acknowledged_at, ack_action,
		       escalation_deadline, signature
		FROM alerts %s
		ORDER BY created_at DESC
		LIMIT $%d OFFSET $%d
	`, whereSQL, argIdx, argIdx+1)

	args = append(args, filter.Limit, filter.Offset)

	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("query alerts: %w", err)
	}
	defer rows.Close()

	var alerts []*EnrichedAlert
	for rows.Next() {
		var alert EnrichedAlert
		var metricsJSON []byte

		err := rows.Scan(
			&alert.AlertID,
			&alert.CreatedAt,
			&alert.Category,
			&alert.Severity,
			&alert.Status,
			&alert.PrimaryNoradID,
			&alert.PrimaryObjectName,
			&alert.SecondaryNoradID,
			&alert.SecondaryObjectName,
			&alert.Title,
			&alert.Message,
			&alert.RecommendedAction,
			&metricsJSON,
			&alert.KLDivergence,
			&alert.Confidence,
			&alert.Latitude,
			&alert.Longitude,
			&alert.AltitudeKm,
			&alert.SovereignVolume,
			&alert.TriggeredRule,
			&alert.AcknowledgedBy,
			&alert.AcknowledgedAt,
			&alert.AckAction,
			&alert.EscalationDeadline,
			&alert.Signature,
		)
		if err != nil {
			slog.Error("Failed to scan alert row", "error", err)
			continue
		}

		if len(metricsJSON) > 0 {
			json.Unmarshal(metricsJSON, &alert.Metrics)
		}

		alerts = append(alerts, &alert)
	}

	return alerts, total, rows.Err()
}

// AcknowledgeAlert updates alert status and logs the action.
func (r *Repository) AcknowledgeAlert(ctx context.Context, alertID, operatorID, action, notes string) error {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	// Update alert status
	now := time.Now()
	_, err = tx.Exec(ctx, `
		UPDATE alerts
		SET status = 'ACKNOWLEDGED',
		    acknowledged_by = $1,
		    acknowledged_at = $2,
		    ack_action = $3
		WHERE alert_id = $4
	`, operatorID, now, action, alertID)
	if err != nil {
		return fmt.Errorf("update alert: %w", err)
	}

	// Log to audit
	details := map[string]interface{}{
		"action": action,
		"notes":  notes,
	}
	detailsJSON, _ := json.Marshal(details)

	_, err = tx.Exec(ctx, `
		INSERT INTO audit_log (logged_at, actor_id, actor_role, action, entity_type, entity_id, details)
		VALUES (NOW(), $1, 'OPERATOR', 'ALERT_ACKNOWLEDGED', 'alert', $2, $3)
	`, operatorID, alertID, detailsJSON)
	if err != nil {
		return fmt.Errorf("insert audit log: %w", err)
	}

	return tx.Commit(ctx)
}

// InsertAuditLog writes an audit entry.
func (r *Repository) InsertAuditLog(ctx context.Context, entry *AuditEntry) error {
	detailsJSON, _ := json.Marshal(entry.Details)
	prevJSON, _ := json.Marshal(entry.PreviousState)
	newJSON, _ := json.Marshal(entry.NewState)

	query := `
		INSERT INTO audit_log (
			logged_at, actor_id, actor_role, action, entity_type, entity_id,
			details, previous_state, new_state, signature
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`

	_, err := r.pool.Exec(ctx, query,
		entry.LoggedAt,
		entry.ActorID,
		entry.ActorRole,
		entry.Action,
		entry.EntityType,
		entry.EntityID,
		detailsJSON,
		prevJSON,
		newJSON,
		entry.Signature,
	)

	if err != nil {
		return fmt.Errorf("insert audit log: %w", err)
	}

	return nil
}

// GetAlertByID retrieves a single alert by ID.
func (r *Repository) GetAlertByID(ctx context.Context, alertID string) (*EnrichedAlert, error) {
	var alert EnrichedAlert
	var metricsJSON []byte

	query := `
		SELECT alert_id, created_at, category, severity, status,
		       primary_norad_id, primary_object_name, secondary_norad_id, secondary_object_name,
		       title, message, recommended_action, metrics, kl_divergence, confidence,
		       latitude, longitude, altitude_km, sovereign_volume,
		       triggered_rule, acknowledged_by, acknowledged_at, ack_action,
		       escalation_deadline, signature
		FROM alerts
		WHERE alert_id = $1
	`

	err := r.pool.QueryRow(ctx, query, alertID).Scan(
		&alert.AlertID,
		&alert.CreatedAt,
		&alert.Category,
		&alert.Severity,
		&alert.Status,
		&alert.PrimaryNoradID,
		&alert.PrimaryObjectName,
		&alert.SecondaryNoradID,
		&alert.SecondaryObjectName,
		&alert.Title,
		&alert.Message,
		&alert.RecommendedAction,
		&metricsJSON,
		&alert.KLDivergence,
		&alert.Confidence,
		&alert.Latitude,
		&alert.Longitude,
		&alert.AltitudeKm,
		&alert.SovereignVolume,
		&alert.TriggeredRule,
		&alert.AcknowledgedBy,
		&alert.AcknowledgedAt,
		&alert.AckAction,
		&alert.EscalationDeadline,
		&alert.Signature,
	)

	if err != nil {
		return nil, fmt.Errorf("get alert by ID: %w", err)
	}

	if len(metricsJSON) > 0 {
		json.Unmarshal(metricsJSON, &alert.Metrics)
	}

	return &alert, nil
}

// AlertFilter specifies query parameters for alert retrieval.
type AlertFilter struct {
	Category Category
	Severity Severity
	Status   Status
	Since    *time.Time
	Limit    int
	Offset   int
}

// DefaultFilter returns a filter with sensible defaults.
func DefaultFilter() AlertFilter {
	return AlertFilter{
		Limit:  100,
		Offset: 0,
	}
}
