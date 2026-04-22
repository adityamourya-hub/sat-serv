// RRI-Based Automated Alerting Service
// Configures automated alerting at RRI > 1.0 with 15-minute decision window
package main

import (
	"encoding/json"
	"log"
	"math"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ssis/pkg/physics"
	"ssis/pkg/tracking"
)

type AlertingService struct {
	imm           *tracking.IMMFilter
	rriCalculator *tracking.ReachabilityRiskIndex
	alertChannel  chan RRIAlert
	config        AlertConfig
}

type AlertConfig struct {
	RRIThreshold         float64 // Default: 1.0
	DecisionWindowMin    float64 // Default: 15 minutes
	AlertEscalationSteps []EscalationStep
}

type EscalationStep struct {
	RRIThreshold   float64
	AlertLevel     string
	ActionRequired string
}

type RRIAlert struct {
	Timestamp         time.Time
	PrimaryObjectID   int32
	SecondaryObjectID int32
	RRI               float64
	TimeToTCAMinutes  float64
	Pc                float64
	Severity          float64
	IntentProb        float64
	ProbabilityBubble ProbabilityBubble
	AlertLevel        string
	ActionRequired    string
	DecisionWindow    float64
}

type ProbabilityBubble struct {
	RxKm float64 // 3-sigma X bound
	RyKm float64 // 3-sigma Y bound
	RzKm float64 // 3-sigma Z bound
}

func main() {
	log.Printf("[ALERT] Initializing RRI-Based Alerting Service...")

	service := NewAlertingService()

	// Configure alerting thresholds
	service.configureAlerting()

	// Start alert processing
	go service.processAlerts()

	// Run alert monitoring cycle
	service.runAlertMonitoring()

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Printf("[ALERT] Shutting down Alerting Service...")
}

func NewAlertingService() *AlertingService {
	// Initialize IMM filter with sample state
	initialState := physics.State{
		Pos:     [3]float64{7000, 0, 0},
		Vel:     [3]float64{0, 7.5, 0},
		EpochJD: float64(time.Now().Unix()) / 86400.0,
	}

	imm := tracking.NewIMMFilter(12345, initialState)
	rriCalc := tracking.NewReachabilityRiskIndex(imm)

	return &AlertingService{
		imm:           imm,
		rriCalculator: rriCalc,
		alertChannel:  make(chan RRIAlert, 1000),
		config: AlertConfig{
			RRIThreshold:      1.0,
			DecisionWindowMin: 15.0,
		},
	}
}

func (as *AlertingService) configureAlerting() {
	// Configure escalation steps based on RRI thresholds
	as.config.AlertEscalationSteps = []EscalationStep{
		{RRIThreshold: 0.1, AlertLevel: "ROUTINE_MONITORING", ActionRequired: "None"},
		{RRIThreshold: 1.0, AlertLevel: "TRACKING_PRIORITY", ActionRequired: "Increase sensor dwell"},
		{RRIThreshold: 5.0, AlertLevel: "TRACKING_PRIORITY", ActionRequired: "Dedicated sensor tasking"},
		{RRIThreshold: 10.0, AlertLevel: "IMMEDIATE_TRACKING_PRIORITY", ActionRequired: "All available assets"},
	}

	log.Printf("[ALERT] Configured alert thresholds:")
	for _, step := range as.config.AlertEscalationSteps {
		log.Printf("  RRI > %.2f: %s - %s", step.RRIThreshold, step.AlertLevel, step.ActionRequired)
	}
}

func (as *AlertingService) runAlertMonitoring() {
	log.Printf("[ALERT] Starting alert monitoring cycle...")

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		as.monitorConjunctions()
	}
}

func (as *AlertingService) monitorConjunctions() {
	// In production, would query catalog for potential conjunctions
	// For demonstration, generate sample RRI calculations

	// Sample: High-risk conjunction with RRI > 1.0
	primaryState := physics.State{
		Pos: [3]float64{7000.1, 0, 0},
		Vel: [3]float64{0, 7.5, 0},
	}
	secondaryState := physics.State{
		Pos: [3]float64{7000.0, 0, 0},
		Vel: [3]float64{0, 8.0, 0},
	}
	timeToTCA := 600.0 // 10 minutes

	rri := as.rriCalculator.Compute(primaryState, secondaryState, timeToTCA)

	// Get probability bubble
	rx, ry, rz := as.rriCalculator.GetProbabilityBubble()

	if rri > as.config.RRIThreshold {
		alert := RRIAlert{
			Timestamp:         time.Now(),
			PrimaryObjectID:   12345,
			SecondaryObjectID: 67890,
			RRI:               rri,
			TimeToTCAMinutes:  timeToTCA / 60.0,
			ProbabilityBubble: ProbabilityBubble{RxKm: rx, RyKm: ry, RzKm: rz},
			DecisionWindow:    as.config.DecisionWindowMin,
		}

		// Determine alert level
		alert.AlertLevel, alert.ActionRequired = as.determineAlertLevel(rri)

		as.alertChannel <- alert
	}
}

func (as *AlertingService) determineAlertLevel(rri float64) (string, string) {
	for i := len(as.config.AlertEscalationSteps) - 1; i >= 0; i-- {
		step := as.config.AlertEscalationSteps[i]
		if rri >= step.RRIThreshold {
			return step.AlertLevel, step.ActionRequired
		}
	}
	return "ROUTINE_MONITORING", "None"
}

func (as *AlertingService) processAlerts() {
	for alert := range as.alertChannel {
		as.handleAlert(alert)
	}
}

func (as *AlertingService) handleAlert(alert RRIAlert) {
	// Check if within decision window
	if alert.TimeToTCAMinutes <= as.config.DecisionWindowMin {
		log.Printf("[ALERT] *** IMMEDIATE ACTION REQUIRED ***")
		log.Printf("  RRI: %.4f (Threshold: %.2f)", alert.RRI, as.config.RRIThreshold)
		log.Printf("  Time to TCA: %.1f minutes (Decision Window: %.1f minutes)",
			alert.TimeToTCAMinutes, alert.DecisionWindow)
		log.Printf("  Alert Level: %s", alert.AlertLevel)
		log.Printf("  Action Required: %s", alert.ActionRequired)
		log.Printf("  Probability Bubble: ±%.3f km (X), ±%.3f km (Y), ±%.3f km (Z)",
			alert.ProbabilityBubble.RxKm, alert.ProbabilityBubble.RyKm, alert.ProbabilityBubble.RzKm)
		log.Printf("  Primary ID: %d, Secondary ID: %d", alert.PrimaryObjectID, alert.SecondaryObjectID)
		log.Printf("======================================")

		// Send to commander dashboard
		as.sendToCommanderDashboard(alert)
	} else {
		log.Printf("[ALERT] Alert pending - TCA in %.1f minutes (outside decision window)",
			alert.TimeToTCAMinutes)
	}
}

func (as *AlertingService) sendToCommanderDashboard(alert RRIAlert) {
	// Placeholder for commander dashboard integration
	alertJSON, _ := json.MarshalIndent(alert, "", "  ")
	log.Printf("[ALERT] Alert sent to Commander Dashboard:\n%s", alertJSON)
}

// Utility function to calculate Pc (probability of collision)
func calculatePc(missDistanceKm, sigmaMissKm float64) float64 {
	normalizedMiss := missDistanceKm / (math.Sqrt(2) * sigmaMissKm)
	return 0.5 * math.Erfc(normalizedMiss)
}
