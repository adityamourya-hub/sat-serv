// Command maneuver provides background maneuver detection service
// Listens for new TLE inserts, analyzes orbital history, and generates alerts
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"ssis/internal/analytics"
	"ssis/internal/db/postgis"
	"ssis/internal/db/repository"
)

// Config holds maneuver monitor configuration
type Config struct {
	CheckInterval   time.Duration // How often to check for new TLEs
	AnalysisWindow  int           // Number of historical records to analyze
	HistoryLookback time.Duration // How far back to look for new TLEs
	DBPool          *pgxpool.Pool
}

// DefaultConfig returns production defaults
func DefaultConfig() *Config {
	return &Config{
		CheckInterval:   5 * time.Minute,  // Check every 5 minutes
		AnalysisWindow:  20,               // Use last 20 historical records
		HistoryLookback: 10 * time.Minute, // Look back 10 minutes for new TLEs
	}
}

// Monitor coordinates maneuver detection
type Monitor struct {
	config   *Config
	alertRepo *repository.ManeuverAlertRepo
	quit     chan struct{}
}

// NewMonitor creates monitor instance
func NewMonitor(config *Config) *Monitor {
	return &Monitor{
		config:    config,
		alertRepo: repository.NewManeuverAlertRepo(config.DBPool),
		quit:      make(chan struct{}),
	}
}

// Start begins the monitoring loop
func (m *Monitor) Start(ctx context.Context) error {
	log.Printf("Maneuver Monitor starting (check interval: %v, window: %d)", 
		m.config.CheckInterval, m.config.AnalysisWindow)

	// Initial check
	if err := m.checkForManeuvers(ctx); err != nil {
		log.Printf("Initial check error: %v", err)
	}

	// Main loop
	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := m.checkForManeuvers(ctx); err != nil {
				log.Printf("Maneuver check error: %v", err)
			}
		case <-m.quit:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Stop gracefully shuts down the monitor
func (m *Monitor) Stop() {
	log.Println("Maneuver Monitor shutting down...")
	close(m.quit)
}

// checkForManeuvers analyzes recent TLE inserts for maneuvers
func (m *Monitor) checkForManeuvers(ctx context.Context) error {
	startTime := time.Now()

	// Get satellites with recent TLE updates
	noradIDs, err := m.alertRepo.GetRecentTLEInserts(ctx, m.config.HistoryLookback, 100)
	if err != nil {
		return err
	}

	if len(noradIDs) == 0 {
		return nil // Nothing to check
	}

	log.Printf("Analyzing %d satellites for maneuvers...", len(noradIDs))

	thresholds := analytics.DefaultThresholds()
	var totalAlerts int

	// Analyze each satellite
	for _, noradID := range noradIDs {
		alerts, err := m.analyzeSatellite(ctx, noradID, thresholds)
		if err != nil {
			log.Printf("Failed to analyze NORAD %d: %v", noradID, err)
			continue
		}
		totalAlerts += alerts
	}

	duration := time.Since(startTime)
	log.Printf("Maneuver check complete: %d alerts generated in %v", totalAlerts, duration)

	return nil
}

// analyzeSatellite checks a single satellite for orbital maneuvers
func (m *Monitor) analyzeSatellite(ctx context.Context, noradID int32, thresholds analytics.MetricThresholds) (int, error) {
	// Fetch orbital history
	samples, err := m.alertRepo.GetOrbitalHistory(ctx, noradID, m.config.AnalysisWindow)
	if err != nil {
		return 0, err
	}

	if len(samples) < 5 {
		return 0, nil // Not enough data
	}

	// Build orbital history struct
	history := &analytics.OrbitalHistory{
		NoradCatID: noradID,
	}

	for _, s := range samples {
		history.Inclinations = append(history.Inclinations, s.Inclination)
		history.MeanMotions = append(history.MeanMotions, s.MeanMotion)
		history.Eccentricities = append(history.Eccentricities, s.Eccentricity)
		history.RAANs = append(history.RAANs, s.RAAN)
		history.BStars = append(history.BStars, s.BStar)
	}

	// Get satellite name
	history.ObjectName, _ = m.alertRepo.GetSatelliteName(ctx, noradID)

	// Detect maneuvers
	reports := analytics.DetectManeuvers(history, thresholds)

	// Insert alerts
	for _, report := range reports {
		maneuverType := analytics.ClassifyManeuver(report)

		err := m.alertRepo.InsertManeuverAlert(ctx,
			noradID,
			history.ObjectName,
			report.Metric,
			report.CurrentValue,
			report.Mean,
			report.StdDev,
			report.ZScore,
			report.AlertLevel,
			maneuverType,
		)
		if err != nil {
			log.Printf("Failed to insert alert for NORAD %d: %v", noradID, err)
		} else {
			log.Printf("Maneuver detected: NORAD %d, %s, Z=%.2f, Level=%s, Type=%s",
				noradID, report.Metric, report.ZScore, report.AlertLevel, maneuverType)
		}
	}

	return len(reports), nil
}

func main() {
	// Load configuration
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://isda_user@localhost:5432/isda?sslmode=require"
	}

	// Initialize database pool
	pgConfig := postgis.DefaultConfig()
	pool, err := postgis.NewPool(pgConfig)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// Verify health
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := pool.HealthCheck(ctx); err != nil {
		log.Fatalf("Database health check failed: %v", err)
	}

	// Create monitor
	config := DefaultConfig()
	config.DBPool = pool.Pool

	monitor := NewMonitor(config)

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		monitor.Stop()
	}()

	// Start monitoring loop
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()

	log.Println("Maneuver Monitor starting...")
	if err := monitor.Start(ctx); err != nil {
		log.Fatalf("Monitor error: %v", err)
	}
}
