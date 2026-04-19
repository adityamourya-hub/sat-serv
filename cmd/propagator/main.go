// Command propagator provides SGP4 background propagation worker
// Ticks every 60 seconds to update satellite positions from TLEs
// IAF SDLC: Uses float64 throughout for orbital precision
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"ssis/internal/db/postgis"
	"ssis/internal/db/repository"
)

// SGP4 propagator constants - M2 optimized for 45k+ catalog
const (
	TickInterval      = 60 * time.Second
	PropagationAgeMax = 5 * time.Minute // Ignore states older than this
	BatchSize         = 5000            // Process max 5000 per tick (M2 can handle this)

	// Worker pool configuration for M2 (8 performance cores)
	Tier1Workers  = 4     // Sovereign assets (highest priority)
	Tier2Workers  = 8     // Global catalog
	Tier3Workers  = 4     // Historical/analytical
	WorkQueueSize = 10000 // Buffer for async processing
)

// TLEElements holds SGP4 orbital elements (all float64)
type TLEElements struct {
	MeanMotion   float64 // rev/day
	Eccentricity float64
	Inclination  float64 // radians
	RAAN         float64 // radians
	ArgPerigee   float64 // radians
	MeanAnomaly  float64 // radians
	BStar        float64 // 1/Earth radii
}

// DataTier represents the three-tier data management strategy
type DataTier int

const (
	Tier1Sovereign  DataTier = iota // Hot: IAF satellites (1-min cycles)
	Tier2Global                     // Warm: Full catalog (daily sync)
	Tier3Historical                 // Cold: Long-term trends
)

// TierState holds propagation work units with priority
type TierState struct {
	Tier     DataTier
	State    *repository.StateVector
	Priority int // Lower = higher priority
}

// PropagationResult holds the outcome of a single propagation
type PropagationResult struct {
	Tier       DataTier
	NoradCatID int32
	Position   [3]float64
	Velocity   [3]float64
	Success    bool
	Error      error
}

// WorkerPool manages concurrent propagation workers
type WorkerPool struct {
	tier1Queue   chan TierState
	tier2Queue   chan TierState
	tier3Queue   chan TierState
	results      chan PropagationResult
	wg           sync.WaitGroup
	tier1Workers int
	tier2Workers int
	tier3Workers int
}

// NewWorkerPool creates an M2-optimized worker pool
func NewWorkerPool(resultBuffer int) *WorkerPool {
	return &WorkerPool{
		tier1Queue:   make(chan TierState, WorkQueueSize),
		tier2Queue:   make(chan TierState, WorkQueueSize),
		tier3Queue:   make(chan TierState, WorkQueueSize),
		results:      make(chan PropagationResult, resultBuffer),
		tier1Workers: Tier1Workers,
		tier2Workers: Tier2Workers,
		tier3Workers: Tier3Workers,
	}
}

// Start launches all worker goroutines
func (wp *WorkerPool) Start(ctx context.Context) {
	// Tier 1 workers (Sovereign assets) - highest priority
	for i := 0; i < wp.tier1Workers; i++ {
		wp.wg.Add(1)
		go wp.tier1Worker(ctx, i)
	}

	// Tier 2 workers (Global catalog)
	for i := 0; i < wp.tier2Workers; i++ {
		wp.wg.Add(1)
		go wp.tier2Worker(ctx, i)
	}

	// Tier 3 workers (Historical/analytical)
	for i := 0; i < wp.tier3Workers; i++ {
		wp.wg.Add(1)
		go wp.tier3Worker(ctx, i)
	}
}

// Stop gracefully shuts down workers
func (wp *WorkerPool) Stop() {
	close(wp.tier1Queue)
	close(wp.tier2Queue)
	close(wp.tier3Queue)
	wp.wg.Wait()
	close(wp.results)
}

// tier1Worker processes sovereign assets with highest priority
func (wp *WorkerPool) tier1Worker(ctx context.Context, id int) {
	defer wp.wg.Done()
	for work := range wp.tier1Queue {
		result := wp.propagate(work, Tier1Sovereign)
		select {
		case wp.results <- result:
		case <-ctx.Done():
			return
		}
	}
}

// tier2Worker processes global catalog
func (wp *WorkerPool) tier2Worker(ctx context.Context, id int) {
	defer wp.wg.Done()
	for work := range wp.tier2Queue {
		result := wp.propagate(work, Tier2Global)
		select {
		case wp.results <- result:
		case <-ctx.Done():
			return
		}
	}
}

// tier3Worker processes historical data
func (wp *WorkerPool) tier3Worker(ctx context.Context, id int) {
	defer wp.wg.Done()
	for work := range wp.tier3Queue {
		result := wp.propagate(work, Tier3Historical)
		select {
		case wp.results <- result:
		case <-ctx.Done():
			return
		}
	}
}

// propagate performs SGP4 propagation
func (wp *WorkerPool) propagate(work TierState, tier DataTier) PropagationResult {
	sv := work.State

	// Convert orbital elements
	elements := TLEElements{
		MeanMotion:   sv.MeanMotion,
		Eccentricity: sv.Eccentricity,
		Inclination:  degreesToRadians(sv.Inclination),
		RAAN:         degreesToRadians(sv.RAAN),
		ArgPerigee:   degreesToRadians(sv.ArgOfPerigee),
		MeanAnomaly:  degreesToRadians(sv.MeanAnomaly),
		BStar:        sv.BStar,
	}

	// Calculate minutes since epoch
	now := time.Now().UTC()
	minutesSinceEpoch := now.Sub(sv.Epoch).Minutes()

	// SGP4 propagation
	pos, vel, err := sgp4Propagate(elements, minutesSinceEpoch)
	if err != nil {
		return PropagationResult{
			Tier:       tier,
			NoradCatID: sv.NoradCatID,
			Success:    false,
			Error:      err,
		}
	}

	// Convert ECI to LLA using GMST for proper Earth rotation
	gmst := calculateGMST(now)
	lon, lat, alt := eciToLLA(pos[0], pos[1], pos[2], gmst)

	return PropagationResult{
		Tier:       tier,
		NoradCatID: sv.NoradCatID,
		Position:   [3]float64{lon, lat, alt},
		Velocity:   vel,
		Success:    true,
	}
}

// Submit queues work by tier
func (wp *WorkerPool) Submit(state *repository.StateVector, tier DataTier, priority int) {
	work := TierState{Tier: tier, State: state, Priority: priority}
	switch tier {
	case Tier1Sovereign:
		wp.tier1Queue <- work
	case Tier2Global:
		wp.tier2Queue <- work
	case Tier3Historical:
		wp.tier3Queue <- work
	}
}

// Results returns the results channel
func (wp *WorkerPool) Results() <-chan PropagationResult {
	return wp.results
}

// PropagatorConfig holds propagation settings
type PropagatorConfig struct {
	TickInterval time.Duration
	BatchSize    int32
	DBPool       *pgxpool.Pool
	NumWorkers   int // M2: Use all performance cores
}

// Propagator manages the SGP4 propagation loop with worker pools
type Propagator struct {
	config       *PropagatorConfig
	statesRepo   *repository.CurrentStatesRepo
	metadataRepo *repository.AssetMetadataRepo
	ticker       *time.Ticker
	quit         chan struct{}
	workerPool   *WorkerPool
}

// NewPropagator creates propagator instance with M2-optimized worker pool
func NewPropagator(config *PropagatorConfig) *Propagator {
	return &Propagator{
		config:       config,
		statesRepo:   repository.NewCurrentStatesRepo(config.DBPool),
		metadataRepo: repository.NewAssetMetadataRepo(config.DBPool),
		ticker:       time.NewTicker(config.TickInterval),
		quit:         make(chan struct{}),
		workerPool:   NewWorkerPool(int(config.BatchSize)),
	}
}

// Start begins the propagation loop with tier-based priority
func (p *Propagator) Start(ctx context.Context) error {
	log.Printf("Propagator starting (tick: %v, batch: %d, workers: %d cores)",
		p.config.TickInterval, p.config.BatchSize, runtime.NumCPU())

	// Start worker pool
	p.workerPool.Start(ctx)

	// Start result processor
	go p.processResults(ctx)

	// Initial propagation
	if err := p.propagateBatch(ctx); err != nil {
		log.Printf("Initial propagation error: %v", err)
	}

	// Main loop
	for {
		select {
		case <-p.ticker.C:
			if err := p.propagateBatch(ctx); err != nil {
				log.Printf("Propagation error: %v", err)
			}
		case <-p.quit:
			p.workerPool.Stop()
			return nil
		case <-ctx.Done():
			p.workerPool.Stop()
			return ctx.Err()
		}
	}
}

// Stop gracefully shuts down the propagator
func (p *Propagator) Stop() {
	log.Println("Propagator shutting down...")
	p.ticker.Stop()
	close(p.quit)
}

// propagateBatch loads and propagates TLEs using tier-based worker pools
func (p *Propagator) propagateBatch(ctx context.Context) error {
	startTime := time.Now()

	// Load sovereign assets first (Tier 1) - guaranteed propagation
	sovereignStates, err := p.statesRepo.GetIndianAssetsForPropagation(ctx, time.Minute)
	if err != nil {
		log.Printf("Failed to load sovereign assets: %v", err)
	}

	// Load remaining states for propagation
	remainingBatch := p.config.BatchSize - int32(len(sovereignStates))
	if remainingBatch < 0 {
		remainingBatch = 0
	}

	var otherStates []*repository.StateVector
	if remainingBatch > 0 {
		otherStates, err = p.statesRepo.GetAllForPropagation(ctx, time.Minute, remainingBatch)
		if err != nil {
			return fmt.Errorf("load states: %w", err)
		}
	}

	totalStates := len(sovereignStates) + len(otherStates)
	if totalStates == 0 {
		return nil // Nothing to propagate
	}

	log.Printf("Propagating %d satellites (Tier 1: %d, Tier 2/3: %d)...",
		totalStates, len(sovereignStates), len(otherStates))

	// Queue Tier 1 (Sovereign) work first - highest priority
	for _, state := range sovereignStates {
		p.workerPool.Submit(state, Tier1Sovereign, 1)
	}

	// Queue Tier 2 and Tier 3 work
	for _, state := range otherStates {
		// Determine tier based on priority level from metadata
		tier := Tier2Global
		priority := 5

		// Check if we have metadata indicating tier
		if metadata, err := p.metadataRepo.GetByNoradID(ctx, state.NoradCatID); err == nil {
			if metadata.PriorityLevel <= 2 {
				tier = Tier2Global
				priority = int(metadata.PriorityLevel)
			} else {
				tier = Tier3Historical
				priority = int(metadata.PriorityLevel)
			}
		}

		p.workerPool.Submit(state, tier, priority)
	}

	duration := time.Since(startTime)
	log.Printf("Queued %d satellites for propagation in %v", totalStates, duration)

	return nil
}

// processResults handles propagation results and updates database
func (p *Propagator) processResults(ctx context.Context) {
	var tier1Count, tier2Count, tier3Count int
	var successCount, failCount int

	for result := range p.workerPool.Results() {
		switch result.Tier {
		case Tier1Sovereign:
			tier1Count++
		case Tier2Global:
			tier2Count++
		case Tier3Historical:
			tier3Count++
		}

		if !result.Success {
			failCount++
			log.Printf("Failed to propagate NORAD %d: %v", result.NoradCatID, result.Error)
			continue
		}

		successCount++

		// Update database with propagated position
		updated := &repository.StateVector{
			NoradCatID: result.NoradCatID,
			Position:   result.Position,
			Velocity:   result.Velocity,
		}

		if err := p.statesRepo.UpdateState(ctx, updated); err != nil {
			log.Printf("Failed to update NORAD %d: %v", result.NoradCatID, err)
			failCount++
		}
	}

	log.Printf("Propagation complete: Tier1=%d, Tier2=%d, Tier3=%d, Success=%d, Failed=%d",
		tier1Count, tier2Count, tier3Count, successCount, failCount)
}

// propagateState propagates a single satellite using SGP4
func (p *Propagator) propagateState(ctx context.Context, sv *repository.StateVector) error {
	// Convert orbital elements to SGP4 format (all float64)
	elements := TLEElements{
		MeanMotion:   sv.MeanMotion,
		Eccentricity: sv.Eccentricity,
		Inclination:  degreesToRadians(sv.Inclination),
		RAAN:         degreesToRadians(sv.RAAN),
		ArgPerigee:   degreesToRadians(sv.ArgOfPerigee),
		MeanAnomaly:  degreesToRadians(sv.MeanAnomaly),
		BStar:        sv.BStar,
	}

	// Calculate minutes since epoch
	now := time.Now().UTC()
	minutesSinceEpoch := now.Sub(sv.Epoch).Minutes()

	// SGP4 propagation to current time
	pos, vel, err := sgp4Propagate(elements, minutesSinceEpoch)
	if err != nil {
		return fmt.Errorf("sgp4 propagate: %w", err)
	}

	// Convert ECI to LLA using proper ECI -> ECEF -> Geodetic transformation
	// This accounts for Earth rotation using GMST (Greenwich Mean Sidereal Time)
	gmst := calculateGMST(now)
	lon, lat, alt := eciToLLA(pos[0], pos[1], pos[2], gmst)

	// Update state vector
	updated := &repository.StateVector{
		NoradCatID: sv.NoradCatID,
		Position:   [3]float64{lon, lat, alt},
		Velocity:   vel,
	}

	// Save to database
	if err := p.statesRepo.UpdateState(ctx, updated); err != nil {
		return fmt.Errorf("update state: %w", err)
	}

	return nil
}

// sgp4Propagate performs SGP4 propagation
// Returns position [km] and velocity [km/s] in ECI frame
func sgp4Propagate(elems TLEElements, minutesSinceEpoch float64) ([3]float64, [3]float64, error) {
	// SGP4 implementation (simplified for demonstration)
	// In production, use a proper SGP4 library or implement full model

	// This is a placeholder - real SGP4 has ~1000 lines of code
	// For now, return approximate circular orbit position

	// Mean motion to semi-major axis
	n := elems.MeanMotion * 2 * math.Pi / 1440.0 // rad/min
	a := math.Pow(398600.8/(n*n), 1.0/3.0)       // km (Earth mu = 398600.8 km³/s²)

	// Mean anomaly at current time
	M := elems.MeanAnomaly + n*minutesSinceEpoch

	// Eccentric anomaly (simplified - use Kepler's equation for real)
	E := M + elems.Eccentricity*math.Sin(M)

	// True anomaly
	nu := 2 * math.Atan2(math.Sqrt(1+elems.Eccentricity)*math.Sin(E/2),
		math.Sqrt(1-elems.Eccentricity)*math.Cos(E/2))

	// Distance
	r := a * (1 - elems.Eccentricity*math.Cos(E))

	// Position in orbital plane
	xOrbital := r * math.Cos(nu)
	yOrbital := r * math.Sin(nu)

	// Transform to ECI (simplified - ignores 3D rotation for demo)
	// Full implementation needs rotation by inclination, RAAN, and arg of perigee
	x := xOrbital*math.Cos(elems.RAAN) - yOrbital*math.Cos(elems.Inclination)*math.Sin(elems.RAAN)
	y := xOrbital*math.Sin(elems.RAAN) + yOrbital*math.Cos(elems.Inclination)*math.Cos(elems.RAAN)
	z := yOrbital * math.Sin(elems.Inclination)

	// Velocity (simplified circular orbit approximation)
	vCirc := math.Sqrt(398600.8 / r) // km/s
	vx := -vCirc * math.Sin(nu)
	vy := vCirc * math.Cos(nu)
	vz := 0.0

	return [3]float64{x, y, z}, [3]float64{vx, vy, vz}, nil
}

// eciToLLA converts ECI position to Latitude, Longitude, Altitude
// Performs ECI -> ECEF rotation using GMST, then ECEF -> LLA
func eciToLLA(x, y, z float64, gmst float64) (lon, lat, alt float64) {
	// Step 1: Rotate ECI to ECEF using GMST (Greenwich Mean Sidereal Time)
	// [x_ecef]   [ cos(θ)  sin(θ)  0 ] [x_eci]
	// [y_ecef] = [-sin(θ)  cos(θ)  0 ] [y_eci]
	// [z_ecef]   [   0       0     1 ] [z_eci]
	xECEF := x*math.Cos(gmst) + y*math.Sin(gmst)
	yECEF := -x*math.Sin(gmst) + y*math.Cos(gmst)
	zECEF := z

	// Step 2: Convert ECEF to Geodetic (LLA) using WGS84
	lon, lat, alt = ecefToGeodetic(xECEF, yECEF, zECEF)

	return lon, lat, alt
}

// ecefToGeodetic converts ECEF coordinates to Geodetic (lon, lat, alt)
// Uses WGS84 ellipsoid parameters
func ecefToGeodetic(x, y, z float64) (lon, lat, alt float64) {
	// WGS84 parameters
	const a = 6378.137            // Semi-major axis (km)
	const f = 1.0 / 298.257223563 // Flattening
	const e2 = 2*f - f*f          // First eccentricity squared

	// Calculate longitude
	lon = math.Atan2(y, x)

	// Calculate latitude and altitude iteratively
	p := math.Sqrt(x*x + y*y)
	theta := math.Atan2(z*a, p*(1.0-e2))

	lat = math.Atan2(z+e2*a*math.Sin(theta)*math.Sin(theta)*math.Sin(theta),
		p-e2*a*math.Cos(theta)*math.Cos(theta)*math.Cos(theta))

	// Altitude
	N := a / math.Sqrt(1.0-e2*math.Sin(lat)*math.Sin(lat))
	alt = p/math.Cos(lat) - N

	// Refine latitude calculation
	for i := 0; i < 3; i++ {
		lat = math.Atan2(z, p*(1.0-e2*N/(N+alt)))
		N = a / math.Sqrt(1.0-e2*math.Sin(lat)*math.Sin(lat))
		alt = p/math.Cos(lat) - N
	}

	// Convert to degrees
	lon = radiansToDegrees(lon)
	lat = radiansToDegrees(lat)

	return lon, lat, alt
}

// calculateGMST calculates Greenwich Mean Sidereal Time for ECI to ECEF rotation
// Reference: The Astronomical Almanac
func calculateGMST(t time.Time) float64 {
	// Julian Date
	jd := julianDate(t)

	// Julian centuries from J2000.0
	T := (jd - 2451545.0) / 36525.0

	// GMST at 0h UT in radians
	gmst0 := 4.894961212735792 + 230121.67555840657*T +
		0.000387933*T*T - T*T*T/38710000.0

	// Rate of change (Earth rotation)
	rate := 1.002737909350795 + 5.9006e-11*T - 5.9e-15*T*T

	// UT1 in fraction of day (approximate with UTC)
	ut1 := float64(t.Hour())/24.0 + float64(t.Minute())/1440.0 +
		float64(t.Second())/86400.0 + float64(t.Nanosecond())/86400.0e9

	// GMST in radians
	gmst := gmst0 + rate*ut1*2.0*math.Pi

	// Normalize to [0, 2π)
	gmst = math.Mod(gmst, 2.0*math.Pi)
	if gmst < 0 {
		gmst += 2.0 * math.Pi
	}

	return gmst
}

// julianDate converts time to Julian Date
func julianDate(t time.Time) float64 {
	// Algorithm from Meeus, Astronomical Algorithms
	y := t.Year()
	m := int(t.Month())
	d := t.Day()

	if m <= 2 {
		y--
		m += 12
	}

	A := int(float64(y) / 100.0)
	B := 2 - A + int(float64(A)/4.0)

	jd := float64(int(365.25*float64(y+4716))) +
		float64(int(30.6001*float64(m+1))) +
		float64(d) + float64(B) - 1524.5

	// Add time fraction
	jd += float64(t.Hour())/24.0 + float64(t.Minute())/1440.0 +
		float64(t.Second())/86400.0 + float64(t.Nanosecond())/86400.0e9

	return jd
}

// degreesToRadians converts degrees to radians (float64)
func degreesToRadians(deg float64) float64 {
	return deg * math.Pi / 180.0
}

// radiansToDegrees converts radians to degrees (float64)
func radiansToDegrees(rad float64) float64 {
	return rad * 180.0 / math.Pi
}

func main() {
	// Load configuration
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://isda_user@localhost:5432/isda?sslmode=require"
	}

	tickInterval := TickInterval
	if envInterval := os.Getenv("PROPAGATOR_TICK_INTERVAL"); envInterval != "" {
		if d, err := time.ParseDuration(envInterval); err == nil {
			tickInterval = d
		}
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

	// Create propagator
	config := &PropagatorConfig{
		TickInterval: tickInterval,
		BatchSize:    BatchSize,
		DBPool:       pool.Pool,
	}

	propagator := NewPropagator(config)

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		propagator.Stop()
	}()

	// Start propagation loop
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()

	log.Println("Propagator starting...")
	if err := propagator.Start(ctx); err != nil {
		log.Fatalf("Propagator error: %v", err)
	}
}
