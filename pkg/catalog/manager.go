package catalog

import (
	"math"
	"sort"
	"sync"
	"time"

	"ssis/pkg/physics"
)

/**
 * CatalogManager provides high-level management of the satellite catalog
 * for Live Catalog Updates.
 *
 * Responsibilities:
 * - Active catalog (latest TLE per object)
 * - TLE history (all TLEs for B* smoothing)
 * - Change detection (new, updates, decays, maneuvers)
 * - Integration with persistence layer
 */
type CatalogManager struct {
	mu sync.RWMutex

	// Active catalog: noradID -> latest SDAObject
	active map[int]SDAObject

	// TLE history: noradID -> chronological TLEs (oldest first)
	history map[int][]physics.TLE

	// Decayed objects: noradID -> decay date
	decayed map[int]time.Time

	// Space weather cache
	spaceWeather *physics.SpaceWeatherCache

	// B* smoother cache
	bstarCache *physics.BStarSmoothedCache

	// Persistence
	persist func(delta Delta)
}

/**
 * ChangeType categorizes the result of ingesting a TLE.
 */
type ChangeType int

const (
	ChangeNone ChangeType = iota
	ChangeNew
	ChangeUpdate
	ChangeDecay
	ChangeManeuver
)

/**
 * ChangeRecord describes a catalog change detected during ingestion.
 */
type ChangeRecord struct {
	SatNum         int
	Type           ChangeType
	OldTLE         *physics.TLE
	NewTLE         *physics.TLE
	BallisticCoeff float64 // Smoothed BC if available
	ManeuverDV     float64 // Delta-V if maneuver detected
	Timestamp      time.Time
}

/**
 * NewCatalogManager creates a manager for live catalog operations.
 */
func NewCatalogManager(swCache *physics.SpaceWeatherCache) *CatalogManager {
	return &CatalogManager{
		active:       make(map[int]SDAObject),
		history:      make(map[int][]physics.TLE),
		decayed:      make(map[int]time.Time),
		spaceWeather: swCache,
		bstarCache:   physics.NewBStarCache(),
	}
}

/**
 * SetPersistenceCallback configures a function to persist catalog changes.
 */
func (cm *CatalogManager) SetPersistenceCallback(fn func(delta Delta)) {
	cm.persist = fn
}

/**
 * IngestTLEs processes a batch of TLEs from Space-Track ingestion.
 *
 * @param tles: TLEs from incremental update
 * @return: List of detected changes
 */
func (cm *CatalogManager) IngestTLEs(tles []*physics.TLE) []ChangeRecord {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var changes []ChangeRecord
	var delta Delta

	for _, tle := range tles {
		if tle == nil || tle.SatelliteNumber == 0 {
			continue
		}

		change := cm.processTLE(tle)
		if change.Type != ChangeNone {
			changes = append(changes, change)

			// Build delta for persistence
			if obj, exists := cm.active[tle.SatelliteNumber]; exists {
				delta.Upserts = append(delta.Upserts, obj)
			}
		}
	}

	// Persist changes
	if cm.persist != nil && len(delta.Upserts) > 0 {
		cm.persist(delta)
	}

	return changes
}

/**
 * processTLE handles a single TLE and detects changes.
 */
func (cm *CatalogManager) processTLE(tle *physics.TLE) ChangeRecord {
	satNum := tle.SatelliteNumber
	now := time.Now()

	record := ChangeRecord{
		SatNum:    satNum,
		Timestamp: now,
	}

	// Check if this is a decayed object
	if decayTime, isDecayed := cm.decayed[satNum]; isDecayed {
		record.Type = ChangeDecay
		record.Timestamp = decayTime
		return record
	}

	// Check if we have an existing object
	existing, hasExisting := cm.active[satNum]

	if !hasExisting {
		// New object
		record.Type = ChangeNew
		record.NewTLE = tle
		cm.addNewObject(tle)
		return record
	}

	// Compare epochs
	existingJD := existing.EpochTT
	newJD := physicsTLEToJulianDate(tle)

	if newJD <= existingJD {
		// Older or same epoch, ignore (we already have newer)
		record.Type = ChangeNone
		return record
	}

	// This is an update - check for maneuver
	record.OldTLE = parseTLEFromStrings(existing.TLELine1, existing.TLELine2)
	record.NewTLE = tle
	record.Type = ChangeUpdate

	// Add to history (for B* smoothing)
	cm.addToHistory(tle)

	// Check for maneuver
	if cm.detectManeuver(satNum) {
		record.Type = ChangeManeuver
		// Calculate approximate delta-V
		if record.OldTLE != nil {
			record.ManeuverDV = estimateDeltaV(record.OldTLE, tle)
		}
	}

	// Update active catalog with smoothed BC
	bc := cm.updateBallisticCoeff(satNum)
	record.BallisticCoeff = bc

	// Update active entry
	cm.active[satNum] = SDAObject{
		ID:        satNum,
		TLELine1:  tle.RawL1,
		TLELine2:  tle.RawL2,
		EpochTT:   newJD,
		Timestamp: float64(now.Unix()),
	}

	return record
}

/**
 * addNewObject creates a new catalog entry.
 */
func (cm *CatalogManager) addNewObject(tle *physics.TLE) {
	// Initialize history
	cm.history[tle.SatelliteNumber] = []physics.TLE{*tle}

	// Create active entry
	jd := physicsTLEToJulianDate(tle)
	cm.active[tle.SatelliteNumber] = SDAObject{
		ID:        tle.SatelliteNumber,
		TLELine1:  tle.RawL1,
		TLELine2:  tle.RawL2,
		EpochTT:   jd,
		Timestamp: float64(time.Now().Unix()),
	}

	// Initial BC estimate from single TLE
	bc := physics.EstimateBallisticCoefficientFromTLE(tle.BStar)
	cm.bstarCache.Set(tle.SatelliteNumber, &physics.BStarSmoothed{
		SatelliteNumber: tle.SatelliteNumber,
		BallisticCoeff:  bc,
		BStarMean:       tle.BStar,
		BStarStd:        0,
		LastUpdated:     time.Now(),
		NumSamples:      1,
		Confidence:      0.3, // Low confidence with single sample
	})
}

/**
 * addToHistory appends a TLE to the object's history.
 */
func (cm *CatalogManager) addToHistory(tle *physics.TLE) {
	history := cm.history[tle.SatelliteNumber]
	history = append(history, *tle)

	// Sort by epoch
	sort.Slice(history, func(i, j int) bool {
		jd1 := physicsTLEToJulianDate(&history[i])
		jd2 := physicsTLEToJulianDate(&history[j])
		return jd1 < jd2
	})

	// Keep only last 90 days of history (prune old)
	cutoffJD := time.Now().Add(-90*24*time.Hour).Unix()/86400 + 2440587
	var pruned []physics.TLE
	for _, h := range history {
		if physicsTLEToJulianDate(&h) > float64(cutoffJD) {
			pruned = append(pruned, h)
		}
	}

	cm.history[tle.SatelliteNumber] = pruned
}

/**
 * detectManeuver checks if recent TLEs indicate a non-drag orbit change.
 */
func (cm *CatalogManager) detectManeuver(satNum int) bool {
	history := cm.history[satNum]
	if len(history) < 3 {
		return false
	}

	// Get TLE pointers for smoother
	var tlePtrs []*physics.TLE
	for i := range history {
		tlePtrs = append(tlePtrs, &history[i])
	}

	detected, _ := DetectManeuver(tlePtrs)
	return detected
}

/**
 * DetectManeuver analyzes TLE history to identify non-drag-related orbit changes.
 * Exported version for external use.
 */
func DetectManeuver(tleHistory []*physics.TLE) (detected bool, deltaV float64) {
	if len(tleHistory) < 3 {
		return false, 0
	}

	// Get semi-major axis trend from drag model
	smoother := physics.NewBStarSmoother()
	bs := smoother.SmoothenBStar(tleHistory)
	if bs == nil {
		return false, 0
	}

	// Expected decay from drag model (km/day)
	expectedDecay := math.Abs(bs.TrendSlope)

	// Actual change between first and last TLE
	first := tleHistory[0]
	last := tleHistory[len(tleHistory)-1]

	// Compute semi-major axes
	a1 := meanMotionToSemiMajorAxis(first.MeanMotion)
	a2 := meanMotionToSemiMajorAxis(last.MeanMotion)

	// Time span between TLEs (days)
	jd1 := physicsTLEToJulianDate(first)
	jd2 := physicsTLEToJulianDate(last)
	dt := jd2 - jd1

	if dt < 0.1 { // Less than ~2.4 hours
		return false, 0
	}

	// Actual change
	actualChange := math.Abs(a2 - a1)
	expectedChange := expectedDecay * dt

	// If actual change >> expected decay, suspect maneuver
	// Threshold: change exceeds 5x expected decay AND > 50m delta
	threshold := math.Max(5.0*expectedChange*dt, 0.05) // 50m minimum

	if actualChange > threshold {
		// Estimate delta-V (simplified vis-viva approximation)
		// delta-v ≈ sqrt(mu/a1) - sqrt(mu/a2) for circular orbits
		mu := physics.EARTH_MU_IAU
		dv := math.Abs(math.Sqrt(mu/a1) - math.Sqrt(mu/a2))
		return true, dv
	}

	return false, 0
}

/**
 * meanMotionToSemiMajorAxis converts mean motion (revs/day) to semi-major axis (km).
 */
func meanMotionToSemiMajorAxis(nRevsPerDay float64) float64 {
	nRadPerSec := nRevsPerDay * 2 * math.Pi / 86400.0
	a := math.Pow(physics.EARTH_MU_IAU/(nRadPerSec*nRadPerSec), 1.0/3.0)
	return a
}

/**
 * updateBallisticCoeff computes smoothed BC from TLE history.
 */
func (cm *CatalogManager) updateBallisticCoeff(satNum int) float64 {
	history := cm.history[satNum]
	if len(history) < 3 {
		// Not enough history, use single TLE estimate
		if len(history) > 0 {
			return physics.EstimateBallisticCoefficientFromTLE(history[len(history)-1].BStar)
		}
		return 50.0 // Nominal
	}

	// Get TLE pointers for smoother
	var tlePtrs []*physics.TLE
	for i := range history {
		tlePtrs = append(tlePtrs, &history[i])
	}

	smoother := physics.NewBStarSmoother()
	bs := smoother.SmoothenBStar(tlePtrs)

	if bs != nil {
		cm.bstarCache.Set(satNum, bs)
		return bs.BallisticCoeff
	}

	// Fallback
	return physics.EstimateBallisticCoefficientFromTLE(history[len(history)-1].BStar)
}

/**
 * MarkDecayed marks an object as decayed.
 */
func (cm *CatalogManager) MarkDecayed(satNum int, decayDate time.Time) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.decayed[satNum] = decayDate
	delete(cm.active, satNum)

	// Persist decay
	if cm.persist != nil {
		cm.persist(Delta{Deletes: []int{satNum}})
	}
}

/**
 * GetActiveCatalog returns the current active catalog.
 */
func (cm *CatalogManager) GetActiveCatalog() map[int]SDAObject {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy
	result := make(map[int]SDAObject, len(cm.active))
	for k, v := range cm.active {
		result[k] = v
	}
	return result
}

/**
 * GetHistory returns the TLE history for an object.
 */
func (cm *CatalogManager) GetHistory(satNum int) []physics.TLE {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return cm.history[satNum]
}

/**
 * GetBallisticCoeff returns the smoothed BC for an object.
 */
func (cm *CatalogManager) GetBallisticCoeff(satNum int) float64 {
	if bs := cm.bstarCache.Get(satNum); bs != nil {
		return bs.BallisticCoeff
	}
	return 50.0 // Nominal
}

/**
 * Helper functions
 */

func physicsTLEToJulianDate(tle *physics.TLE) float64 {
	return tleToJulianDate(tle.EpochYear, tle.EpochDay)
}

func tleToJulianDate(year int, day float64) float64 {
	var jdYear0 float64
	if year >= 2000 {
		jdYear0 = 2451545.0 + float64(year-2000)*365.25
	} else {
		jdYear0 = 2451545.0 - float64(2000-year)*365.25
	}
	return jdYear0 + day - 1.0
}

func parseTLEFromStrings(line1, line2 string) *physics.TLE {
	tle, _ := physics.ParseTLE(line1 + "\n" + line2)
	return tle
}

func estimateDeltaV(oldTLE, newTLE *physics.TLE) float64 {
	// Simplified vis-viva for circular orbits
	mu := physics.EARTH_MU_IAU

	// Convert mean motion to semi-major axis
	n1 := oldTLE.MeanMotion * 2 * math.Pi / 86400.0 // rad/s
	n2 := newTLE.MeanMotion * 2 * math.Pi / 86400.0

	a1 := math.Pow(mu/(n1*n1), 1.0/3.0)
	a2 := math.Pow(mu/(n2*n2), 1.0/3.0)

	// Circular orbit velocity
	v1 := math.Sqrt(mu / a1)
	v2 := math.Sqrt(mu / a2)

	return math.Abs(v2 - v1) // km/s
}
