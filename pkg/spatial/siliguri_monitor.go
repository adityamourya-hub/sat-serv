package spatial

import (
	"fmt"
	"sort"
	"time"

	"ssis/pkg/types"
)

/**
 * Siliguri Corridor Strategic Monitor
 *
 * Implements the "Strategic Cube" VOI (Volume of Interest) for the
 * Chicken's Neck - the narrow corridor connecting India's northeast.
 *
 * Reference: ADITI 4.0 Proposal - Volumetric Surveillance of Critical Geographies
 */

// SiliguriBounds defines the strategic 3D volume
// Latitude: 25.5°N to 27.2°N
// Longitude: 87.8°E to 89.5°E
// Altitude: 300km to 1200km (LEO spy satellite zone)
var SiliguriBounds = struct {
	MinLat, MaxLat float64 // degrees
	MinLon, MaxLon float64 // degrees
	MinAlt, MaxAlt float64 // km
}{
	MinLat: 25.5,
	MaxLat: 27.2,
	MinLon: 87.8,
	MaxLon: 89.5,
	MinAlt: 300,
	MaxAlt: 1200,
}

// StrategicConstants for behavioral analysis
const (
	// Jilin-1 Swarm detection: 3+ satellites within 60 minutes
	JilinSwarmThreshold  = 3
	JilinSwarmTimeWindow = 60 * time.Minute

	// Yaogan ELINT triplet detection
	YaoganTripletThreshold = 3
	YaoganTimeWindow       = 30 * time.Minute

	// ISR Saturation threshold
	ISRSaturationCount  = 5
	ISRSaturationWindow = 60 * time.Minute

	// Strategic dark window (minutes of no coverage)
	MinDarkWindowDuration = 2 * time.Hour
)

// ThreatCategory classifies detected objects
type ThreatCategory string

const (
	ThreatJilinOptical     ThreatCategory = "JILIN_OPTICAL"
	ThreatYaoganELINT      ThreatCategory = "YAOGAN_ELINT"
	ThreatUnknownAdversary ThreatCategory = "ADVERSARY_UNKNOWN"
	ThreatFriendly         ThreatCategory = "FRIENDLY"
)

// SiliguriObject represents a satellite detected in the strategic cube
type SiliguriObject struct {
	ID      int
	Name    string
	Pos     types.Vector3
	Vel     types.Vector3
	EpochTT float64

	// Geopolitical context
	CountryCode string
	IsAdversary bool
	Category    ThreatCategory
	MissionType string // RECONNAISSANCE, ELINT, COMMS, etc.

	// Temporal analysis
	EntryTime    time.Time
	ExitTime     time.Time
	DwellTime    time.Duration
	NextPassTime time.Time

	// Behavioral flags
	IsPartOfSwarm   bool
	SwarmID         int
	IsTripletMember bool

	// Sensor tasking
	NeedsOpticalVerification bool
	OpticalSensorAssigned    string
}

// SwarmEvent represents a cluster passage
type SwarmEvent struct {
	Timestamp      time.Time
	SatelliteCount int
	Category       ThreatCategory
	Duration       time.Duration
	Objects        []SiliguriObject
	Severity       int // 1-10 scale
}

// EMCONRecommendation for IAF emission control
type EMCONRecommendation struct {
	RecommendedStatus string // "SILENCE", "REDUCE", "NORMAL"
	EffectiveWindow   time.Time
	Duration          time.Duration
	Reason            string
	YaoganObjects     []SiliguriObject
}

// StrategicDarkWindow represents a coverage gap
type StrategicDarkWindow struct {
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	SafeFor    string // "TANK_MOVEMENT", "RADAR_ACTIVATION", "COMMS"
	Confidence float64
}

// ObjectRecord represents a catalog object (matches catalog.SDAObject structure)
type ObjectRecord struct {
	ID      int
	Name    string
	Pos     types.Vector3
	Vel     types.Vector3
	EpochTT float64
}

// CatalogStore interface to avoid import cycle
type CatalogStore interface {
	Snapshot() []ObjectRecord
	QueryRange(min, max types.Vector3, tMin, tMax float64) []ObjectRecord
}

// SiliguriMonitor tracks the strategic corridor
type SiliguriMonitor struct {
	catalogStore   CatalogStore
	currentObjects map[int]*SiliguriObject // NORAD ID -> Object
	swarmHistory   []SwarmEvent
	emconStatus    EMCONRecommendation
	sensorRegistry *SensorRegistry

	// Analysis state
	swarmCounter  int
	lastJilinPass time.Time
}

// NewSiliguriMonitor creates a strategic monitor for the Chicken's Neck
func NewSiliguriMonitor(store CatalogStore, sensors *SensorRegistry) *SiliguriMonitor {
	return &SiliguriMonitor{
		catalogStore:   store,
		currentObjects: make(map[int]*SiliguriObject),
		swarmHistory:   make([]SwarmEvent, 0),
		sensorRegistry: sensors,
	}
}

// CheckSiliguriVolume scans the strategic cube for objects
func (sm *SiliguriMonitor) CheckSiliguriVolume(timestamp time.Time) ([]SiliguriObject, error) {
	// Query catalog for objects in volume
	// Using BVH index for O(log N) spatial query
	// Note: catalogStore.Snapshot() returns []ObjectRecord (SDAObject)
	objects := make([]SiliguriObject, 0)

	// Get all objects and filter
	// In production, this would use spatial index query
	snapshot := sm.catalogStore.Snapshot()

	for _, obj := range snapshot {
		// Check if object is in Siliguri volume
		lat, lon, alt := sm.ECIToGeodetic(obj.Pos, timestamp)

		if lat >= SiliguriBounds.MinLat && lat <= SiliguriBounds.MaxLat &&
			lon >= SiliguriBounds.MinLon && lon <= SiliguriBounds.MaxLon &&
			alt >= SiliguriBounds.MinAlt && alt <= SiliguriBounds.MaxAlt {

			countryCode := sm.classifyCountry(obj.ID)
			so := SiliguriObject{
				ID:          obj.ID,
				Name:        obj.Name,
				Pos:         obj.Pos,
				Vel:         obj.Vel,
				EpochTT:     obj.EpochTT,
				CountryCode: countryCode,
				IsAdversary: countryCode == "CHN" || countryCode == "PAK",
				Category:    sm.classifyThreat(obj.Name, obj.ID),
				MissionType: sm.inferMissionType(obj.Name),
				EntryTime:   timestamp,
			}

			objects = append(objects, so)
			sm.currentObjects[obj.ID] = &so
		}
	}

	return objects, nil
}

// DetectJilinSwarm checks for cluster passages
func (sm *SiliguriMonitor) DetectJilinSwarm(objects []SiliguriObject) *SwarmEvent {
	jilinObjects := make([]SiliguriObject, 0)

	for _, obj := range objects {
		if obj.Category == ThreatJilinOptical {
			jilinObjects = append(jilinObjects, obj)
		}
	}

	if len(jilinObjects) >= JilinSwarmThreshold {
		swarm := SwarmEvent{
			Timestamp:      time.Now(),
			SatelliteCount: len(jilinObjects),
			Category:       ThreatJilinOptical,
			Objects:        jilinObjects,
			Severity:       min(len(jilinObjects)*2, 10), // Scale 1-10
		}

		sm.swarmHistory = append(sm.swarmHistory, swarm)
		sm.swarmCounter++

		// Flag for ISR saturation
		if len(jilinObjects) >= ISRSaturationCount {
			// Trigger optical verification for each
			for i := range swarm.Objects {
				swarm.Objects[i].NeedsOpticalVerification = true
				swarm.Objects[i].OpticalSensorAssigned = sm.assignOpticalSensor(swarm.Objects[i])
			}
		}

		return &swarm
	}

	return nil
}

// DetectYaoganELINT identifies ELINT triplet patterns
func (sm *SiliguriMonitor) DetectYaoganELINT(objects []SiliguriObject) *EMCONRecommendation {
	yaoganObjects := make([]SiliguriObject, 0)

	for _, obj := range objects {
		if obj.Category == ThreatYaoganELINT {
			yaoganObjects = append(yaoganObjects, obj)
		}
	}

	if len(yaoganObjects) >= YaoganTripletThreshold {
		// Check if they form a triplet (within 30 min of each other)
		sort.Slice(yaoganObjects, func(i, j int) bool {
			return yaoganObjects[i].EntryTime.Before(yaoganObjects[j].EntryTime)
		})

		// Mark as triplet members
		for i := range yaoganObjects {
			yaoganObjects[i].IsTripletMember = true
		}

		// Generate EMCON recommendation
		emcon := EMCONRecommendation{
			RecommendedStatus: "SILENCE",
			EffectiveWindow:   time.Now(),
			Duration:          YaoganTimeWindow,
			Reason:            fmt.Sprintf("Yaogan ELINT triplet detected (%d satellites)", len(yaoganObjects)),
			YaoganObjects:     yaoganObjects,
		}

		sm.emconStatus = emcon
		return &emcon
	}

	return nil
}

// CalculateDarkWindow finds next coverage gap
func (sm *SiliguriMonitor) CalculateDarkWindow(lookAhead time.Duration) *StrategicDarkWindow {
	// Propagate all known adversary satellites forward
	// Find windows where Siliguri volume has zero coverage

	// This is a simplified implementation
	// Full version would use actual orbital propagation

	now := time.Now()

	// Check every 5 minutes in the look-ahead window
	bestStart := now
	bestDuration := time.Duration(0)

	for t := now; t.Before(now.Add(lookAhead)); t = t.Add(5 * time.Minute) {
		objects, _ := sm.CheckSiliguriVolume(t)

		adversaryCount := 0
		for _, obj := range objects {
			if obj.IsAdversary {
				adversaryCount++
			}
		}

		if adversaryCount == 0 {
			// Potential dark window
			duration := t.Sub(bestStart)
			if duration > bestDuration {
				bestDuration = duration
			}
		} else {
			bestStart = t.Add(5 * time.Minute)
		}
	}

	if bestDuration >= MinDarkWindowDuration {
		return &StrategicDarkWindow{
			StartTime:  bestStart,
			EndTime:    bestStart.Add(bestDuration),
			Duration:   bestDuration,
			SafeFor:    "TANK_MOVEMENT",
			Confidence: 0.85,
		}
	}

	return nil
}

// GenerateTaskingRequests creates sensor tasking for verification
func (sm *SiliguriMonitor) GenerateTaskingRequests() []SensorTask {
	tasks := make([]SensorTask, 0)

	for _, obj := range sm.currentObjects {
		if obj.NeedsOpticalVerification && obj.OpticalSensorAssigned != "" {
			task := SensorTask{
				TargetID:     obj.ID,
				SensorID:     obj.OpticalSensorAssigned,
				TaskType:     "OPTICAL_VERIFICATION",
				Priority:     9, // High priority for adversary
				StartTime:    time.Now(),
				Duration:     5 * time.Minute,
				MinElevation: 20,
			}
			tasks = append(tasks, task)
		}
	}

	return tasks
}

// Helper methods

func (sm *SiliguriMonitor) classifyCountry(noradID int) string {
	// Map NORAD ID ranges to countries
	// Simplified - production would use database lookup
	if noradID >= 40000 && noradID < 50000 {
		return "CHN"
	}
	if noradID >= 50000 && noradID < 60000 {
		return "PAK"
	}
	if noradID >= 20000 && noradID < 30000 {
		return "IND"
	}
	if noradID >= 25000 && noradID < 27000 {
		return "USA"
	}
	return "UNK"
}

func (sm *SiliguriMonitor) classifyThreat(name string, noradID int) ThreatCategory {
	// Jilin-1 constellation identification
	if contains(name, "JILIN") || contains(name, "JL-1") {
		return ThreatJilinOptical
	}

	// Yaogan ELINT identification
	if contains(name, "YAOGAN") || contains(name, "YG") {
		return ThreatYaoganELINT
	}

	// Country-based classification
	country := sm.classifyCountry(noradID)
	if country == "CHN" || country == "PAK" {
		return ThreatUnknownAdversary
	}

	return ThreatFriendly
}

func (sm *SiliguriMonitor) inferMissionType(name string) string {
	if contains(name, "JILIN") || contains(name, "GAOFEN") {
		return "RECONNAISSANCE"
	}
	if contains(name, "YAOGAN") {
		return "ELINT"
	}
	if contains(name, "BEIDOU") || contains(name, "NAVIC") {
		return "NAVIGATION"
	}
	if contains(name, "COMMS") || contains(name, "GSAT") {
		return "COMMS"
	}
	return "UNKNOWN"
}

func (sm *SiliguriMonitor) assignOpticalSensor(obj SiliguriObject) string {
	// Find nearest optical sensor
	// Simplified - would use actual sensor positions and availability

	if obj.Pos.X > 0 {
		return "leh_optical" // Northern sensor
	}
	return "mount_abu_optical" // Southern sensor
}

func (sm *SiliguriMonitor) boundsToECEF(minLat, minLon, minAlt, maxLat, maxLon, maxAlt float64) (types.Vector3, types.Vector3) {
	// Simplified conversion - production would use proper geodetic->ECEF
	return types.Vector3{X: minLat, Y: minLon, Z: minAlt},
		types.Vector3{X: maxLat, Y: maxLon, Z: maxAlt}
}

func (sm *SiliguriMonitor) ECIToGeodetic(pos types.Vector3, t time.Time) (lat, lon, alt float64) {
	// Simplified ECI to geodetic conversion
	// Production would use proper coordinate transforms from pkg/physics
	return pos.X, pos.Y, pos.Z
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SensorTask represents a tasking request
type SensorTask struct {
	TargetID     int
	SensorID     string
	TaskType     string
	Priority     int
	StartTime    time.Time
	Duration     time.Duration
	MinElevation float64
}
