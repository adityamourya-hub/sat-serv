package spatial

import (
	"math"
	"time"

	"github.com/golang/geo/s2"
	"ssis/pkg/types"
)

/**
 * Hypersonic Launch Tripwires (Challenge 37)
 * 
 * S2 Cell "Gate-Keepers" for Mach 5+ detection. Monitors atmospheric exit points
 * at Siliguri Gate and other strategic corridors using O(1) S2 containment checks.
 * 
 * Reference: Steve Brunton Multi-Resolution DMD for spatiotemporal mode analysis
 */

// Speed thresholds for atmospheric flight regimes
const (
	Mach1           = 340.0    // m/s at sea level
	Mach5Threshold  = 1700.0   // Hypersonic entry (Mach 5)
	Mach10Threshold = 3400.0   // High-hypersonic (Mach 10)
)

// TripwireLocation defines strategic monitoring points
type TripwireLocation struct {
	Name        string
	CellID      s2.CellID
	Center      s2.LatLng
	RadiusKm    float64  // S2 cell level determines precision
	Description string
}

// Strategic tripwires for IAF defense
var (
	// Siliguri Gate - Northern approach to Chicken's Neck
	SiliguriGate = TripwireLocation{
		Name:        "Siliguri_Gate",
		CellID:      s2.CellFromLatLng(s2.LatLngFromDegrees(27.0, 88.0)).ID(),
		Center:      s2.LatLngFromDegrees(27.0, 88.0),
		RadiusKm:    100.0,
		Description: "Eastern Himalayas atmospheric exit corridor",
}
	
	// Rajasthan Border - Western approach
	RajasthanGate = TripwireLocation{
		Name:        "Rajasthan_Gate",
		CellID:      s2.CellFromLatLng(s2.LatLngFromDegrees(28.5, 72.0)).ID(),
		Center:      s2.LatLngFromDegrees(28.5, 72.0),
		RadiusKm:    150.0,
		Description: "Thar desert hypersonic trajectory entry",
}
	
	// Tibetan Plateau - Northern threat axis
	TibetanGate = TripwireLocation{
		Name:        "Tibetan_Gate",
		CellID:      s2.CellFromLatLng(s2.LatLngFromDegrees(30.0, 85.0)).ID(),
		Center:      s2.LatLngFromDegrees(30.0, 85.0),
		RadiusKm:    200.0,
		Description: "Plateau launch detection for HGV trajectories",
}
)

// TargetState represents a tracked object at a time instant
type TargetState struct {
	ID        int
	Timestamp time.Time
	Pos       types.Vector3  // ECI coordinates (km)
	Vel       types.Vector3  // ECI velocity (km/s)
	LatLng    s2.LatLng      // S2 cell lookup
	CellID    s2.CellID
	Altitude  float64        // km
}

// HypersonicTripwire monitors S2 cells for Mach 5+ velocity breaches
type HypersonicTripwire struct {
	GateCell          s2.CellID
	VelocityThreshold float64  // m/s (Mach 5 default)
	CellLevel         int      // S2 cell level (smaller = larger area)
	
	// Breach detection state
	sequenceBuffer []TargetState  // Recent track history
	breachHistory  []BreachEvent
}

// BreachEvent records a hypersonic gate crossing
type BreachEvent struct {
	Timestamp    time.Time
	TargetID     int
	VelocityMs   float64  // Actual velocity in m/s
	MachNumber   float64
	EntryPoint   types.Vector3
	Trajectory   []TargetState
	Confidence   float64  // DMD mode confidence
}

// NewHypersonicTripwire creates a Mach 5+ detection gate
func NewHypersonicTripwire(gateCell s2.CellID, cellLevel int) *HypersonicTripwire {
	return &HypersonicTripwire{
		GateCell:          gateCell,
		VelocityThreshold: Mach5Threshold,
		CellLevel:         cellLevel,
		sequenceBuffer:    make([]TargetState, 0, 100),
		breachHistory:     make([]BreachEvent, 0),
	}
}

// IsBreached checks if target sequence crosses gate at Mach 5+
func (t *HypersonicTripwire) IsBreached(sequence []TargetState) (*BreachEvent, bool) {
	if len(sequence) < 2 {
		return nil, false
	}
	
	// Check if latest position is in gate cell (O(1) S2 containment)
	latest := sequence[len(sequence)-1]
	if !t.GateCell.ContainsPoint(latest.CellID.Point()) {
		return nil, false
	}
	
	// Calculate instantaneous velocity across sequence
	v := t.calculateVelocity(sequence)
	
	// Check hypersonic threshold
	if v > t.VelocityThreshold {
		mach := v / Mach1
		event := BreachEvent{
			Timestamp:  time.Now(),
			TargetID:   latest.ID,
			VelocityMs: v,
			MachNumber: mach,
			EntryPoint: latest.Pos,
			Trajectory: sequence,
			Confidence: 0.85, // Will be updated by mrDMD
		}
		t.breachHistory = append(t.breachHistory, event)
		return &event, true
	}
	
	return nil, false
}

// calculateVelocity computes speed from position sequence
func (t *HypersonicTripwire) calculateVelocity(sequence []TargetState) float64 {
	if len(sequence) < 2 {
		return 0.0
	}
	
	// Use last two points for instantaneous velocity
	p1 := sequence[len(sequence)-2]
	p2 := sequence[len(sequence)-1]
	
	dt := p2.Timestamp.Sub(p1.Timestamp).Seconds()
	if dt == 0 {
		return 0.0
	}
	
	// ECI distance (km)
	dx := p2.Pos.X - p1.Pos.X
	dy := p2.Pos.Y - p1.Pos.Y
	dz := p2.Pos.Z - p1.Pos.Z
	distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
	
	// Convert to m/s
	velocityKms := distance / dt
	return velocityKms * 1000.0  // km/s → m/s
}

// TripwireNetwork manages multiple strategic gates
type TripwireNetwork struct {
	tripwires map[string]*HypersonicTripwire
	active    bool
}

// NewTripwireNetwork creates a distributed tripwire system
func NewTripwireNetwork() *TripwireNetwork {
	tn := &TripwireNetwork{
		tripwires: make(map[string]*HypersonicTripwire),
		active:    true,
	}
	
	// Initialize strategic gates
	tn.AddTripwire("Siliguri", NewHypersonicTripwire(SiliguriGate.CellID, 12))
	tn.AddTripwire("Rajasthan", NewHypersonicTripwire(RajasthanGate.CellID, 12))
	tn.AddTripwire("Tibetan", NewHypersonicTripwire(TibetanGate.CellID, 11))
	
	return tn
}

// AddTripwire registers a monitoring gate
func (tn *TripwireNetwork) AddTripwire(name string, tripwire *HypersonicTripwire) {
	tn.tripwires[name] = tripwire
}

// MonitorSequence checks all gates against a target track
func (tn *TripwireNetwork) MonitorSequence(sequence []TargetState) ([]BreachEvent, bool) {
	breaches := make([]BreachEvent, 0)
	
	for name, tripwire := range tn.tripwires {
		if event, breached := tripwire.IsBreached(sequence); breached {
			breaches = append(breaches, *event)
			_ = name // Could log which gate was breached
		}
	}
	
	return breaches, len(breaches) > 0
}

// GetActiveGates returns currently monitored tripwires
func (tn *TripwireNetwork) GetActiveGates() map[string]*TripwireLocation {
	gates := make(map[string]*TripwireLocation)
	gates["Siliguri"] = &SiliguriGate
	gates["Rajasthan"] = &RajasthanGate
	gates["Tibetan"] = &TibetanGate
	return gates
}

// ContainsPoint checks if any gate contains the given S2 cell
func (tn *TripwireNetwork) ContainsPoint(cellID s2.CellID) (string, bool) {
	for name, tripwire := range tn.tripwires {
		if tripwire.GateCell.ContainsPoint(cellID.Point()) {
			return name, true
		}
	}
	return "", false
}
