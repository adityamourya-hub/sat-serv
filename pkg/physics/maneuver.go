package physics

/**
 * ManeuverType defines the classification of orbital changes.
 */
type ManeuverType string

const (
	ManeuverNone       ManeuverType = "none"
	ManeuverImpulsive  ManeuverType = "impulsive"
	ManeuverLowThrust  ManeuverType = "low_thrust"
	ManeuverHypersonic ManeuverType = "hypersonic_like"
)

/**
 * ManeuverEvent represents a detected change in an orbital track.
 * Moved to physics for single source of truth (Expert Recommendation).
 */
type ManeuverEvent struct {
	ObjectID  int
	TimeIndex int
	Score     float64
	DeltaV    float64
	Type      ManeuverType

	// Characterization metrics
	DeltaVRSW        Vector3
	ManeuverTimeSec  float64
	IntentProxy      bool
	IntentProxyScore float64
	CoPlanarAlign    float64
	ProximityOps     bool
	StealthyManeuver bool
}

/**
 * ManeuverFrame is a model-ready temporal feature vector.
 */
type ManeuverFrame struct {
	Residual float64
	DeltaV   float64
}

/**
 * FeatureContribution represents the SHAP/Influence of a single feature.
 */
type FeatureContribution struct {
	Feature      string
	Value        float64
	Contribution float64
}

/**
 * ManeuverInference is the schema for ML model outputs.
 *
 * Added HamiltonianInvariant and HamiltonianHash for Sovereign Handshake protocol.
 * Enables spoof detection via Hamiltonian invariant validation (σ_H / H_avg >= 10^-7 self-invalidates).
 */
type ManeuverInference struct {
	Probability float64
	Type        ManeuverType
	DeltaV      float64
	Confidence  float64
	SHAP        []FeatureContribution

	ManeuverDeltaVRSW Vector3
	ManeuverTime      float64

	// Hamiltonian invariants for Sovereign Handshake
	HamiltonianInvariant float64 // Computed Hamiltonian value at maneuver time
	HamiltonianHash      string  // Hash of invariant for spoof detection (σ_H / H_avg threshold: 10^-7)
}

/**
 * ManeuverCharacterization is the detailed forensic output of a maneuver.
 */
type ManeuverCharacterization struct {
	ManeuverTimeSec  float64
	DeltaVECI        Vector3
	DeltaVRSW        Vector3
	BurnType         ManeuverType
	IntentProxy      bool
	IntentProxyScore float64
	CoPlanarAlign    float64
	ProximityOps     bool
	StealthyManeuver bool
	Confidence       float64
}

/**
 * ManeuverCharacterizationOutput is the forensic result of a maneuver detection pass.
 */
type ManeuverCharacterizationOutput struct {
	Detected            bool
	DetectionIndex      int
	ManeuverTimeSec     float64
	DeltaVMagnitudeKmS  float64
	DeltaVRSW           Vector3
	DeltaVECI           Vector3
	BurnType            ManeuverType
	IntentProxy         bool
	IntentProxyScore    float64
	CoPlanarAlign       float64
	ProximityOps        bool
	StealthyManeuver    bool
	Confidence          float64
	PriorFitResidualRMS float64
	PostFitResidualRMS  float64
}
