package physics

/**
 * HypersonicState represents the 8-state vector for maneuvering targets.
 * [x, y, z, vx, vy, vz, beta, ld]
 */
type HypersonicState struct {
	Pos  Vector3
	Vel  Vector3
	Beta float64
	LD   float64
}

type ThreatClass string

const (
	ThreatUnknown    ThreatClass = "unknown"
	ThreatOrbital    ThreatClass = "orbital"
	ThreatBallistic  ThreatClass = "ballistic"
	ThreatHypersonic ThreatClass = "hypersonic"
)

type RapidTrack struct {
	ID                  int
	Mach                float64
	AltitudeKm          float64
	BurnoutVelocityKmS  float64
	FlightPathAngleRad  float64
	AzimuthRad          float64
	LaunchLat           float64
	LaunchLon           float64
	DistanceToAssetKm   float64
	RadarCrossSection   float64
	ManeuverIntentProxy bool
	ManeuverIntentScore float64
}

type RapidRankInference struct {
	Score           float64
	Confidence      float64
	UncertaintyLow  float64
	UncertaintyHigh float64
	FeatureWeights  map[string]float64
}

type RapidPolicyInference struct {
	Action     string
	Confidence float64
	Reason     string
}
