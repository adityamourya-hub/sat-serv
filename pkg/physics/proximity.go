package physics

/**
 * CollisionRiskInput defines the parameters for a conjunction assessment.
 */
type CollisionRiskInput struct {
	MissDistanceKm      float64
	HardbodyRadiusKm    float64
	RelativeVelocityKmS float64
	TimeToTCASeconds    float64
	SigmaRadialKm       float64
	SigmaIntrackKm      float64
	SigmaCrosstrackKm   float64
	PrimaryID           int
	SecondaryID         int
}

// CollisionNeighborhoodGraph defines the graph structure for GNN-based collision detection.
type CollisionNeighborhoodGraph struct {
	Nodes []CollisionGraphNode
	Edges []CollisionGraphEdge
}

// CollisionGraphNode represents a single object in the collision graph.
type CollisionGraphNode struct {
	ObjectID   int
	Position   []float64
	Velocity   []float64
	Covariance [][]float64
	Features   []float64
}

// CollisionGraphEdge represents a potential conjunction between two objects.
type CollisionGraphEdge struct {
	SourceID     int
	TargetID     int
	MissDistance float64
	Probability  float64
	Weight       float64
}

// CollisionGraphInference contains the GNN model's collision risk prediction.
type CollisionGraphInference struct {
	GraphID             string
	RiskScore           float64
	ConjunctionPairs    []CollisionGraphEdge
	UncertaintyEstimate float64
	UncertaintyLower    float64
	UncertaintyUpper    float64
	Confidence          float64
	NodeAttention       map[int]float64
	Heatmap             []CollisionHeatmapCell
}

/**
 * CollisionHeatmapCell represents a single node's contribution to a collision risk estimate.
 */
type CollisionHeatmapCell struct {
	NodeID       int
	Attention    float64
	Contribution float64
}
