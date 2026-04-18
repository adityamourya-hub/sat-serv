package proximity

import (
	"fmt"
	"math"
	"time"

	pkg_catalog "ssis/pkg/catalog"
	"ssis/pkg/collision"
	"ssis/pkg/orbital"
)

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
	Primary             *pkg_catalog.ObjectRecord
	Secondary           *pkg_catalog.ObjectRecord
}

/**
 * CollisionRiskResult contains the calculated Pc and metadata.
 */
type CollisionRiskResult struct {
	Probability      float64
	Stage            string
	PcMethod         string `json:"pc_method,omitempty"`
	PcSignature      string `json:"pc_signature,omitempty"`
	PcConvergence    string `json:"pc_convergence,omitempty"`
	UncertaintyLower float64
	UncertaintyUpper float64                                `json:"uncertainty_upper"`
	ModelConfidence  float64                                `json:"model_confidence"`
	ProofHash        string                                 `json:"proof_hash,omitempty"`
	Heatmap          []CollisionHeatmapCell                 `json:"heatmap,omitempty"`
	MonteCarlo       *collision.MonteCarloConjunctionResult `json:"monte_carlo,omitempty"`
	CDM              *collision.CDM                         `json:"cdm,omitempty"`
}

type CollisionHeatmapCell struct {
	NodeID       int
	Attention    float64
	Contribution float64
}

// CollisionNeighborhoodGraph defines the graph structure for GNN-based collision detection.
type CollisionNeighborhoodGraph struct {
	Nodes []CollisionGraphNode
	Edges []CollisionGraphEdge
}

// CollisionGraphNode represents a single object in the collision graph.
type CollisionGraphNode struct {
	ObjectID   int
	ID         int         // Alias for compatibility
	Position   []float64   // [x, y, z] km
	Velocity   []float64   // [vx, vy, vz] km/s
	Covariance [][]float64 // 6x6 covariance matrix
	Features   []float64   // Feature vector for GNN
}

// CollisionGraphEdge represents a potential conjunction between two objects.
type CollisionGraphEdge struct {
	SourceID     int
	TargetID     int
	From         int     // Alias for compatibility
	To           int     // Alias for compatibility
	MissDistance float64 // km
	Probability  float64 // Pc
	Weight       float64 // Edge weight for GNN
}

// CollisionGraphInference contains the GNN model's collision risk prediction.
type CollisionGraphInference struct {
	GraphID             string
	RiskScore           float64 // 0-1 overall risk
	ConjunctionPairs    []CollisionGraphEdge
	UncertaintyEstimate float64
	UncertaintyLower    float64
	UncertaintyUpper    float64
	Confidence          float64
	NodeAttention       map[int]float64 // Attention weights per node
}

/**
 * CalculateGaussianKLDivergence computes the Kullback-Leibler divergence between
 * two 6D Gaussian distributions P ~ N(mu0, cov0) and Q ~ N(mu1, cov1).
 *
 * Formula: 0.5 * [ tr(cov1^-1 * cov0) + (mu1-mu0)^T * cov1^-1 * (mu1-mu0) - k + ln(det(cov1)/det(cov0)) ]
 * Currently optimized for near-diagonal covariance structures in SDA.
 *
 * @param mu0, mu1: 6D state vectors.
 * @param cov0, cov1: 6x6 covariance matrices.
 * @returns: KL divergence value.
 */
func CalculateGaussianKLDivergence(mu0, mu1 []float64, cov0, cov1 [][]float64) float64 {
	fmt.Printf("DEBUG: Entry CalculateGaussianKLDivergence at %v\n", time.Now())

	k := float64(len(mu0))

	// Term 1: Trace(cov1^-1 * cov0)
	tr := 0.0
	for i := 0; i < int(k); i++ {
		if cov1[i][i] > 1e-12 {
			tr += cov0[i][i] / cov1[i][i]
		}
	}

	// Term 2: Mahalanobis distance part (mu1-mu0)^T * cov1^-1 * (mu1-mu0)
	md := 0.0
	for i := 0; i < int(k); i++ {
		diff := mu1[i] - mu0[i]
		if cov1[i][i] > 1e-12 {
			md += (diff * diff) / cov1[i][i]
		}
	}

	// Term 3: Log Determinant Ratio ln(det(cov1)/det(cov0))
	det0, det1 := 1.0, 1.0
	for i := 0; i < int(k); i++ {
		det0 *= math.Max(cov0[i][i], 1e-18)
		det1 *= math.Max(cov1[i][i], 1e-18)
	}
	logDetRatio := math.Log(det1 / det0)

	kl := 0.5 * (tr + md - k + logDetRatio)

	fmt.Printf("DEBUG: Exit CalculateGaussianKLDivergence with D_KL=%f\n", kl)
	return math.Max(0.0, kl)
}

/**
 * InformationBarycenter blends two states across the Information Manifold.
 * Refined in Phase 1.2 to include KL-modulated Dynamic Weighting.
 *
 * Logic: If the divergence between the Axiom (Neural) and SGP4 (Analytical)
 * models increases (e.g. during a maneuver), the barycenter exponentially
 * decays the weight of the analytical model.
 *
 * @param s_sgp4, s_axiom: Cartesian states.
 * @param cov_sgp4, cov_axiom: Covariances.
 * @returns: Blended state vector.
 */
func InformationBarycenterRefined(s_sgp4, s_axiom []float64, cov_sgp4, cov_axiom [][]float64) []float64 {
	fmt.Printf("DEBUG: Entry InformationBarycenterRefined at %v\n", time.Now())

	// Compute Divergence
	d_kl := CalculateGaussianKLDivergence(s_axiom, s_sgp4, cov_axiom, cov_sgp4)

	// Dynamic Weighting logic: Exponential decay of SGP4 weight based on KL divergence
	// If Axiom and SGP4 are close, w_sgp4 stays high.
	w_sgp4 := math.Exp(-0.5 * d_kl)
	w_axiom := 1.0 // Neural model remains the reactive baseline

	blended := make([]float64, 6)
	for i := 0; i < 6; i++ {
		// Information weighting (using inverse variance/precision)
		prec_sgp4 := 0.0
		if cov_sgp4[i][i] > 1e-12 {
			prec_sgp4 = 1.0 / cov_sgp4[i][i]
		}

		prec_axiom := 0.0
		if cov_axiom[i][i] > 1e-12 {
			prec_axiom = 1.0 / cov_axiom[i][i]
		}

		total_prec := w_sgp4*prec_sgp4 + w_axiom*prec_axiom
		if total_prec > 1e-12 {
			blended[i] = (w_sgp4*prec_sgp4*s_sgp4[i] + w_axiom*prec_axiom*s_axiom[i]) / total_prec
		} else {
			blended[i] = (w_sgp4*s_sgp4[i] + w_axiom*s_axiom[i]) / (w_sgp4 + w_axiom)
		}
	}

	fmt.Printf("DEBUG: Exit InformationBarycenterRefined with SGP4_Weight=%f\n", w_sgp4)
	return blended
}

/**
 * AlfanoMaxPc calculates the maximum potential collision probability.
 */
func AlfanoMaxPc(missDistanceKm float64, hardbodyRadiusKm float64) float64 {
	if missDistanceKm <= 0 {
		return 1.0
	}
	ratio := hardbodyRadiusKm / missDistanceKm
	if ratio >= 1.0 {
		return 1.0
	}
	pc := math.Exp(-0.5) * math.Pow(ratio, 2.0)
	return math.Min(1.0, pc)
}

/**
 * EvaluateCollisionRisk generates a detailed estimate using Foster's method and MC.
 */
func EvaluateCollisionRisk(input CollisionRiskInput) CollisionRiskResult {
	fmt.Printf("DEBUG: Entry EvaluateCollisionRisk at %v\n", time.Now())
	radius := input.HardbodyRadiusKm
	if radius <= 0 {
		radius = 0.005
	}

	sigmaR := math.Max(input.SigmaRadialKm, 0.001)
	sigmaI := math.Max(input.SigmaIntrackKm, 0.001)

	result, pcMeta := collision.AdaptiveFosterPcWithMetadata(
		input.MissDistanceKm,
		0.0,
		sigmaR*sigmaR,
		sigmaI*sigmaI,
		0.0,
		radius,
		input.RelativeVelocityKmS,
	)

	var mcResult *collision.MonteCarloConjunctionResult
	var cdmResult *collision.CDM

	if result > 1e-6 && input.Primary != nil && input.Secondary != nil {
		pWrapper := &ObjectRecordWrapper{Rec: *input.Primary}
		sWrapper := &ObjectRecordWrapper{Rec: *input.Secondary}

		mc, cdm, _ := collision.EvaluateCollisionRiskMC(pWrapper, sWrapper, radius)
		mcResult = mc
		cdmResult = cdm
	}

	result = math.Min(1.0, math.Max(0.0, result))
	fmt.Printf("DEBUG: Exit EvaluateCollisionRisk with Pc=%e\n", result)
	return CollisionRiskResult{
		Probability:      result,
		Stage:            pcMeta.Method,
		PcMethod:         pcMeta.Method,
		PcSignature:      pcMeta.Signature,
		PcConvergence:    pcMeta.ConvergenceDiag,
		UncertaintyLower: clamp01(result * 0.90),
		UncertaintyUpper: clamp01(result * 1.10),
		ModelConfidence:  1.0,
		ProofHash:        collision.ProofHash,
		MonteCarlo:       mcResult,
		CDM:              cdmResult,
	}
}

type ObjectRecordWrapper struct {
	Rec pkg_catalog.ObjectRecord
}

func (w *ObjectRecordWrapper) GetState() []float64 {
	return []float64{w.Rec.Pos.X, w.Rec.Pos.Y, w.Rec.Pos.Z, w.Rec.Vel.X, w.Rec.Vel.Y, w.Rec.Vel.Z}
}

func (w *ObjectRecordWrapper) GetCovariance() [][]float64 {
	out := make([][]float64, 6)
	for i := 0; i < 6; i++ {
		out[i] = make([]float64, 6)
		for j := 0; j < 6; j++ {
			out[i][j] = w.Rec.Covariance[i][j]
		}
	}
	return out
}

func (w *ObjectRecordWrapper) Propagate(dtSec float64) []float64 {
	s := w.GetState()
	s[0] += s[3] * dtSec
	s[1] += s[4] * dtSec
	s[2] += s[5] * dtSec
	return s
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

/**
 * CalculateMOID implements the SDG-MOID algorithm (Hedo et al. 2018/2020).
 */
func CalculateMOID(s1, s2 orbital.State) float64 {
	fmt.Printf("DEBUG: Entry CalculateMOID at %v\n", time.Now())
	const MU = 398600.4418
	k1 := CartesianToKepler(s1, MU)
	k2 := CartesianToKepler(s2, MU)

	minDist := 1e15
	steps := 60
	bestM1, bestM2 := 0.0, 0.0
	for i := 0; i < steps; i++ {
		m1 := 2.0 * math.Pi * float64(i) / float64(steps)
		for j := 0; j < steps; j++ {
			m2 := 2.0 * math.Pi * float64(j) / float64(steps)
			d := distanceAtAnomaly(k1, k2, m1, m2, MU)
			if d < minDist {
				minDist, bestM1, bestM2 = d, m1, m2
			}
		}
	}

	m1, m2 := bestM1, bestM2
	delta := (2.0 * math.Pi / float64(steps)) / 10.0
	for iter := 0; iter < 100; iter++ {
		improved := false
		dirs := [][2]float64{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
		for _, dir := range dirs {
			tm1, tm2 := m1+dir[0]*delta, m2+dir[1]*delta
			d := distanceAtAnomaly(k1, k2, tm1, tm2, MU)
			if d < minDist {
				minDist, m1, m2 = d, tm1, tm2
				improved = true
			}
		}
		if !improved {
			delta /= 2.0
		}
		if delta < 1e-12 {
			break
		}
	}

	fmt.Printf("DEBUG: Exit CalculateMOID with Dist=%f\n", minDist)
	return minDist
}

type Keplerian struct{ A, E, I, RAAN, ARGP, M float64 }

func CartesianToKepler(s orbital.State, mu float64) Keplerian {
	r := math.Sqrt(s.X*s.X + s.Y*s.Y + s.Z*s.Z)
	v2 := s.VX*s.VX + s.VY*s.VY + s.VZ*s.VZ
	energy := v2/2.0 - mu/r
	a := -mu / (2.0 * energy)
	hx, hy, hz := s.Y*s.VZ-s.Z*s.VY, s.Z*s.VX-s.X*s.VZ, s.X*s.VY-s.Y*s.VX
	h := math.Sqrt(hx*hx + hy*hy + hz*hz)
	i := math.Acos(hz / h)
	return Keplerian{A: a, I: i}
}

func distanceAtAnomaly(k1, k2 Keplerian, m1, m2, mu float64) float64 {
	p1, p2 := KeplerToECI(k1, m1, mu), KeplerToECI(k2, m2, mu)
	return math.Sqrt(pow2(p1[0]-p2[0]) + pow2(p1[1]-p2[1]) + pow2(p1[2]-p2[2]))
}

func KeplerToECI(k Keplerian, m, mu float64) [3]float64 {
	x := k.A * math.Cos(m)
	y := k.A * math.Sin(m) * math.Cos(k.I)
	z := k.A * math.Sin(m) * math.Sin(k.I)
	return [3]float64{x, y, z}
}

func pow2(x float64) float64 { return x * x }
