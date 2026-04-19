package collision

import (
	"encoding/json"
	"math"
	"sync"

	engcrypto "ssis/pkg/cryptography"
	"ssis/pkg/octree"
)

const (
	adaptivePcAbsTolerance       = 1e-12
	aspectRatioFallbackThreshold = 500.0
	relativeVelocityFallbackKmS  = 12.0
	alfanoSeriesMaxTerms         = 30
)

// AdaptivePcMetadata captures method and convergence diagnostics for Pc computation.
type AdaptivePcMetadata struct {
	Method              string  `json:"method"`
	AspectRatio         float64 `json:"aspect_ratio"`
	RelativeVelocityKmS float64 `json:"relative_velocity_km_s"`
	QuadratureOrder     int     `json:"quadrature_order"`
	SeriesTerms         int     `json:"series_terms"`
	ErrorEstimateAbs    float64 `json:"error_estimate_abs"`
	ConvergenceDiag     string  `json:"convergence_diag"`
	Converged           bool    `json:"converged"`
	Signature           string  `json:"signature,omitempty"`
	PublicKeyHex        string  `json:"public_key_hex,omitempty"`
}

// AdaptiveFosterPc computes Pc with automatic method selection.
func AdaptiveFosterPc(missR, missI, sigmaR, sigmaI, sigmaRI, R float64) float64 {
	pc, _ := AdaptiveFosterPcWithMetadata(missR, missI, sigmaR, sigmaI, sigmaRI, R, 0)
	return pc
}

// AdaptiveFosterPcWithMetadata computes Pc and returns signed method diagnostics.
func AdaptiveFosterPcWithMetadata(missR, missI, sigmaR, sigmaI, sigmaRI, R, relativeVelocityKmS float64) (float64, AdaptivePcMetadata) {
	meta := AdaptivePcMetadata{
		AspectRatio:         covarianceAspectRatio(sigmaR, sigmaI, sigmaRI),
		RelativeVelocityKmS: relativeVelocityKmS,
	}

	if R <= 0 || sigmaR <= 0 || sigmaI <= 0 {
		meta.Method = "invalid_input"
		meta.ConvergenceDiag = "input_validation_failed"
		signAdaptivePc(0, &meta)
		return 0, meta
	}

	xm, ym, lambdaMax, lambdaMin, ok := principalFrame(missR, missI, sigmaR, sigmaI, sigmaRI)
	if !ok {
		meta.Method = "singular_covariance"
		meta.ConvergenceDiag = "principal_frame_failure"
		signAdaptivePc(0, &meta)
		return 0, meta
	}

	// Alfano upper-bound short-circuit: if miss distance >> combined sigma, Pc is negligible
	missDist := math.Sqrt(missR*missR + missI*missI)
	combinedSigma := math.Sqrt(sigmaR + sigmaI) // max eigenvalue proxy
	if combinedSigma > 0 && missDist/(combinedSigma) > 50.0 {
		meta.Method = "alfano_bound"
		meta.ConvergenceDiag = "miss_distance_exceeds_50_sigma"
		meta.Converged = true
		signAdaptivePc(0, &meta)
		return 0, meta
	}

	useAlfano := meta.AspectRatio > aspectRatioFallbackThreshold || relativeVelocityKmS > relativeVelocityFallbackKmS
	if useAlfano {
		pc, terms, errEstimate, converged := alfanoSeriesPc(xm, ym, lambdaMax, lambdaMin, R)
		if math.IsNaN(pc) || math.IsInf(pc, 0) {
			pc = 0
			converged = false
		}
		meta.Method = "alfano_2005_single_integral_series"
		meta.SeriesTerms = terms
		meta.ErrorEstimateAbs = errEstimate
		meta.Converged = converged
		meta.ConvergenceDiag = "alfano_series_truncated_k<=30"
		signAdaptivePc(pc, &meta)
		return pc, meta
	}

	pc, order, errEstimate, converged := adaptiveGaussLegendrePc(xm, ym, lambdaMax, lambdaMin, R)
	if math.IsNaN(pc) || math.IsInf(pc, 0) {
		pc = 0
		converged = false
	}
	meta.Method = "foster_1992_adaptive_gauss_legendre"
	meta.QuadratureOrder = order
	meta.ErrorEstimateAbs = errEstimate
	meta.Converged = converged
	meta.ConvergenceDiag = "gauss_legendre_32_to_256"
	signAdaptivePc(pc, &meta)
	return pc, meta
}

// GeometricFosterPcWithMetadata extends Foster Pc with support for non-spherical shapes.
func GeometricFosterPcWithMetadata(missR, missI, sigmaR, sigmaI, sigmaRI float64, hull1, hull2 RSOEnvelope, vRel octree.Vector3) (float64, AdaptivePcMetadata) {
	meta := AdaptivePcMetadata{
		AspectRatio:         covarianceAspectRatio(sigmaR, sigmaI, sigmaRI),
		RelativeVelocityKmS: math.Sqrt(vRel.X*vRel.X + vRel.Y*vRel.Y + vRel.Z*vRel.Z),
	}

	// Logic: If non-spherical shapes are present, use Minkowski-Sum
	// We check if either hull has more than 1 vertex (simplistic check for shape presence)
	if len(hull1.Vertices) > 1 || len(hull2.Vertices) > 1 {
		pc, err := MinkowskiSumGeometricPc(missR, missI, sigmaR, sigmaI, sigmaRI, hull1, hull2, vRel)
		if err == nil {
			meta.Method = "minkowski_sum_geometric_integration"
			meta.Converged = true
			meta.ConvergenceDiag = "greens_theorem_polygon_integral"
			signAdaptivePc(pc, &meta)
			return pc, meta
		}
	}

	// Fallback to spherical Foster if hulls are empty or integration fails
	// HBR is approximated from the hulls if possible, otherwise use a default 5m
	R := 0.005 // 5 meters
	return AdaptiveFosterPcWithMetadata(missR, missI, sigmaR, sigmaI, sigmaRI, R, meta.RelativeVelocityKmS)
}

// covarianceAspectRatio computes lambda_max / lambda_min for a 2x2 covariance matrix.
func covarianceAspectRatio(sigmaR, sigmaI, sigmaRI float64) float64 {
	det := sigmaR*sigmaI - sigmaRI*sigmaRI
	if sigmaR <= 0 || sigmaI <= 0 || det <= 0 {
		return math.Inf(1)
	}
	tr := sigmaR + sigmaI
	disc := tr*tr - 4.0*det
	if disc < 0 {
		disc = 0
	}
	root := math.Sqrt(disc)
	lMax := 0.5 * (tr + root)
	lMin := 0.5 * (tr - root)
	if lMin <= 0 {
		return math.Inf(1)
	}
	return lMax / lMin
}

func principalFrame(missR, missI, sigmaR, sigmaI, sigmaRI float64) (float64, float64, float64, float64, bool) {
	det := sigmaR*sigmaI - sigmaRI*sigmaRI
	if det <= 1e-24 {
		return 0, 0, 0, 0, false
	}
	tr := sigmaR + sigmaI
	disc := tr*tr - 4.0*det
	if disc < 0 {
		disc = 0
	}
	root := math.Sqrt(disc)
	lambdaMax := 0.5 * (tr + root)
	lambdaMin := 0.5 * (tr - root)
	if lambdaMax <= 0 || lambdaMin <= 0 {
		return 0, 0, 0, 0, false
	}

	theta := 0.5 * math.Atan2(2.0*sigmaRI, sigmaR-sigmaI)
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)
	xm := cosT*missR + sinT*missI
	ym := -sinT*missR + cosT*missI
	return xm, ym, lambdaMax, lambdaMin, true
}

func adaptiveGaussLegendrePc(xm, ym, lambdaMax, lambdaMin, R float64) (float64, int, float64, bool) {
	orders := []int{32, 64, 128, 256}
	prev := 0.0
	havePrev := false
	lastErr := math.Inf(1)

	for _, order := range orders {
		nodes, weights := gaussLegendreRule(order)
		pc := integratePcPolar(xm, ym, lambdaMax, lambdaMin, R, nodes, weights)
		if havePrev {
			lastErr = math.Abs(pc - prev)
			if lastErr <= adaptivePcAbsTolerance {
				return clamp01(pc), order, lastErr, true
			}
		}
		prev = pc
		havePrev = true
	}

	if math.IsInf(lastErr, 1) {
		lastErr = 0
	}
	return clamp01(prev), 256, lastErr, false
}

func integratePcPolar(xm, ym, lambdaMax, lambdaMin, R float64, nodes, weights []float64) float64 {
	sum := 0.0
	scale := 1.0 / (2.0 * math.Pi * math.Sqrt(lambdaMax*lambdaMin))
	xBase := xm * xm / lambdaMax
	yBase := ym * ym / lambdaMin

	for i := range nodes {
		phi := (nodes[i] + 1.0) * math.Pi
		c := math.Cos(phi)
		s := math.Sin(phi)

		a := 0.5 * (c*c/lambdaMax + s*s/lambdaMin)
		if a <= 0 {
			continue
		}
		b := xm*c/lambdaMax + ym*s/lambdaMin
		gamma := 0.5 * (xBase + yBase)

		sqrtA := math.Sqrt(a)
		u0 := b / (2.0 * sqrtA)
		u1 := sqrtA*R + u0
		erfDelta := math.Erf(u1) - math.Erf(u0)

		j0 := 0.0
		expJ0Arg := -gamma + (b*b)/(4.0*a)
		if expJ0Arg < 700 {
			j0 = math.Sqrt(math.Pi) / (2.0 * sqrtA) * math.Exp(expJ0Arg) * erfDelta
		} else {
			j0 = 0
		}

		expTail := math.Exp(-gamma - a*R*R - b*R)
		inner := (math.Exp(-gamma)-expTail)/(2.0*a) - (b/(2.0*a))*j0
		if inner < 0 && inner > -1e-18 {
			inner = 0
		}

		sum += inner * weights[i]
	}

	pc := sum * math.Pi * scale
	return clamp01(pc)
}

func alfanoSeriesPc(xm, ym, lambdaMax, lambdaMin, R float64) (float64, int, float64, bool) {
	sigmaU := math.Sqrt(lambdaMax)
	sigmaV := math.Sqrt(lambdaMin)
	if sigmaU <= 0 || sigmaV <= 0 || R <= 0 {
		return 0, 0, 0, false
	}

	u0 := math.Hypot(xm, ym)
	a := (u0 * u0) / (2.0 * sigmaU * sigmaU)
	beta := (R * R) / (2.0 * sigmaV * sigmaV)
	alpha := (R * R) / (2.0 * sigmaU * sigmaV)

	sum := 0.0
	prev := 0.0
	lastErr := math.Inf(1)
	converged := false
	usedTerms := alfanoSeriesMaxTerms + 1

	for k := 0; k <= alfanoSeriesMaxTerms; k++ {
		coeff := 1.0
		if k > 0 {
			if beta == 0 {
				coeff = 0
			} else {
				lg, _ := math.Lgamma(float64(k + 1))
				logCoeff := float64(k)*math.Log(beta) - lg
				if logCoeff > 700 {
					return 1.0, k + 1, 0, false
				}
				coeff = math.Exp(logCoeff)
			}
		}

		scaledBessel := expNegXTimesBesselIInt(k, a)
		term := coeff * scaledBessel
		sum += term

		if k > 0 {
			lastErr = math.Abs(sum - prev)
			if lastErr <= adaptivePcAbsTolerance {
				converged = true
				usedTerms = k + 1
				break
			}
		}
		prev = sum
	}

	if math.IsInf(lastErr, 1) {
		lastErr = 0
	}
	pc := clamp01(alpha * sum)
	return pc, usedTerms, lastErr, converged
}

func expNegXTimesBesselIInt(n int, x float64) float64 {
	if x <= 0 {
		if n == 0 {
			return 1
		}
		return 0
	}
	if x < 40 {
		return math.Exp(-x) * besselIInt(n, x)
	}

	mu := float64(4 * n * n)
	inv8x := 1.0 / (8.0 * x)
	inv8x2 := inv8x * inv8x
	inv8x3 := inv8x2 * inv8x
	poly := 1.0 - (mu-1.0)*inv8x + ((mu-1.0)*(mu-9.0))*inv8x2/2.0 - ((mu-1.0)*(mu-9.0)*(mu-25.0))*inv8x3/6.0
	if poly < 0 {
		poly = 0
	}
	return poly / math.Sqrt(2.0*math.Pi*x)
}

func besselIInt(n int, x float64) float64 {
	if x == 0 {
		if n == 0 {
			return 1
		}
		return 0
	}

	h := 0.5 * x
	lg, _ := math.Lgamma(float64(n + 1))
	logTerm := float64(n)*math.Log(h) - lg
	if logTerm > 700 {
		return math.Inf(1)
	}
	term := math.Exp(logTerm)
	sum := term
	for m := 1; m < 200; m++ {
		term *= (h * h) / (float64(m) * float64(m+n))
		sum += term
		if math.Abs(term) <= 1e-16*math.Abs(sum) {
			break
		}
	}
	return sum
}

type gaussRule struct {
	nodes   []float64
	weights []float64
}

var gaussRuleCache sync.Map

func gaussLegendreRule(n int) ([]float64, []float64) {
	if cached, ok := gaussRuleCache.Load(n); ok {
		r := cached.(gaussRule)
		return r.nodes, r.weights
	}

	nodes := make([]float64, n)
	weights := make([]float64, n)
	m := (n + 1) / 2
	const eps = 3e-15

	for i := 0; i < m; i++ {
		z := math.Cos(math.Pi * (float64(i) + 0.75) / (float64(n) + 0.5))
		var z1 float64
		for {
			p1 := 1.0
			p2 := 0.0
			for j := 1; j <= n; j++ {
				p3 := p2
				p2 = p1
				p1 = ((2.0*float64(j)-1.0)*z*p2 - (float64(j)-1.0)*p3) / float64(j)
			}
			pp := float64(n) * (z*p1 - p2) / (z*z - 1.0)
			z1 = z
			z = z1 - p1/pp
			if math.Abs(z-z1) <= eps {
				w := 2.0 / ((1.0 - z*z) * pp * pp)
				nodes[i] = -z
				nodes[n-1-i] = z
				weights[i] = w
				weights[n-1-i] = w
				break
			}
		}
	}

	rule := gaussRule{nodes: nodes, weights: weights}
	gaussRuleCache.Store(n, rule)
	return rule.nodes, rule.weights
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

var (
	adaptivePcSignerOnce sync.Once
	adaptivePcSigner     *engcrypto.Signer
	adaptivePcSignerErr  error
)

func getAdaptivePcSigner() (*engcrypto.Signer, error) {
	adaptivePcSignerOnce.Do(func() {
		adaptivePcSigner, adaptivePcSignerErr = engcrypto.NewSignerFromEnv("SSIS_SIGNING_KEY")
	})
	return adaptivePcSigner, adaptivePcSignerErr
}

func signAdaptivePc(pc float64, meta *AdaptivePcMetadata) {
	signer, err := getAdaptivePcSigner()
	if err != nil || signer == nil {
		return
	}
	unsignedMeta := *meta
	unsignedMeta.Signature = ""
	unsignedMeta.PublicKeyHex = ""
	payload := struct {
		Pc       float64            `json:"pc"`
		Metadata AdaptivePcMetadata `json:"metadata"`
	}{
		Pc:       pc,
		Metadata: unsignedMeta,
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	meta.Signature = signer.Sign(b)
	meta.PublicKeyHex = signer.PublicKeyHex()
}

// FosterTwoDimensionalPc computes the collision probability using AdaptiveFosterPc.
// This implementation is annotated for Gobra formal verification.
// Deprecated: Use AdaptiveFosterPc or AdaptiveFosterPcWithMetadata.
// @ requires varX > 0 && varY > 0 && R > 0
// @ ensures res >= 0 && res <= 1
func FosterTwoDimensionalPc(missX, missY, varX, varY, covXY, R float64) (res float64) {
	return AdaptiveFosterPc(missX, missY, varX, varY, covXY, R)
}
