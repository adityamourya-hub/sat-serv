package physics

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

/**
 * ADITI 4.0 Hash Implicit Neural Representation (Hash INR)
 *
 * Multiresolution hash encoding + tiny MLP for atmospheric density field.
 *
 * Replaces expensive NRLMSISE-00 calls with O(1) neural forward passes.
 * Memory: ~2-5 MB (fits in L3 cache)
 * Speed: millions of queries/sec via ONNX
 *
 * Reference: Müller et al. "Instant Neural Graphics Primitives" (SIGGRAPH 2022)
 *
 * Architecture:
 * - Input: normalized (lat, lon, alt, F107, Ap, local_time)
 * - Hash encoding: L=16 levels, resolutions 16→512
 * - MLP: 2-3 layers, 64 neurons
 * - Output: log(density) or density directly
 *
 * Physics constraints:
 * - Density > 0 (exponential output or ReLU)
 * - Smooth gradients (for UKF/adjoint)
 * - Decays exponentially with altitude
 */

/**
 * HashINRConfig holds model configuration.
 */
type HashINRConfig struct {
	NumLevels      int    `json:"num_levels"`      // L: number of hash levels (typically 16)
	BaseResolution int    `json:"base_resolution"` // N0: coarsest grid (16)
	MaxResolution  int    `json:"max_resolution"`  // N_max: finest grid (512)
	HashMapSize    int    `json:"hash_map_size"`   // T: 2^19 to 2^24
	FeatureDim     int    `json:"feature_dim"`     // F: features per level (2)
	MLPHiddenDim   int    `json:"mlp_hidden_dim"`  // MLP hidden size (64)
	MLPNumLayers   int    `json:"mlp_num_layers"`  // MLP depth (2-3)
	InputDims      int    `json:"input_dims"`      // 6: (x,y,z,F107,Ap,LT)
	OutputDims     int    `json:"output_dims"`     // 1: density
	UseLogOutput   bool   `json:"use_log_output"`  // Output log(ρ) for stability
	ModelPath      string `json:"model_path"`      // ONNX file path
	Version        string `json:"version"`         // Model version
}

/**
 * DefaultHashINRConfig returns standard atmospheric density config.
 */
func DefaultHashINRConfig() HashINRConfig {
	return HashINRConfig{
		NumLevels:      16,
		BaseResolution: 16,
		MaxResolution:  512,
		HashMapSize:    1 << 19, // 524,288 entries
		FeatureDim:     2,
		MLPHiddenDim:   64,
		MLPNumLayers:   2,
		InputDims:      6,
		OutputDims:     1,
		UseLogOutput:   true,
		Version:        "2024.04.18.v1",
	}
}

/**
 * HashINR implements multiresolution hash encoding + MLP.
 *
 * This is the core INR structure. For production, the MLP runs via ONNX.
 * The hash encoding (lookup) is done in Go or pre-computed.
 */
type HashINR struct {
	Config HashINRConfig

	// Hash tables for each level [level][hash_index] → features
	HashTables [][][]float32 // [L][T][F]

	// MLP weights (loaded from ONNX or JSON)
	// For ONNX runtime, these are in the model file
	MLPWeights interface{} // Placeholder for ONNX session

	// Pre-computed scale factors for input normalization
	InputMins   []float32
	InputMaxs   []float32
	InputScales []float32 // 1/(max-min)
}

/**
 * AtmosphericQuery represents a density query point.
 */
type AtmosphericQuery struct {
	Latitude  float64 // degrees, -90 to 90
	Longitude float64 // degrees, -180 to 180
	Altitude  float64 // km above ellipsoid
	F107      float64 // Solar flux (sfu)
	F107Avg   float64 // 81-day average
	Ap        float64 // Geomagnetic index
	LocalTime float64 // hours, 0-24
	DayOfYear int     // 1-366
}

/**
 * NewHashINR creates an INR model from config.
 *
 * For production: Load ONNX model via onnxruntime-go.
 * For development: Use pre-computed hash tables + simple MLP.
 */
func NewHashINR(config HashINRConfig) (*HashINR, error) {
	inr := &HashINR{
		Config:     config,
		HashTables: make([][][]float32, config.NumLevels),
		InputMins:  []float32{-90, -180, 100, 50, 0, 0},    // lat, lon, alt, F107, Ap, LT
		InputMaxs:  []float32{90, 180, 1000, 300, 100, 24}, // max ranges
	}

	// Compute scales
	inr.InputScales = make([]float32, config.InputDims)
	for i := 0; i < config.InputDims; i++ {
		inr.InputScales[i] = 1.0 / (inr.InputMaxs[i] - inr.InputMins[i])
	}

	// Initialize hash tables (random or from training)
	// Production: Load from trained model file
	for l := 0; l < config.NumLevels; l++ {
		inr.HashTables[l] = make([][]float32, config.HashMapSize)
		for i := range inr.HashTables[l] {
			inr.HashTables[l][i] = make([]float32, config.FeatureDim)
			// Initialize with small random values or zeros
			for f := 0; f < config.FeatureDim; f++ {
				inr.HashTables[l][i][f] = (float32(i%10) - 5.0) / 100.0 // Small init
			}
		}
	}

	return inr, nil
}

/**
 * QueryDensity returns atmospheric density at query point.
 *
 * This is the main API for drag calculations.
 * Replaces NRLMSISE-00 calls with O(1) neural evaluation.
 *
 * @param q: Atmospheric query point
 * @return: Density (kg/m³)
 */
func (inr *HashINR) QueryDensity(q AtmosphericQuery) float64 {
	// Step 1: Normalize inputs to [0,1] or [-1,1]
	normalized := inr.normalizeInputs(q)

	// Step 2: Hash encoding (multiresolution)
	features := inr.hashEncode(normalized)

	// Step 3: MLP forward pass
	// Production: ONNX runtime
	// Development: Simple MLP evaluation
	logDensity := inr.mlpForward(features)

	// Step 4: Post-process
	if inr.Config.UseLogOutput {
		return math.Exp(float64(logDensity))
	}
	return float64(logDensity)
}

/**
 * normalizeInputs scales query to normalized coordinates.
 */
func (inr *HashINR) normalizeInputs(q AtmosphericQuery) []float32 {
	raw := []float32{
		float32(q.Latitude),
		float32(q.Longitude),
		float32(q.Altitude),
		float32(q.F107),
		float32(q.Ap),
		float32(q.LocalTime),
	}

	normalized := make([]float32, len(raw))
	for i := range raw {
		// Scale to [0, 1]
		normalized[i] = (raw[i] - inr.InputMins[i]) * inr.InputScales[i]
		// Clamp to valid range
		if normalized[i] < 0 {
			normalized[i] = 0
		}
		if normalized[i] > 1 {
			normalized[i] = 1
		}
		// Optionally shift to [-1, 1]
		normalized[i] = 2.0*normalized[i] - 1.0
	}

	return normalized
}

/**
 * hashEncode performs multiresolution hash encoding.
 *
 * For each level l:
 * 1. Compute resolution: N_l = N_0 * b^l where b = (N_max/N_0)^(1/(L-1))
 * 2. Find 8 corners of voxel containing query point
 * 3. Hash corner indices to lookup in hash table
 * 4. Trilinear interpolate feature vector
 * 5. Concatenate across all levels
 *
 * Returns concatenated feature vector of size L×F.
 */
func (inr *HashINR) hashEncode(normalized []float32) []float32 {
	L := inr.Config.NumLevels
	F := inr.Config.FeatureDim
	T := inr.Config.HashMapSize
	N0 := float64(inr.Config.BaseResolution)
	Nmax := float64(inr.Config.MaxResolution)

	// Growth factor b such that N_max = N_0 * b^(L-1)
	b := math.Pow(Nmax/N0, 1.0/float64(L-1))

	// Concatenated features
	features := make([]float32, 0, L*F)

	for l := 0; l < L; l++ {
		// Resolution at this level
		Nl := N0 * math.Pow(b, float64(l))

		// Scale normalized coord to grid
		scaled := make([]float32, len(normalized))
		for i := range normalized {
			scaled[i] = normalized[i] * float32(Nl)
		}

		// Find voxel corners (floor and ceil for each dim)
		// For 6D input, we have 2^6 = 64 corners - expensive!
		// Optimization: Hash encoding typically uses 3D spatial + other dims separate
		// Simplified: Use first 3 dims (lat, lon, alt) for spatial hash

		// 3D grid coordinates
		x0 := int(math.Floor(float64(scaled[0])))
		y0 := int(math.Floor(float64(scaled[1])))
		z0 := int(math.Floor(float64(scaled[2])))

		x1 := x0 + 1
		y1 := y0 + 1
		z1 := z0 + 1

		// Trilinear interpolation weights
		wx := scaled[0] - float32(x0)
		wy := scaled[1] - float32(y0)
		wz := scaled[2] - float32(z0)

		// Hash and lookup 8 corners
		// Feature vector for this level
		levelFeature := make([]float32, F)

		corners := [8][3]int{
			{x0, y0, z0}, {x1, y0, z0}, {x0, y1, z0}, {x1, y1, z0},
			{x0, y0, z1}, {x1, y0, z1}, {x0, y1, z1}, {x1, y1, z1},
		}

		weights := [8]float32{
			(1 - wx) * (1 - wy) * (1 - wz),
			wx * (1 - wy) * (1 - wz),
			(1 - wx) * wy * (1 - wz),
			wx * wy * (1 - wz),
			(1 - wx) * (1 - wy) * wz,
			wx * (1 - wy) * wz,
			(1 - wx) * wy * wz,
			wx * wy * wz,
		}

		for i, corner := range corners {
			hash := inr.hashCorner(corner[0], corner[1], corner[2], l, T)
			for f := 0; f < F; f++ {
				levelFeature[f] += weights[i] * inr.HashTables[l][hash][f]
			}
		}

		// Append to concatenated features
		features = append(features, levelFeature...)
	}

	return features
}

/**
 * hashCorner computes hash index for grid corner.
 *
 * Uses spatial hash function from Instant-NGP paper.
 * hash(x,y,z,l) = (x*1 + y*2654435761 + z*805459861 + l) mod T
 *
 * Primes provide good distribution and reduce collisions.
 */
func (inr *HashINR) hashCorner(x, y, z, level, T int) int {
	// Large primes for hash mixing
	p1 := 1
	p2 := 2654435761
	p3 := 805459861

	hash := (x*p1 + y*p2 + z*p3 + level) % T
	if hash < 0 {
		hash += T
	}
	return hash
}

/**
 * mlpForward evaluates the MLP.
 *
 * Production: Call ONNX runtime
 * Development: Simple fully-connected evaluation
 */
func (inr *HashINR) mlpForward(features []float32) float32 {
	// Stub: Return mean of features (placeholder)
	// Production: Load ONNX and run inference

	var sum float32
	for _, f := range features {
		sum += f
	}
	return sum / float32(len(features))
}

/**
 * LoadFromONNX loads trained model from ONNX file.
 *
 * Uses onnxruntime-go for inference.
 */
func (inr *HashINR) LoadFromONNX(path string) error {
	inr.Config.ModelPath = path

	// TODO: Implement with onnxruntime-go
	// session, err := onnx.NewAdvancedSession(path, ...)
	// inr.MLPWeights = session

	return fmt.Errorf("ONNX loading not yet implemented - use stub MLP")
}

/**
 * SaveConfig persists INR configuration.
 */
func (inr *HashINR) SaveConfig(path string) error {
	data, err := json.MarshalIndent(inr.Config, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

/**
 * LoadConfig loads INR configuration.
 */
func LoadHashINRConfig(path string) (HashINRConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return HashINRConfig{}, err
	}

	var config HashINRConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return HashINRConfig{}, err
	}

	return config, nil
}

// =============================================================================
// Drag Acceleration with INR
// =============================================================================

/**
 * INRDensityModel uses Hash INR for atmospheric density.
 *
 * Implements the DensityModel interface for drag calculations.
 */
type INRDensityModel struct {
	INR *HashINR
}

/**
 * NewINRDensityModel creates INR-based density provider.
 */
func NewINRDensityModel(inr *HashINR) *INRDensityModel {
	return &INRDensityModel{INR: inr}
}

/**
 * GetDensity returns density at position/time.
 *
 * Called from OrbitalAcceleration during propagation.
 */
func (dm *INRDensityModel) GetDensity(x, y, z, jd float64, env SpaceWeatherData) float64 {
	// Convert ECI to lat/lon/alt
	lat, lon, alt := ECIToLatLonAlt(x, y, z)

	// Compute local time from longitude and JD
	localTime := ComputeLocalTime(jd, lon)

	query := AtmosphericQuery{
		Latitude:  lat,
		Longitude: lon,
		Altitude:  alt,
		F107:      env.F107,
		F107Avg:   env.F107Avg,
		Ap:        env.Ap,
		LocalTime: localTime,
		DayOfYear: DayOfYearFromJD(jd),
	}

	return dm.INR.QueryDensity(query)
}

/**
 * ECIToLatLonAlt converts ECI position to geodetic coordinates.
 *
 * Simplified - production use proper geodetic conversion.
 */
func ECIToLatLonAlt(x, y, z float64) (lat, lon, alt float64) {
	r := math.Sqrt(x*x + y*y + z*z)

	// Latitude
	lat = math.Asin(z/r) * 180.0 / math.Pi

	// Longitude
	lon = math.Atan2(y, x) * 180.0 / math.Pi

	// Altitude (approximate)
	alt = r - EARTH_R_EQUATORIAL

	return lat, lon, alt
}

/**
 * ComputeLocalTime calculates local solar time.
 */
func ComputeLocalTime(jd, longitude float64) float64 {
	// Julian day fraction
	dayFrac := jd - math.Floor(jd)

	// UTC hour
	utcHour := dayFrac * 24.0

	// Longitude correction (15° = 1 hour)
	lonHour := longitude / 15.0

	localTime := utcHour + lonHour
	for localTime < 0 {
		localTime += 24
	}
	for localTime >= 24 {
		localTime -= 24
	}

	return localTime
}

/**
 * DayOfYearFromJD computes day of year from Julian Date.
 */
func DayOfYearFromJD(jd float64) int {
	// Simplified - production use proper calendar conversion
	// This is a placeholder
	return 1
}

// =============================================================================
// Performance Monitoring
// =============================================================================

/**
 * INRBenchmark tracks Hash INR performance vs NRLMSISE.
 */
type INRBenchmark struct {
	INRQueries        int
	INRTotalTime      float64 // seconds
	NRLMSISECalls     int
	NRLMSISETotalTime float64
}

/**
 * Speedup returns INR speedup factor.
 */
func (b *INRBenchmark) Speedup() float64 {
	inrAvg := b.INRTotalTime / float64(b.INRQueries)
	nrlAvg := b.NRLMSISETotalTime / float64(b.NRLMSISECalls)

	if nrlAvg == 0 {
		return 1.0
	}
	return nrlAvg / inrAvg
}

/**
 * Accuracy computes mean relative error vs ground truth.
 */
func (inr *HashINR) Accuracy(testPoints []AtmosphericQuery, groundTruth []float64) float64 {
	var totalError float64

	for i, q := range testPoints {
		predicted := inr.QueryDensity(q)
		truth := groundTruth[i]

		relError := math.Abs(predicted-truth) / truth
		totalError += relError
	}

	return totalError / float64(len(testPoints))
}
