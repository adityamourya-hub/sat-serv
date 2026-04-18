package spatial

import "math"

type BlindSpotSample struct {
	TimestampHour int64
	BlindRatio    float64
	OrbitalLoad   float64
	SensorLoad    float64
}

type BlindSpotForecastPoint struct {
	HourAhead         int
	PredictedRatio    float64
	UncertaintyLower  float64
	UncertaintyUpper  float64
	ModelContribution float64
}

type BlindSpotForecastResult struct {
	Points       []BlindSpotForecastPoint
	ModelName    string
	Confidence   float64
	UsedFallback bool
}

type BlindSpotForecastModel interface {
	Name() string
	Forecast(history []BlindSpotSample, horizonHours int) (BlindSpotForecastResult, error)
}

type HybridBlindSpotForecaster struct {
	Model              BlindSpotForecastModel
	MinModelConfidence float64
	ModelBlendWeight   float64
}

func NewHybridBlindSpotForecaster(model BlindSpotForecastModel) *HybridBlindSpotForecaster {
	return &HybridBlindSpotForecaster{
		Model:              model,
		MinModelConfidence: 0.55,
		ModelBlendWeight:   0.45,
	}
}

type CoverageIndex struct {
	latCells int
	lonCells int
	total    int
	covered  []uint64
}

func NewCoverageIndex(latCells, lonCells int) *CoverageIndex {
	if latCells <= 0 {
		latCells = 180
	}
	if lonCells <= 0 {
		lonCells = 360
	}
	total := latCells * lonCells
	return &CoverageIndex{
		latCells: latCells,
		lonCells: lonCells,
		total:    total,
		covered:  make([]uint64, (total+63)/64),
	}
}

func (c *CoverageIndex) CellID(latDeg, lonDeg float64) int {
	lat := clamp(latDeg, -90.0, 90.0)
	lon := normalizeLon(lonDeg)

	latIdx := int((lat + 90.0) / 180.0 * float64(c.latCells))
	lonIdx := int((lon + 180.0) / 360.0 * float64(c.lonCells))

	if latIdx >= c.latCells {
		latIdx = c.latCells - 1
	}
	if lonIdx >= c.lonCells {
		lonIdx = c.lonCells - 1
	}
	if latIdx < 0 {
		latIdx = 0
	}
	if lonIdx < 0 {
		lonIdx = 0
	}
	return latIdx*c.lonCells + lonIdx
}

func (c *CoverageIndex) MarkCircularCoverage(centerLatDeg, centerLonDeg, radiusDeg float64) {
	if radiusDeg <= 0 {
		c.mark(c.CellID(centerLatDeg, centerLonDeg))
		return
	}

	minLat := clamp(centerLatDeg-radiusDeg, -90.0, 90.0)
	maxLat := clamp(centerLatDeg+radiusDeg, -90.0, 90.0)
	minLon := centerLonDeg - radiusDeg
	maxLon := centerLonDeg + radiusDeg

	minLatIdx := int((minLat + 90.0) / 180.0 * float64(c.latCells))
	maxLatIdx := int((maxLat + 90.0) / 180.0 * float64(c.latCells))
	if maxLatIdx >= c.latCells {
		maxLatIdx = c.latCells - 1
	}
	if minLatIdx < 0 {
		minLatIdx = 0
	}

	for latIdx := minLatIdx; latIdx <= maxLatIdx; latIdx++ {
		cellLat := -90.0 + (float64(latIdx)+0.5)*(180.0/float64(c.latCells))

		minLonNorm := normalizeLon(minLon)
		maxLonNorm := normalizeLon(maxLon)
		if minLonNorm <= maxLonNorm {
			c.markLonRange(latIdx, cellLat, centerLatDeg, centerLonDeg, radiusDeg, minLonNorm, maxLonNorm)
			continue
		}

		c.markLonRange(latIdx, cellLat, centerLatDeg, centerLonDeg, radiusDeg, -180.0, maxLonNorm)
		c.markLonRange(latIdx, cellLat, centerLatDeg, centerLonDeg, radiusDeg, minLonNorm, 180.0)
	}
}

func (c *CoverageIndex) BlindSpotCount() int {
	covered := 0
	for _, w := range c.covered {
		covered += bitsCount(w)
	}
	if covered > c.total {
		covered = c.total
	}
	return c.total - covered
}

func (c *CoverageIndex) BlindSpotRatio() float64 {
	return float64(c.BlindSpotCount()) / float64(c.total)
}

func (c *CoverageIndex) Snapshot(hourUnix int64) BlindSpotSample {
	if hourUnix <= 0 {
		hourUnix = 1
	}
	return BlindSpotSample{
		TimestampHour: hourUnix / 3600,
		BlindRatio:    c.BlindSpotRatio(),
		OrbitalLoad:   0.5,
		SensorLoad:    0.5,
	}
}

func ForecastBlindSpots(history []BlindSpotSample, horizonHours int, forecaster *HybridBlindSpotForecaster) BlindSpotForecastResult {
	if horizonHours <= 0 {
		horizonHours = 24
	}
	if horizonHours > 72 {
		horizonHours = 72
	}

	baseline := prophetLikeForecast(history, horizonHours)
	if forecaster == nil || forecaster.Model == nil {
		baseline.UsedFallback = true
		return baseline
	}

	if forecaster.MinModelConfidence <= 0 {
		forecaster.MinModelConfidence = 0.55
	}
	if forecaster.ModelBlendWeight <= 0 || forecaster.ModelBlendWeight > 1 {
		forecaster.ModelBlendWeight = 0.45
	}

	ml, err := forecaster.Model.Forecast(history, horizonHours)
	if err != nil || ml.Confidence < forecaster.MinModelConfidence || len(ml.Points) == 0 {
		baseline.UsedFallback = true
		return baseline
	}

	blend := baseline
	blend.ModelName = ml.ModelName + "+prophet_hybrid"
	blend.UsedFallback = false
	blend.Confidence = clamp01(0.5*baseline.Confidence + 0.5*ml.Confidence)

	points := make([]BlindSpotForecastPoint, 0, len(blend.Points))
	for i := 0; i < len(blend.Points); i++ {
		b := blend.Points[i]
		if i >= len(ml.Points) {
			b.ModelContribution = 0
			points = append(points, b)
			continue
		}
		m := ml.Points[i]
		w := forecaster.ModelBlendWeight
		pred := clamp01((1.0-w)*b.PredictedRatio + w*m.PredictedRatio)
		lo := clamp01((1.0-w)*b.UncertaintyLower + w*m.UncertaintyLower)
		hi := clamp01((1.0-w)*b.UncertaintyUpper + w*m.UncertaintyUpper)
		if lo > hi {
			lo, hi = hi, lo
		}
		points = append(points, BlindSpotForecastPoint{
			HourAhead:         b.HourAhead,
			PredictedRatio:    pred,
			UncertaintyLower:  lo,
			UncertaintyUpper:  hi,
			ModelContribution: w,
		})
	}
	blend.Points = points
	return blend
}

func prophetLikeForecast(history []BlindSpotSample, horizonHours int) BlindSpotForecastResult {
	if horizonHours <= 0 {
		horizonHours = 24
	}

	defaultRatio := 1.0
	if len(history) > 0 {
		defaultRatio = clamp(history[len(history)-1].BlindRatio, 0, 1)
	}

	if len(history) < 3 {
		pts := make([]BlindSpotForecastPoint, 0, horizonHours)
		for h := 1; h <= horizonHours; h++ {
			pts = append(pts, BlindSpotForecastPoint{
				HourAhead:         h,
				PredictedRatio:    defaultRatio,
				UncertaintyLower:  clamp01(defaultRatio * 0.8),
				UncertaintyUpper:  clamp01(defaultRatio * 1.2),
				ModelContribution: 0,
			})
		}
		return BlindSpotForecastResult{Points: pts, ModelName: "prophet_like_fallback", Confidence: 0.45, UsedFallback: true}
	}

	first := history[0].BlindRatio
	last := history[len(history)-1].BlindRatio
	trendSlope := (last - first) / float64(len(history)-1)

	var mean float64
	for _, s := range history {
		mean += s.BlindRatio
	}
	mean /= float64(len(history))

	var variance float64
	for _, s := range history {
		d := s.BlindRatio - mean
		variance += d * d
	}
	variance /= float64(len(history))
	std := math.Sqrt(variance)

	pts := make([]BlindSpotForecastPoint, 0, horizonHours)
	baseHour := history[len(history)-1].TimestampHour
	if baseHour <= 0 {
		baseHour = int64(len(history))
	}

	for h := 1; h <= horizonHours; h++ {
		seasonPhase := float64((baseHour+int64(h))%24) / 24.0
		season := 0.03 * math.Sin(2.0*math.Pi*seasonPhase)
		pred := clamp01(last + trendSlope*float64(h) + season)
		spread := clamp(0.03+1.25*std, 0.02, 0.25)
		pts = append(pts, BlindSpotForecastPoint{
			HourAhead:         h,
			PredictedRatio:    pred,
			UncertaintyLower:  clamp01(pred - spread),
			UncertaintyUpper:  clamp01(pred + spread),
			ModelContribution: 0,
		})
	}

	conf := clamp01(0.55 + (1.0-math.Min(std/0.2, 1.0))*0.35)
	return BlindSpotForecastResult{Points: pts, ModelName: "prophet_like_fallback", Confidence: conf, UsedFallback: true}
}

func (c *CoverageIndex) markLonRange(latIdx int, cellLat, centerLatDeg, centerLonDeg, radiusDeg, minLon, maxLon float64) {
	minLonIdx := int((minLon + 180.0) / 360.0 * float64(c.lonCells))
	maxLonIdx := int((maxLon + 180.0) / 360.0 * float64(c.lonCells))
	if minLonIdx < 0 {
		minLonIdx = 0
	}
	if maxLonIdx >= c.lonCells {
		maxLonIdx = c.lonCells - 1
	}

	for lonIdx := minLonIdx; lonIdx <= maxLonIdx; lonIdx++ {
		cellLon := -180.0 + (float64(lonIdx)+0.5)*(360.0/float64(c.lonCells))
		if angularDistanceDeg(cellLat, cellLon, centerLatDeg, centerLonDeg) <= radiusDeg {
			c.mark(latIdx*c.lonCells + lonIdx)
		}
	}
}

func (c *CoverageIndex) mark(cellID int) {
	if cellID < 0 || cellID >= c.total {
		return
	}
	word := cellID / 64
	bit := uint(cellID % 64)
	c.covered[word] |= 1 << bit
}

func normalizeLon(lon float64) float64 {
	for lon > 180.0 {
		lon -= 360.0
	}
	for lon < -180.0 {
		lon += 360.0
	}
	return lon
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clamp01(v float64) float64 {
	return clamp(v, 0.0, 1.0)
}

func angularDistanceDeg(lat1, lon1, lat2, lon2 float64) float64 {
	lat1r := lat1 * math.Pi / 180.0
	lon1r := lon1 * math.Pi / 180.0
	lat2r := lat2 * math.Pi / 180.0
	lon2r := lon2 * math.Pi / 180.0

	dLat := lat2r - lat1r
	dLon := lon2r - lon1r
	a := math.Sin(dLat/2.0)*math.Sin(dLat/2.0) + math.Cos(lat1r)*math.Cos(lat2r)*math.Sin(dLon/2.0)*math.Sin(dLon/2.0)
	c := 2.0 * math.Atan2(math.Sqrt(a), math.Sqrt(1.0-a))
	return c * 180.0 / math.Pi
}

func bitsCount(v uint64) int {
	count := 0
	for v != 0 {
		v &= v - 1
		count++
	}
	return count
}
