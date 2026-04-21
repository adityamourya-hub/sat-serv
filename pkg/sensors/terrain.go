package sensors

/**
 * Terrain Modeling for SAR Blind Spot Detection
 *
 * Models digital elevation (DEM) for computing:
 * - SAR layover zones (radar beam hitting terrain)
 * - Terrain shadows (behind mountains)
 * - Line-of-sight occlusion
 *
 * References:
 * - Li et al. (2022): Shadow/layover segmentation for multi-angle SAR
 * - Strobl, Kropatsch: Geometric ray-tracing for SAR shadows
 */

import (
	"math"
)

// DEMDataSource defines interface for different DEM providers
type DEMDataSource interface {
	GetElevation(lat, lon float64) (float64, error) // Returns elevation in meters
	GetResolution() float64                         // Resolution in meters
	GetBounds() (minLat, maxLat, minLon, maxLon float64)
}

// SRTMDataSource implements SRTM (Shuttle Radar Topography Mission) DEM
// SRTM provides 90m resolution global coverage
type SRTMDataSource struct {
	dataPath   string               // Path to SRTM .hgt files directory
	cache      map[string][][]int16 // Cached DEM tiles
	resolution float64              // 90m for SRTM
}

// NewSRTMDataSource creates SRTM data source
func NewSRTMDataSource(dataPath string) *SRTMDataSource {
	return &SRTMDataSource{
		dataPath:   dataPath,
		cache:      make(map[string][][]int16),
		resolution: 90.0,
	}
}

// GetElevation loads SRTM tile and returns elevation
func (s *SRTMDataSource) GetElevation(lat, lon float64) (float64, error) {
	// SRTM tiles are 1x1 degree, named like N27E086.hgt
	latInt := int(math.Floor(lat))
	lonInt := int(math.Floor(lon))

	latDir := "N"
	if latInt < 0 {
		latDir = "S"
		latInt = -latInt
	}

	lonDir := "E"
	if lonInt < 0 {
		lonDir = "W"
		lonInt = -lonInt
	}

	tileName := string(latDir) + string(rune(latInt)) + string(lonDir) + string(rune(lonInt)) + ".hgt"

	// Check cache
	if tile, ok := s.cache[tileName]; ok {
		return s.interpolate(tile, lat, lon), nil
	}

	// Load tile from file (placeholder - would read actual .hgt file)
	// For now, return default elevation
	return 1000.0, nil
}

// interpolate performs bilinear interpolation on DEM tile
func (s *SRTMDataSource) interpolate(tile [][]int16, _lat, _lon float64) float64 {
	// Convert lat/lon to tile coordinates
	// latFrac := lat - math.Floor(lat)
	// lonFrac := lon - math.Floor(lon)

	// TODO: Implement proper bilinear interpolation
	// For now, return simple mean from tile center
	if len(tile) == 0 || len(tile[0]) == 0 {
		return 1000.0
	}
	return float64(tile[len(tile)/2][len(tile[0])/2])
}

func (s *SRTMDataSource) GetResolution() float64 {
	return s.resolution
}

func (s *SRTMDataSource) GetBounds() (minLat, maxLat, minLon, maxLon float64) {
	return -90.0, 90.0, -180.0, 180.0 // Global coverage
}

// ASTERDataSource implements ASTER GDEM (Advanced Spaceborne Thermal Emission and Reflection Radiometer)
// ASTER provides 30m resolution global coverage
type ASTERDataSource struct {
	dataPath   string
	cache      map[string][][]int16
	resolution float64 // 30m for ASTER
}

// NewASTERDataSource creates ASTER data source
func NewASTERDataSource(dataPath string) *ASTERDataSource {
	return &ASTERDataSource{
		dataPath:   dataPath,
		cache:      make(map[string][][]int16),
		resolution: 30.0,
	}
}

func (a *ASTERDataSource) GetElevation(lat, lon float64) (float64, error) {
	// Similar to SRTM but with 30m resolution
	return 1000.0, nil
}

func (a *ASTERDataSource) GetResolution() float64 {
	return a.resolution
}

func (a *ASTERDataSource) GetBounds() (minLat, maxLat, minLon, maxLon float64) {
	return -83.0, 83.0, -180.0, 180.0 // ASTER covers ±83° latitude
}

// TerrainDEM represents digital elevation model with pluggable data source
type TerrainDEM struct {
	DataSource DEMDataSource // DEM data provider
	MinLat     float64       // degrees
	MaxLat     float64       // degrees
	MinLon     float64       // degrees
	MaxLon     float64       // degrees
}

// NewTerrainDEM creates terrain model with SRTM data source
func NewTerrainDEM(minLat, maxLat, minLon, maxLon float64) *TerrainDEM {
	return NewTerrainDEMWithDataSource(minLat, maxLat, minLon, maxLon, nil)
}

// NewTerrainDEMWithDataSource creates terrain model with custom data source
func NewTerrainDEMWithDataSource(minLat, maxLat, minLon, maxLon float64, source DEMDataSource) *TerrainDEM {
	if source == nil {
		// Use simple fallback if no data source provided
		source = &SimpleDEM{MeanElevation: 1000.0, MaxElevation: 8000.0}
	}

	return &TerrainDEM{
		DataSource: source,
		MinLat:     minLat,
		MaxLat:     maxLat,
		MinLon:     minLon,
		MaxLon:     maxLon,
	}
}

// SimpleDEM is a fallback data source with constant elevation
type SimpleDEM struct {
	MeanElevation float64
	MaxElevation  float64
}

func (s *SimpleDEM) GetElevation(lat, lon float64) (float64, error) {
	return s.MeanElevation, nil
}

func (s *SimpleDEM) GetResolution() float64 {
	return 1000.0 // 1km resolution (coarse)
}

func (s *SimpleDEM) GetBounds() (minLat, maxLat, minLon, maxLon float64) {
	return -90.0, 90.0, -180.0, 180.0
}

// GetElevation returns terrain height at lat/lon
func (t *TerrainDEM) GetElevation(lat, lon float64) float64 {
	// Check bounds
	if lat < t.MinLat || lat > t.MaxLat || lon < t.MinLon || lon > t.MaxLon {
		return 0.0 // Ocean/unknown
	}

	elev, err := t.DataSource.GetElevation(lat, lon)
	if err != nil {
		// Fallback to 0 on error
		return 0.0
	}

	return elev
}

// GetMeanElevation returns average elevation over the region
func (t *TerrainDEM) GetMeanElevation() float64 {
	// Sample center of region
	centerLat := (t.MinLat + t.MaxLat) / 2.0
	centerLon := (t.MinLon + t.MaxLon) / 2.0
	return t.GetElevation(centerLat, centerLon)
}

// GetMaxElevation returns maximum elevation over the region
func (t *TerrainDEM) GetMaxElevation() float64 {
	// For now, sample corners and take max
	// In production, this would scan the entire DEM
	maxElev := 0.0
	corners := [][2]float64{
		{t.MinLat, t.MinLon},
		{t.MinLat, t.MaxLon},
		{t.MaxLat, t.MinLon},
		{t.MaxLat, t.MaxLon},
	}
	for _, corner := range corners {
		elev := t.GetElevation(corner[0], corner[1])
		if elev > maxElev {
			maxElev = elev
		}
	}
	return maxElev
}

// IsVisible checks if target is visible from sensor (line-of-sight)
func (t *TerrainDEM) IsVisible(
	sensorPos [3]float64, // ECI km
	targetPos [3]float64, // ECI km
) bool {
	// Convert to lat/lon for terrain check
	_, _, sensorAlt := ECIToLatLonAlt(sensorPos)
	_, _, _ = ECIToLatLonAlt(targetPos)

	// Simple check: sensor must be above terrain at its location
	terrainAtSensor := t.GetMeanElevation() / 1000.0 // km
	if sensorAlt < terrainAtSensor+0.1 {             // 100m clearance
		return false
	}

	// For space-based sensors, terrain is typically not blocking
	// But for mountainous regions, check line-of-sight
	if t.GetMaxElevation() > 3000.0 {
		// Check if line between sensor and target intersects terrain
		return !t.lineIntersectsTerrain(sensorPos, targetPos)
	}

	return true
}

// lineIntersectsTerrain checks if line segment hits terrain
func (t *TerrainDEM) lineIntersectsTerrain(p1, p2 [3]float64) bool {
	// Simplified ray-trace: check midpoint against mean elevation
	mid := [3]float64{
		(p1[0] + p2[0]) / 2,
		(p1[1] + p2[1]) / 2,
		(p1[2] + p2[2]) / 2,
	}

	_, _, midAlt := ECIToLatLonAlt(mid)
	terrainAlt := t.GetMeanElevation() / 1000.0

	return midAlt < terrainAlt
}

// ComputeSARShadow computes shadow zone for SAR beam
func (t *TerrainDEM) ComputeSARShadow(
	satPos [3]float64,
	beamAzimuth float64, // degrees from North
	beamElevation float64, // degrees from horizontal
	beamWidth float64, // degrees
) []SpatialRegion {

	// For a side-looking SAR, compute terrain shadow zone
	// Simplified: return region behind max elevation

	if t.GetMaxElevation() < 1000.0 {
		return nil // Flat terrain, no significant shadow
	}

	// Shadow extends from mountain peaks away from sensor
	shadowRegions := []SpatialRegion{
		{
			MinLat: t.MinLat,
			MaxLat: t.MaxLat,
			MinLon: t.MinLon,
			MaxLon: t.MaxLon,
			MinAlt: 0,
			MaxAlt: t.GetMaxElevation() / 1000.0, // km
		},
	}

	return shadowRegions
}

// ComputeLayoverZone identifies SAR layover regions
func (t *TerrainDEM) ComputeLayoverZone(
	satPos [3]float64,
	incidenceAngle float64, // from nadir
) SpatialRegion {

	// Layover: when terrain slope > incidence angle
	// Steep terrain facing sensor causes layover

	return SpatialRegion{
		MinLat: t.MinLat,
		MaxLat: t.MaxLat,
		MinLon: t.MinLon,
		MaxLon: t.MaxLon,
		MinAlt: t.GetMeanElevation() / 1000.0,
		MaxAlt: t.GetMaxElevation() / 1000.0,
	}
}
