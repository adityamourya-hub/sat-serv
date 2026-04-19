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

// TerrainDEM represents digital elevation model
type TerrainDEM struct {
	Resolution float64 // meters per cell
	MinLat     float64 // degrees
	MaxLat     float64 // degrees
	MinLon     float64 // degrees
	MaxLon     float64 // degrees
	// Simplified: use mean elevation + variance for now
	MeanElevation float64 // meters above WGS84
	MaxElevation  float64 // meters (peak)
}

// NewTerrainDEM creates terrain model for a region
func NewTerrainDEM(minLat, maxLat, minLon, maxLon float64) *TerrainDEM {
	return &TerrainDEM{
		Resolution:    90.0, // 90m SRTM resolution
		MinLat:        minLat,
		MaxLat:        maxLat,
		MinLon:        minLon,
		MaxLon:        maxLon,
		MeanElevation: 1000.0, // Default 1km mean (Himalayan-like)
		MaxElevation:  8000.0, // Default 8km max (Everest-like)
	}
}

// GetElevation returns terrain height at lat/lon (simplified)
func (t *TerrainDEM) GetElevation(lat, lon float64) float64 {
	// Check bounds
	if lat < t.MinLat || lat > t.MaxLat || lon < t.MinLon || lon > t.MaxLon {
		return 0.0 // Ocean/unknown
	}

	// Simplified terrain model: use mean elevation
	// In production, this would interpolate actual DEM data
	return t.MeanElevation
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
	terrainAtSensor := t.MeanElevation / 1000.0 // km
	if sensorAlt < terrainAtSensor+0.1 {        // 100m clearance
		return false
	}

	// For space-based sensors, terrain is typically not blocking
	// But for mountainous regions, check line-of-sight
	if t.MaxElevation > 3000.0 {
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
	terrainAlt := t.MeanElevation / 1000.0

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

	if t.MaxElevation < 1000.0 {
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
			MaxAlt: t.MaxElevation / 1000.0, // km
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
		MinAlt: t.MeanElevation / 1000.0,
		MaxAlt: t.MaxElevation / 1000.0,
	}
}
