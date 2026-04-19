// Package sigint provides FlightRadar24 integration for ADS-B correlation
// Implements sovereign caching for air-gap resilience
package sigint

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Aircraft represents a tracked aircraft
type Aircraft struct {
	ICAO24         string  `json:"icao24"`
	Callsign       string  `json:"callsign"`
	OriginCountry  string  `json:"origin_country"`
	TimePosition   float64 `json:"time_position"`
	LastContact    float64 `json:"last_contact"`
	Longitude      float64 `json:"longitude"`
	Latitude       float64 `json:"latitude"`
	BaroAltitude   float64 `json:"baro_altitude"`
	OnGround       bool    `json:"on_ground"`
	Velocity       float64 `json:"velocity"`
	TrueTrack      float64 `json:"true_track"`
	VerticalRate   float64 `json:"vertical_rate"`
	Sensors        []int   `json:"sensors"`
	GeoAltitude    float64 `json:"geo_altitude"`
	Squawk         string  `json:"squawk"`
	Spi            bool    `json:"spi"`
	PositionSource int     `json:"position_source"`
}

// FR24Bridge provides FlightRadar24 API integration with local caching
type FR24Bridge struct {
	apiKey    string
	apiBase   string
	cacheDB   *sql.DB
	cachePath string
	client    *http.Client
}

// NewFR24Bridge creates a new FlightRadar24 bridge
func NewFR24Bridge(apiKey string, cachePath string) (*FR24Bridge, error) {
	// Ensure cache directory exists
	if err := os.MkdirAll(cachePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	dbPath := filepath.Join(cachePath, "fr24_cache.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open cache database: %w", err)
	}

	bridge := &FR24Bridge{
		apiKey:    apiKey,
		apiBase:   "https://fr24api.com/v1",
		cacheDB:   db,
		cachePath: cachePath,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	// Initialize database schema
	if err := bridge.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return bridge, nil
}

// initSchema creates the cache database schema
func (fb *FR24Bridge) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS aircraft_cache (
		icao24 TEXT PRIMARY KEY,
		callsign TEXT,
		origin_country TEXT,
		latitude REAL,
		longitude REAL,
		baro_altitude REAL,
		velocity REAL,
		true_track REAL,
		vertical_rate REAL,
		squawk TEXT,
		last_updated TIMESTAMP,
		cache_ttl TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_position ON aircraft_cache(latitude, longitude);
	CREATE INDEX IF NOT EXISTS idx_time ON aircraft_cache(last_updated);
	CREATE INDEX IF NOT EXISTS idx_ttl ON aircraft_cache(cache_ttl);

	-- Create a view for active (non-expired) aircraft
	CREATE VIEW IF NOT EXISTS active_aircraft AS
	SELECT * FROM aircraft_cache WHERE cache_ttl > datetime('now');
	`

	_, err := fb.cacheDB.Exec(schema)
	return err
}

// GetADSBAround fetches aircraft around a geographic point
// Implements PACE model: Primary (API), Alternate (Cache)
func (fb *FR24Bridge) GetADSBAround(lat, lon float64, radiusKm int) ([]Aircraft, error) {
	// 1. Check local cache first (Air-gap resilience)
	cached, err := fb.getFromCache(lat, lon, radiusKm)
	if err == nil && len(cached) > 0 {
		// Check if cache is fresh (< 15 minutes old)
		if fb.isCacheFresh(cached) {
			return cached, nil
		}
	}

	// 2. Fetch from REST API (Primary)
	aircraft, err := fb.fetchFromAPI(lat, lon, radiusKm)
	if err != nil {
		// API failed, use cache even if stale (Contingency)
		if len(cached) > 0 {
			return cached, nil
		}
		return nil, fmt.Errorf("API failed and no cache available: %w", err)
	}

	// 3. Update cache with new data
	if err := fb.updateCache(aircraft); err != nil {
		// Log but don't fail - we have the data
		fmt.Printf("Warning: failed to update cache: %v\n", err)
	}

	return aircraft, nil
}

// fetchFromAPI fetches data from FlightRadar24 API
func (fb *FR24Bridge) fetchFromAPI(lat, lon float64, radiusKm int) ([]Aircraft, error) {
	// Construct bounding box (simplified radius calculation)
	// 1 degree latitude ≈ 111 km
	latDelta := float64(radiusKm) / 111.0
	// 1 degree longitude varies by latitude
	lonDelta := float64(radiusKm) / (111.0 * cosDeg(lat))

	minLat := lat - latDelta
	maxLat := lat + latDelta
	minLon := lon - lonDelta
	maxLon := lon + lonDelta

	url := fmt.Sprintf("%s/bounds/%f,%f,%f,%f",
		fb.apiBase, minLat, minLon, maxLat, maxLon)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	// Add authentication header
	req.Header.Set("Authorization", "Bearer "+fb.apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := fb.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var result struct {
		Data []Aircraft `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode API response: %w", err)
	}

	return result.Data, nil
}

// cosDeg returns cosine of angle in degrees
func cosDeg(deg float64) float64 {
	return math.Cos(deg * math.Pi / 180.0)
}

// getFromCache retrieves aircraft from local cache
func (fb *FR24Bridge) getFromCache(lat, lon float64, radiusKm int) ([]Aircraft, error) {
	// Calculate bounding box
	latDelta := float64(radiusKm) / 111.0
	lonDelta := float64(radiusKm) / (111.0 * cosDeg(lat))

	rows, err := fb.cacheDB.Query(
		`SELECT icao24, callsign, origin_country, latitude, longitude,
		baro_altitude, velocity, true_track, vertical_rate, squawk, last_updated
		FROM active_aircraft
		WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?`,
		lat-latDelta, lat+latDelta, lon-lonDelta, lon+lonDelta,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	aircraft := []Aircraft{}
	for rows.Next() {
		var ac Aircraft
		var lastUpdated string
		err := rows.Scan(
			&ac.ICAO24, &ac.Callsign, &ac.OriginCountry,
			&ac.Latitude, &ac.Longitude, &ac.BaroAltitude,
			&ac.Velocity, &ac.TrueTrack, &ac.VerticalRate,
			&ac.Squawk, &lastUpdated,
		)
		if err != nil {
			continue
		}
		// Parse timestamp
		ac.LastContact = parseTime(lastUpdated)
		aircraft = append(aircraft, ac)
	}

	return aircraft, nil
}

// parseTime parses RFC3339 timestamp
func parseTime(s string) float64 {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0
	}
	return float64(t.Unix())
}

// isCacheFresh checks if cached data is fresh (< 15 minutes old)
func (fb *FR24Bridge) isCacheFresh(aircraft []Aircraft) bool {
	if len(aircraft) == 0 {
		return false
	}

	// Check the most recent update
	mostRecent := aircraft[0].LastContact
	for _, ac := range aircraft {
		if ac.LastContact > mostRecent {
			mostRecent = ac.LastContact
		}
	}

	age := float64(time.Now().Unix()) - mostRecent
	return age < 900 // 15 minutes = 900 seconds
}

// updateCache updates the local cache with new aircraft data
func (fb *FR24Bridge) updateCache(aircraft []Aircraft) error {
	tx, err := fb.cacheDB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(
		`INSERT OR REPLACE INTO aircraft_cache
		(icao24, callsign, origin_country, latitude, longitude,
		baro_altitude, velocity, true_track, vertical_rate, squawk,
		last_updated, cache_ttl)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now', '+15 minutes'))`,
	)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, ac := range aircraft {
		_, err := stmt.Exec(
			ac.ICAO24, ac.Callsign, ac.OriginCountry,
			ac.Latitude, ac.Longitude, ac.BaroAltitude,
			ac.Velocity, ac.TrueTrack, ac.VerticalRate,
			ac.Squawk,
		)
		if err != nil {
			continue
		}
	}

	return tx.Commit()
}

// CheckNegativeCorrelation checks if a detected signal correlates with known aircraft
// Returns true if the signal is likely civilian aircraft (not a threat)
func (fb *FR24Bridge) CheckNegativeCorrelation(lat, lon, altitude float64, timestamp int64) (bool, *Aircraft, error) {
	aircraft, err := fb.GetADSBAround(lat, lon, 50) // 50km radius
	if err != nil {
		return false, nil, err
	}

	for _, ac := range aircraft {
		// Position match: within 5km horizontally
		distKm := haversine(lat, lon, ac.Latitude, ac.Longitude)
		if distKm > 5.0 {
			continue
		}

		// Altitude match: within 1000ft (304m)
		altDiff := abs(altitude - ac.GeoAltitude)
		if altDiff > 304.0 {
			continue
		}

		// Time match: within 30 seconds
		timeDiff := abs(float64(timestamp) - ac.LastContact)
		if timeDiff > 30.0 {
			continue
		}

		// Strong correlation found - likely civilian aircraft
		return true, &ac, nil
	}

	// No correlation - potential threat
	return false, nil, nil
}

// haversine calculates great-circle distance between two points
func haversine(lat1, lon1, lat2, lon2 float64) float64 {
	// Convert to radians
	lat1Rad := lat1 * math.Pi / 180.0
	lat2Rad := lat2 * math.Pi / 180.0
	deltaLat := (lat2 - lat1) * math.Pi / 180.0
	deltaLon := (lon2 - lon1) * math.Pi / 180.0

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return 6371.0 * c // Earth radius in km
}

// GetCacheStats returns cache statistics
func (fb *FR24Bridge) GetCacheStats() (int, int, error) {
	var totalCount, activeCount int

	err := fb.cacheDB.QueryRow("SELECT COUNT(*) FROM aircraft_cache").Scan(&totalCount)
	if err != nil {
		return 0, 0, err
	}

	err = fb.cacheDB.QueryRow("SELECT COUNT(*) FROM active_aircraft").Scan(&activeCount)
	if err != nil {
		return totalCount, 0, err
	}

	return totalCount, activeCount, nil
}

// ClearExpiredCache removes expired cache entries
func (fb *FR24Bridge) ClearExpiredCache() error {
	_, err := fb.cacheDB.Exec("DELETE FROM aircraft_cache WHERE cache_ttl < datetime('now')")
	return err
}

// Close closes the cache database
func (fb *FR24Bridge) Close() error {
	return fb.cacheDB.Close()
}
