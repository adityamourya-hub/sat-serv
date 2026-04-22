package spaceweather

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"ssis/pkg/physics"
)

/**
 * NOAAClient fetches real-time space weather indices from NOAA SWPC.
 *
 * Data Sources:
 * - F10.7 cm Solar Flux: https://services.swpc.noaa.gov/json/f107_cm_flux.json
 * - Planetary K-index (Kp): https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
 * - Ap Index: Derived from Kp
 *
 * Space weather drives atmospheric density variations for drag modeling.
 */
type NOAAClient struct {
	httpClient *http.Client
	baseURL    string

	// Rate limiting (NOAA is generous, but be polite)
	mu          sync.Mutex
	lastFetch   time.Time
	minInterval time.Duration

	// Cache
	cache *physics.SpaceWeatherCache
}

/**
 * NOAAF107Data represents the F10.7 flux JSON structure from NOAA.
 */
type NOAAF107Data []struct {
	TimeTag  string  `json:"time_tag"`
	Flux     float64 `json:"flux"`
	Observed float64 `json:"observed_flux"`
	Adjusted float64 `json:"adjusted_flux"`
}

/**
 * NOAAKpData represents the Kp index JSON structure from NOAA.
 */
type NOAAKpData []struct {
	TimeTag string  `json:"time_tag"`
	Kp      string  `json:"kp"`
	KpValue float64 // Parsed from string "3+" etc.
}

/**
 * NewNOAAClient creates a space weather client with sensible defaults.
 */
func NewNOAAClient() *NOAAClient {
	return &NOAAClient{
		httpClient:  &http.Client{Timeout: 10 * time.Second},
		baseURL:     "https://services.swpc.noaa.gov/json",
		minInterval: 5 * time.Minute, // NOAA updates every 1-3 minutes, fetch every 5
		cache:       physics.NewSpaceWeatherCache(),
	}
}

/**
 * FetchLatest retrieves the most recent space weather data from NOAA.
 *
 * @return: SpaceWeatherData with F10.7, Kp, and derived Ap
 */
func (c *NOAAClient) FetchLatest() (physics.SpaceWeatherData, error) {
	// Rate limiting
	c.mu.Lock()
	elapsed := time.Since(c.lastFetch)
	if elapsed < c.minInterval {
		// Return cached data if available
		cached := c.cache.Get()
		if cached.F107 > 0 {
			c.mu.Unlock()
			return cached, nil
		}
	}
	c.lastFetch = time.Now()
	c.mu.Unlock()

	// Fetch F10.7
	f107, err := c.fetchF107()
	if err != nil {
		return physics.SpaceWeatherData{}, fmt.Errorf("F10.7 fetch failed: %w", err)
	}

	// Fetch Kp
	kp, err := c.fetchKp()
	if err != nil {
		return physics.SpaceWeatherData{}, fmt.Errorf("Kp fetch failed: %w", err)
	}

	// Calculate Ap from Kp (approximate conversion)
	ap := kpToAp(kp)

	sw := physics.SpaceWeatherData{
		F107:      f107,
		F107Avg:   f107, // Use current as 81-day avg if unavailable
		Ap:        ap,
		Kp:        kp,
		Timestamp: time.Now(),
	}

	// Update cache
	c.cache.Update(sw)

	return sw, nil
}

/**
 * fetchF107 retrieves the latest F10.7 cm solar flux.
 */
func (c *NOAAClient) fetchF107() (float64, error) {
	url := fmt.Sprintf("%s/f107_cm_flux.json", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}

	var data NOAAF107Data
	if err := json.Unmarshal(body, &data); err != nil {
		return 0, err
	}

	if len(data) == 0 {
		return 0, fmt.Errorf("no F10.7 data available")
	}

	// Use the most recent entry
	latest := data[len(data)-1]

	// Prefer adjusted flux, fall back to observed
	flux := latest.Adjusted
	if flux == 0 {
		flux = latest.Observed
	}
	if flux == 0 {
		flux = latest.Flux
	}

	return flux, nil
}

/**
 * fetchKp retrieves the latest planetary K-index.
 */
func (c *NOAAClient) fetchKp() (float64, error) {
	url := fmt.Sprintf("%s/planetary_k_index_1m.json", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}

	var data NOAAKpData
	if err := json.Unmarshal(body, &data); err != nil {
		return 0, err
	}

	if len(data) == 0 {
		return 0, fmt.Errorf("no Kp data available")
	}

	// Use the most recent entry
	latest := data[len(data)-1]
	kp := parseKpString(latest.Kp)

	return kp, nil
}

/**
 * parseKpString converts NOAA Kp format ("3+", "5-", "7o", "9+") to numeric.
 */
func parseKpString(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}

	// Handle the modifier
	modifier := 0.0
	if strings.HasSuffix(s, "+") {
		modifier = 0.33
		s = strings.TrimSuffix(s, "+")
	} else if strings.HasSuffix(s, "-") {
		modifier = -0.33
		s = strings.TrimSuffix(s, "-")
	} else if strings.HasSuffix(s, "o") {
		modifier = 0.0
		s = strings.TrimSuffix(s, "o")
	}

	// Parse base value
	base, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}

	return base + modifier
}

/**
 * kpToAp converts Kp index to Ap index (planetary amplitude).
 *
 * Approximate conversion (NOAA standard):
 * Kp 0 = Ap 0, Kp 1 = Ap 4, Kp 2 = Ap 7, Kp 3 = Ap 15, ... Kp 9 = Ap 400
 */
func kpToAp(kp float64) float64 {
	// Conversion table (NOAA SWPC)
	// This is an approximate piecewise linear conversion
	if kp <= 0 {
		return 0
	}
	if kp <= 1 {
		return kp * 4 // 0-4
	}
	if kp <= 2 {
		return 4 + (kp-1)*3 // 4-7
	}
	if kp <= 3 {
		return 7 + (kp-2)*8 // 7-15
	}
	if kp <= 4 {
		return 15 + (kp-3)*12 // 15-27
	}
	if kp <= 5 {
		return 27 + (kp-4)*30 // 27-57
	}
	if kp <= 6 {
		return 57 + (kp-5)*65 // 57-122
	}
	if kp <= 7 {
		return 122 + (kp-6)*108 // 122-230
	}
	if kp <= 8 {
		return 230 + (kp-7)*132 // 230-362
	}
	return 362 + (kp-8)*38 // 362-400 for Kp 9
}

/**
 * StartPoller begins a background goroutine that fetches space weather periodically.
 *
 * @param interval: How often to poll NOAA (recommended: 5-15 minutes)
 * @param onUpdate: Callback invoked when new data is available
 */
func (c *NOAAClient) StartPoller(interval time.Duration, onUpdate func(physics.SpaceWeatherData)) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// Immediate first fetch
		if sw, err := c.FetchLatest(); err == nil && onUpdate != nil {
			onUpdate(sw)
		}

		for range ticker.C {
			if sw, err := c.FetchLatest(); err == nil && onUpdate != nil {
				onUpdate(sw)
			}
		}
	}()
}

/**
 * GetCache returns the internal space weather cache.
 */
func (c *NOAAClient) GetCache() *physics.SpaceWeatherCache {
	return c.cache
}
