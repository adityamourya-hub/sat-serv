// Command ingestor provides Space-Track TLE/CDM/TIP ingestion worker
// IAF SDLC: Implements strict "Exact ID" parser - no wildcards allowed
package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/time/rate"

	"ssis/internal/db/postgis"
	"ssis/internal/db/repository"
)

// Config holds ingestor configuration
type Config struct {
	// Space-Track credentials
	SpaceTrackUsername string
	SpaceTrackPassword string
	SpaceTrackBaseURL  string

	// Database
	DatabaseURL string

	// Ingestion settings
	BatchSize       int
	RateLimitPerSec int
	QueryTimeout    time.Duration
	IngestionWindow time.Duration // How far back to query

	// IAF Air-Gap Support
	CacheDir    string // Local GZIP cache for air-gap mode
	EnableCache bool   // Write all API responses to cache
	AirGapMode  bool   // Skip API calls, use cache only

	// Source attribution
	SourceAttribution string // "SPACETRACK" or sensor ID
}

// DefaultConfig returns IAF production defaults
func DefaultConfig() *Config {
	return &Config{
		SpaceTrackBaseURL: "https://www.space-track.org",
		BatchSize:         100,
		RateLimitPerSec:   1, // Conservative: 1 req/sec to avoid throttling
		QueryTimeout:      30 * time.Second,
		IngestionWindow:   24 * time.Hour,
		CacheDir:          "./data/cache",
		EnableCache:       true,
		AirGapMode:        false,
		SourceAttribution: "SPACETRACK",
	}
}

// SpaceTrackClient provides authenticated access to Space-Track API
// Uses standard cookiejar for session persistence and TeeReader for dual-stream caching
type SpaceTrackClient struct {
	baseURL     string
	username    string
	password    string
	httpClient  *http.Client
	rateLimiter *rate.Limiter
	cacheDir    string
	enableCache bool
	airGapMode  bool
}

// NewSpaceTrackClient creates authenticated client with standard cookiejar
func NewSpaceTrackClient(baseURL, username, password string, rateLimit int, cacheDir string, enableCache, airGapMode bool) *SpaceTrackClient {
	// Create standard cookiejar for session management
	jar, _ := cookiejar.New(nil)

	return &SpaceTrackClient{
		baseURL:  baseURL,
		username: username,
		password: password,
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // Extended for bulk catalog download
			Jar:     jar,              // Standard cookiejar handles session
		},
		rateLimiter: rate.NewLimiter(rate.Limit(rateLimit), 1),
		cacheDir:    cacheDir,
		enableCache: enableCache,
		airGapMode:  airGapMode,
	}
}

// ensureCacheDir creates cache directory if needed
func (c *SpaceTrackClient) ensureCacheDir() error {
	if !c.enableCache {
		return nil
	}
	return os.MkdirAll(c.cacheDir, 0755)
}

// cacheFilename generates cache filename from URL
func (c *SpaceTrackClient) cacheFilename(url string) string {
	hash := strconv.Itoa(int(time.Now().Unix()))
	return filepath.Join(c.cacheDir, fmt.Sprintf("spacetrack_%s.json.gz", hash))
}

// writeToCache saves response to GZIP file
func (c *SpaceTrackClient) writeToCache(filename string, data []byte) error {
	if !c.enableCache {
		return nil
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	gzWriter := gzip.NewWriter(file)
	defer gzWriter.Close()

	_, err = gzWriter.Write(data)
	return err
}

// readFromCache loads response from GZIP file
func (c *SpaceTrackClient) readFromCache(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzReader.Close()

	return io.ReadAll(gzReader)
}

// Authenticate performs login to Space-Track
func (c *SpaceTrackClient) Authenticate(ctx context.Context) error {
	// Wait for rate limit
	if err := c.rateLimiter.Wait(ctx); err != nil {
		return fmt.Errorf("rate limit wait: %w", err)
	}

	loginURL := fmt.Sprintf("%s/ajaxauth/login", c.baseURL)
	data := url.Values{
		"identity": {c.username},
		"password": {c.password},
	}

	req, err := http.NewRequestWithContext(ctx, "POST", loginURL, strings.NewReader(data.Encode()))
	if err != nil {
		return fmt.Errorf("create login request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("login request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("login failed: status %d", resp.StatusCode)
	}

	// Standard cookiejar in http.Client handles session automatically
	log.Println("Space-Track authentication successful")
	return nil
}

// QueryFilter represents strict exact-ID query parameters
// NO WILDCARDS ALLOWED - IAF requirement
type QueryFilter struct {
	NoradCatID *int32 // Exact NORAD catalog ID
	ObjectID   string // International Designator: YYYY-XXXCCC format
	ObjectName string // Exact object name match
	EpochStart string // ISO 8601
	EpochEnd   string // ISO 8601
}

// Validate enforces strict exact-ID parsing (no wildcards)
// Alpha-5 Support: 9-digit NORAD IDs for catalog approaching 100,000 objects
func (f *QueryFilter) Validate() error {
	// Check for wildcards in object_name
	if strings.Contains(f.ObjectName, "*") || strings.Contains(f.ObjectName, "%") ||
		strings.Contains(f.ObjectName, "?") || strings.Contains(f.ObjectName, "_") {
		return fmt.Errorf("WILDCARD_NOT_ALLOWED: object_name contains forbidden character: %s", f.ObjectName)
	}

	// Validate International Designator format: YYYY-XXXCCC
	if f.ObjectID != "" {
		validID := regexp.MustCompile(`^\d{4}-\d{3}[A-Z]{1,3}$`)
		if !validID.MatchString(f.ObjectID) {
			return fmt.Errorf("INVALID_INTL_DESIGNATOR: %s (expected format: YYYY-XXXCCC)", f.ObjectID)
		}
	}

	// Validate NORAD ID range (Alpha-5: supports 1-339999, beyond legacy 99999 limit)
	if f.NoradCatID != nil {
		if *f.NoradCatID < 1 || *f.NoradCatID > 339999 {
			return fmt.Errorf("INVALID_NORAD_ID: %d (expected 1-339999 for Alpha-5 compatibility)", *f.NoradCatID)
		}
	}

	return nil
}

// FetchTLE queries Space-Track for TLE data
func (c *SpaceTrackClient) FetchTLE(ctx context.Context, filter *QueryFilter) ([]TLEData, error) {
	if err := filter.Validate(); err != nil {
		return nil, err
	}

	// Wait for rate limit
	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	// Build query URL
	queryURL := fmt.Sprintf("%s/basicspacedata/query/class/tle_latest/", c.baseURL)

	// Add predicates (exact matches only)
	var predicates []string
	if filter.NoradCatID != nil {
		predicates = append(predicates, fmt.Sprintf("NORAD_CAT_ID/%d", *filter.NoradCatID))
	}
	if filter.ObjectID != "" {
		predicates = append(predicates, fmt.Sprintf("OBJECT_ID/%s", url.QueryEscape(filter.ObjectID)))
	}
	if filter.ObjectName != "" {
		predicates = append(predicates, fmt.Sprintf("OBJECT_NAME/%s", url.QueryEscape(filter.ObjectName)))
	}
	if filter.EpochStart != "" {
		predicates = append(predicates, fmt.Sprintf("EPOCH/%s--%s", filter.EpochStart, filter.EpochEnd))
	}

	if len(predicates) > 0 {
		queryURL += strings.Join(predicates, "/") + "/"
	}

	queryURL += "orderby/EPOCH desc/"
	queryURL += fmt.Sprintf("limit/%d/", 100) // Max 100 per query
	queryURL += "format/json"

	req, err := http.NewRequestWithContext(ctx, "GET", queryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Add auth cookies
	// Standard cookiejar in http.Client automatically adds cookies

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		// Re-authenticate and retry once
		if err := c.Authenticate(ctx); err != nil {
			return nil, fmt.Errorf("re-authentication failed: %w", err)
		}
		// Retry request
		return c.FetchTLE(ctx, filter)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Space-Track API error: status %d", resp.StatusCode)
	}

	var tles []TLEData
	if err := json.NewDecoder(resp.Body).Decode(&tles); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return tles, nil
}

// TLEData represents Space-Track TLE response
type TLEData struct {
	NORADCatID     int32   `json:"NORAD_CAT_ID"`
	ObjectID       string  `json:"OBJECT_ID"` // Intl Designator
	ObjectName     string  `json:"OBJECT_NAME"`
	Classification string  `json:"CLASSIFICATION_TYPE"`
	Epoch          string  `json:"EPOCH"`
	MeanMotion     float64 `json:"MEAN_MOTION"`
	Eccentricity   float64 `json:"ECCENTRICITY"`
	Inclination    float64 `json:"INCLINATION"`
	RAAN           float64 `json:"RA_OF_ASC_NODE"`
	ArgPerigee     float64 `json:"ARG_OF_PERICENTER"`
	MeanAnomaly    float64 `json:"MEAN_ANOMALY"`
	BStar          float64 `json:"BSTAR"`
	TLELine1       string  `json:"TLE_LINE1"`
	TLELine2       string  `json:"TLE_LINE2"`
}

// TIPData represents Space-Track TIP (Tracking and Impact Prediction) message
type TIPData struct {
	NORADCatID      int32   `json:"NORAD_CAT_ID"`
	ObjectName      string  `json:"OBJECT_NAME"`
	MsgEpoch        string  `json:"MSG_EPOCH"`
	WindowStart     string  `json:"WINDOW_START"`
	WindowEnd       string  `json:"WINDOW_END"`
	ImpactLatitude  float64 `json:"IMPACT_LATITUDE"`
	ImpactLongitude float64 `json:"IMPACT_LONGITUDE"`
	ImpactAltitude  float64 `json:"IMPACT_ALTITUDE"`
	Uncertainty     float64 `json:"IMPACT_UNCERTAINTY_KM"`
	ReentryAngle    float64 `json:"REENTRY_ANGLE"`
	VelocitySurvive float64 `json:"VELOCITY_AT_SURVIVE"`
}

// FetchTIP queries Space-Track for re-entry prediction messages (Module h)
func (c *SpaceTrackClient) FetchTIP(ctx context.Context, noradID *int32) ([]TIPData, error) {
	if c.airGapMode {
		return nil, fmt.Errorf("TIP fetch disabled in air-gap mode")
	}

	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	queryURL := fmt.Sprintf("%s/basicspacedata/query/class/tip/", c.baseURL)
	if noradID != nil {
		queryURL += fmt.Sprintf("NORAD_CAT_ID/%d/", *noradID)
	}
	queryURL += "orderby/WINDOW_START desc/format/json"

	req, err := http.NewRequestWithContext(ctx, "GET", queryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Standard cookiejar in http.Client automatically adds cookies

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		if err := c.Authenticate(ctx); err != nil {
			return nil, fmt.Errorf("re-authentication failed: %w", err)
		}
		return c.FetchTIP(ctx, noradID)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Space-Track API error: status %d", resp.StatusCode)
	}

	var tips []TIPData
	if err := json.NewDecoder(resp.Body).Decode(&tips); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return tips, nil
}

// CDMData represents Conjunction Data Message
type CDMData struct {
	CDMID         int64   `json:"CDM_ID"`
	MessageFor    string  `json:"MESSAGE_FOR"`
	NORADCatID    int32   `json:"NORAD_CAT_ID"`
	TCA           string  `json:"TCA"` // Time of Closest Approach
	MissDistance  float64 `json:"MISS_DISTANCE"`
	RelativeSpeed float64 `json:"RELATIVE_SPEED"`
	PC            float64 `json:"PC"` // Probability of Collision
	TCATRange     float64 `json:"TCA_T_RANGE"`
	TCARangeRate  float64 `json:"TCA_R_RANGE"`
	TCARadialRate float64 `json:"TCA_RADIAL_RANGE"`
}

// FetchCDM queries Space-Track for conjunction data messages (Module d)
func (c *SpaceTrackClient) FetchCDM(ctx context.Context, noradID *int32) ([]CDMData, error) {
	if c.airGapMode {
		return nil, fmt.Errorf("CDM fetch disabled in air-gap mode")
	}

	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	queryURL := fmt.Sprintf("%s/basicspacedata/query/class/cdm/", c.baseURL)
	if noradID != nil {
		queryURL += fmt.Sprintf("NORAD_CAT_ID/%d/", *noradID)
	}
	queryURL += "orderby/TCA asc/format/json"

	req, err := http.NewRequestWithContext(ctx, "GET", queryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Standard cookiejar in http.Client automatically adds cookies

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		if err := c.Authenticate(ctx); err != nil {
			return nil, fmt.Errorf("re-authentication failed: %w", err)
		}
		return c.FetchCDM(ctx, noradID)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Space-Track API error: status %d", resp.StatusCode)
	}

	var cdms []CDMData
	if err := json.NewDecoder(resp.Body).Decode(&cdms); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return cdms, nil
}

// SatcatData represents Satellite Catalog metadata
type SatcatData struct {
	NORADCatID    int32   `json:"NORAD_CAT_ID"`
	ObjectID      string  `json:"OBJECT_ID"` // International Designator
	SatelliteName string  `json:"SATNAME"`
	Country       string  `json:"COUNTRY"`
	LaunchDate    string  `json:"LAUNCH_DATE"`
	LaunchSite    string  `json:"LAUNCH_SITE"`
	DecayDate     *string `json:"DECAY_DATE,omitempty"`
	OrbitalPeriod float64 `json:"PERIOD,omitempty"`
	Inclination   float64 `json:"INCLINATION,omitempty"`
	Apogee        float64 `json:"APOGEE,omitempty"`
	Perigee       float64 `json:"PERIGEE,omitempty"`
	RCS           float64 `json:"RCS,omitempty"` // Radar Cross Section
}

// FetchSatcat queries Space-Track for satellite catalog metadata
func (c *SpaceTrackClient) FetchSatcat(ctx context.Context, noradID *int32) ([]SatcatData, error) {
	if c.airGapMode {
		return nil, fmt.Errorf("satcat fetch disabled in air-gap mode")
	}

	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	queryURL := fmt.Sprintf("%s/basicspacedata/query/class/satcat/", c.baseURL)
	if noradID != nil {
		queryURL += fmt.Sprintf("NORAD_CAT_ID/%d/", *noradID)
	}
	queryURL += "format/json"

	req, err := http.NewRequestWithContext(ctx, "GET", queryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Standard cookiejar in http.Client automatically adds cookies

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		if err := c.Authenticate(ctx); err != nil {
			return nil, fmt.Errorf("re-authentication failed: %w", err)
		}
		return c.FetchSatcat(ctx, noradID)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Space-Track API error: status %d", resp.StatusCode)
	}

	var satcat []SatcatData
	if err := json.NewDecoder(resp.Body).Decode(&satcat); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return satcat, nil
}

// GPHistoryFilter for gp_history endpoint
type GPHistoryFilter struct {
	NoradCatID *int32
	StartDate  string // YYYY-MM-DD
	EndDate    string // YYYY-MM-DD
}

// FetchGPHistory queries Space-Track for historical TLEs (Module f - Manoeuvre Detection)
func (c *SpaceTrackClient) FetchGPHistory(ctx context.Context, filter *GPHistoryFilter) ([]TLEData, error) {
	if c.airGapMode {
		return nil, fmt.Errorf("gp_history fetch disabled in air-gap mode")
	}

	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	queryURL := fmt.Sprintf("%s/basicspacedata/query/class/gp_history/", c.baseURL)
	if filter.NoradCatID != nil {
		queryURL += fmt.Sprintf("NORAD_CAT_ID/%d/", *filter.NoradCatID)
	}
	if filter.StartDate != "" && filter.EndDate != "" {
		queryURL += fmt.Sprintf("EPOCH/%s--%s/", filter.StartDate, filter.EndDate)
	}
	queryURL += "orderby/EPOCH desc/format/json"

	req, err := http.NewRequestWithContext(ctx, "GET", queryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Standard cookiejar in http.Client automatically adds cookies

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		if err := c.Authenticate(ctx); err != nil {
			return nil, fmt.Errorf("re-authentication failed: %w", err)
		}
		return c.FetchGPHistory(ctx, filter)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Space-Track API error: status %d", resp.StatusCode)
	}

	var tles []TLEData
	if err := json.NewDecoder(resp.Body).Decode(&tles); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return tles, nil
}

// Ingestor coordinates Space-Track ingestion into ISDA database
type Ingestor struct {
	client            *SpaceTrackClient
	statesRepo        *repository.CurrentStatesRepo
	metadataRepo      *repository.AssetMetadataRepo
	sourceAttribution string // SPACETRACK or sensor ID
	enableCache       bool
}

// NewIngestor creates ingestor instance
func NewIngestor(client *SpaceTrackClient, pool *pgxpool.Pool, sourceAttribution string, enableCache bool) *Ingestor {
	return &Ingestor{
		client:            client,
		statesRepo:        repository.NewCurrentStatesRepo(pool),
		metadataRepo:      repository.NewAssetMetadataRepo(pool),
		sourceAttribution: sourceAttribution,
		enableCache:       enableCache,
	}
}

// RunDailySync performs full daily synchronization
func (i *Ingestor) RunDailySync(ctx context.Context) error {
	log.Println("Starting daily Space-Track sync...")

	// Re-authenticate
	if err := i.client.Authenticate(ctx); err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	// Fetch recent TLEs (last 24 hours)
	now := time.Now().UTC()
	start := now.Add(-24 * time.Hour)

	filter := &QueryFilter{
		EpochStart: start.Format("2006-01-02"),
		EpochEnd:   now.Format("2006-01-02"),
	}

	tles, err := i.client.FetchTLE(ctx, filter)
	if err != nil {
		return fmt.Errorf("fetch TLEs: %w", err)
	}

	log.Printf("Fetched %d TLEs from Space-Track", len(tles))

	// Process in batches
	for _, tle := range tles {
		if err := i.processTLE(ctx, &tle); err != nil {
			log.Printf("Failed to process NORAD %d: %v", tle.NORADCatID, err)
			continue
		}
	}

	log.Println("Daily sync completed")
	return nil
}

// processTLE inserts/updates TLE and asset metadata
func (i *Ingestor) processTLE(ctx context.Context, tle *TLEData) error {
	// Parse epoch
	epoch, err := time.Parse("2006-01-02 15:04:05.000000", tle.Epoch)
	if err != nil {
		epoch = time.Now() // Fallback
	}

	// Create state vector
	sv := &repository.StateVector{
		NoradCatID:   tle.NORADCatID,
		ObjectID:     tle.ObjectID,
		ObjectName:   tle.ObjectName,
		Epoch:        epoch,
		MeanMotion:   tle.MeanMotion,
		Eccentricity: tle.Eccentricity,
		Inclination:  tle.Inclination,
		RAAN:         tle.RAAN,
		ArgOfPerigee: tle.ArgPerigee,
		MeanAnomaly:  tle.MeanAnomaly,
		BStar:        tle.BStar,
		Owner:        "UNKNOWN", // TODO: Map from country code
		ObjectType:   "UNKNOWN",
		TLELine1:     tle.TLELine1,
		TLELine2:     tle.TLELine2,
	}

	// Insert/update current_states
	if err := i.statesRepo.UpdateTLE(ctx, sv); err != nil {
		return fmt.Errorf("update state: %w", err)
	}

	// Create/update asset metadata (static data)
	am := &repository.AssetMetadata{
		NoradCatID:     tle.NORADCatID,
		IntlDesignator: tle.ObjectID,
		ObjectName:     tle.ObjectName,
		ObjectType:     mapObjectType(tle.Classification),
	}

	if err := i.metadataRepo.Upsert(ctx, am); err != nil {
		return fmt.Errorf("upsert metadata: %w", err)
	}

	return nil
}

// mapObjectType maps Space-Track classification to ISDA object type
func mapObjectType(class string) string {
	switch class {
	case "U":
		return "UNKNOWN"
	case "C":
		return "CLASSIFIED"
	default:
		return "UNKNOWN"
	}
}

func main() {
	// Load configuration from environment
	cfg := DefaultConfig()
	cfg.SpaceTrackUsername = os.Getenv("SPACETRACK_USERNAME")
	cfg.SpaceTrackPassword = os.Getenv("SPACETRACK_PASSWORD")
	cfg.DatabaseURL = os.Getenv("DATABASE_URL")

	if cfg.SpaceTrackUsername == "" || cfg.SpaceTrackPassword == "" {
		log.Fatal("SPACETRACK_USERNAME and SPACETRACK_PASSWORD environment variables required")
	}

	if cfg.DatabaseURL == "" {
		cfg.DatabaseURL = "postgres://isda_user@localhost:5432/isda?sslmode=require"
	}

	// Initialize database pool
	pgConfig := postgis.DefaultConfig()
	pgConfig.User = cfg.SpaceTrackUsername // Reuse for DB or parse from URL
	pgConfig.Password = ""

	pool, err := postgis.NewPool(pgConfig)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// Verify health
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := pool.HealthCheck(ctx); err != nil {
		log.Fatalf("Database health check failed: %v", err)
	}

	// Ensure cache directory exists
	if err := os.MkdirAll(cfg.CacheDir, 0755); err != nil {
		log.Printf("Warning: Failed to create cache dir: %v", err)
	}

	// Create Space-Track client with caching and air-gap support
	client := NewSpaceTrackClient(cfg.SpaceTrackBaseURL, cfg.SpaceTrackUsername, cfg.SpaceTrackPassword,
		cfg.RateLimitPerSec, cfg.CacheDir, cfg.EnableCache, cfg.AirGapMode)

	// Create ingestor
	ingestor := NewIngestor(client, pool.Pool, cfg.SourceAttribution, cfg.EnableCache)

	// Run sync
	ctx, cancel = context.WithTimeout(context.Background(), cfg.QueryTimeout)
	defer cancel()

	if err := ingestor.RunDailySync(ctx); err != nil {
		log.Fatalf("Ingestion failed: %v", err)
	}
}
