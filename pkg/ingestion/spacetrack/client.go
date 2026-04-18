package spacetrack

import (
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strings"
	"time"

	"ssis/pkg/orbital"
)

/**
 * Client implements a production-grade interface for the Space-Track.org REST API.
 * This client manages authenticated sessions via a CookieJar and provides methods
 * for querying the unified space catalog for tactical SSA data.
 * Adheres to sovereign data ingestion protocols for the SSIS platform.
 */
type Client struct {
	httpClient *http.Client
	baseURL    string
}

/**
 * NewClient initializes a new Space-Track API client with an internal cookie jar
 * for session persistence. Sets reasonable timeouts for tactical ingestion.
 * <plan_complexity>Medium</plan_complexity><total_lines_estimate>120</total_lines_estimate>
 */
func NewClient() (*Client, error) {
	fmt.Println("DEBUG: Initializing Space-Track Client")
	jar, err := cookiejar.New(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create cookie jar: %w", err)
	}

	return &Client{
		httpClient: &http.Client{
			Jar:     jar,
			Timeout: 30 * time.Second,
		},
		baseURL: "https://www.space-track.org",
	}, nil
}

/**
 * Login authenticates against the Space-Track.org service.
 * Stores authentication cookies in the client's internal jar for subsequent requests.
 * Uses the /ajaxauth/login endpoint as requested by the protocol.
 */
func (c *Client) Login(username, password string) error {
	fmt.Printf("DEBUG: Attempting Space-Track login for user: %s\n", username)
	loginURL := fmt.Sprintf("%s/ajaxauth/login", c.baseURL)
	data := url.Values{}
	data.Set("identity", username)
	data.Set("password", password)

	resp, err := c.httpClient.PostForm(loginURL, data)
	if err != nil {
		return fmt.Errorf("login request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("login failed with status: %s", resp.Status)
	}

	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "failed") || strings.Contains(string(body), "Unable") {
		return fmt.Errorf("login credentials rejected by Space-Track")
	}

	fmt.Println("DEBUG: Space-Track login successful")
	return nil
}

/**
 * QueryTLE executes a custom basicspacedata query against the TLE class.
 * Returns a slice of parsed orbital.TLE objects.
 * Supports filtering string (e.g., "NORAD_CAT_ID/25544,38779/orderby/NORAD_CAT_ID/format/tle").
 */
func (c *Client) QueryTLE(query string) ([]*orbital.TLE, error) {
	fmt.Printf("DEBUG: Executing Space-Track TLE query: %s\n", query)
	fullURL := fmt.Sprintf("%s/basicspacedata/query/class/tle_latest/%s", c.baseURL, query)
	
	resp, err := c.httpClient.Get(fullURL)
	if err != nil {
		return nil, fmt.Errorf("query request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("query failed with status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read query response: %w", err)
	}

	rawString := string(body)
	if strings.TrimSpace(rawString) == "" {
		return nil, fmt.Errorf("received empty TLE response from Space-Track")
	}

	// Space-Track 'format/tle' returns raw TLE blocks, which we split by object
	// Note: 2-line elements are often preceded by a name line in '3-line' format
	return parseRawTLEBlocks(rawString)
}

/**
 * parseRawTLEBlocks converts a concatenated string of TLEs into structured objects.
 * It leverages the orbital package's hardened parsing logic.
 */
func parseRawTLEBlocks(data string) ([]*orbital.TLE, error) {
	fmt.Println("DEBUG: Parsing raw TLE blocks from API response")
	lines := strings.Split(data, "\n")
	var results []*orbital.TLE
	var currentBlock []string

	for _, line := range lines {
		// Only trim surrounding block whitespace, but preserve internal column alignment
		// A TLE line must be at least 69 characters; significant padding is preserved.
		if strings.TrimSpace(line) == "" {
			continue
		}
		currentBlock = append(currentBlock, line)
		
		// Standard TLE lines start with '1 ' or '2 '
		if len(currentBlock) >= 2 && strings.HasPrefix(currentBlock[len(currentBlock)-1], "2 ") {
			dataBlock := strings.Join(currentBlock, "\n")
			tle, err := orbital.ParseTLE(dataBlock)
			if err == nil {
				results = append(results, tle)
			} else {
				fmt.Printf("DEBUG: Skipping invalid TLE block: %v\n", err)
			}
			currentBlock = nil
		}
	}

	fmt.Printf("DEBUG: Successfully parsed %d TLEs\n", len(results))
	return results, nil
}
