package spacetrack

import (
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strings"
	"time"

	"ssis/pkg/physics"
)

/**
 * Client implements a production-grade interface for the Space-Track.org REST API.
 * This client manages authenticated sessions via a CookieJar and provides methods
 * for querying the unified space catalog for tactical SSA data.
 */
type Client struct {
	httpClient *http.Client
	baseURL    string
}

/**
 * NewClient initializes a new Space-Track API client with an internal cookie jar
 * for session persistence. Sets reasonable timeouts for tactical ingestion.
 */
func NewClient() (*Client, error) {
	fmt.Printf("[TRC] Entering NewClient\n")
	defer fmt.Printf("[TRC] Exiting NewClient\n")

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
 */
func (c *Client) Login(username, password string) error {
	fmt.Printf("[TRC] Entering Client.Login for user: %s\n", username)
	defer fmt.Printf("[TRC] Exiting Client.Login\n")

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

	return nil
}

/**
 * QueryTLE executes a custom basicspacedata query against the TLE class.
 * Returns a slice of parsed physics.TLE objects.
 */
func (c *Client) QueryTLE(query string) ([]*physics.TLE, error) {
	fmt.Printf("[TRC] Entering Client.QueryTLE: %s\n", query)
	defer fmt.Printf("[TRC] Exiting Client.QueryTLE\n")

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

	return parseRawTLEBlocks(rawString)
}

/**
 * parseRawTLEBlocks converts a concatenated string of TLEs into structured objects.
 */
func parseRawTLEBlocks(data string) ([]*physics.TLE, error) {
	fmt.Printf("[TRC] Entering parseRawTLEBlocks\n")
	defer fmt.Printf("[TRC] Exiting parseRawTLEBlocks\n")

	lines := strings.Split(data, "\n")
	var results []*physics.TLE
	var currentBlock []string

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		currentBlock = append(currentBlock, line)
		
		if len(currentBlock) >= 2 && strings.HasPrefix(currentBlock[len(currentBlock)-1], "2 ") {
			dataBlock := strings.Join(currentBlock, "\n")
			tle, err := physics.ParseTLE(dataBlock)
			if err == nil {
				results = append(results, tle)
			} else {
				fmt.Printf("[TRC] Skipping invalid TLE block: %v\n", err)
			}
			currentBlock = nil
		}
	}

	return results, nil
}
