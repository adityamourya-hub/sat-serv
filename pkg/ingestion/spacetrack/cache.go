package spacetrack

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"ssis/pkg/orbital"
)

/**
 * CacheEntry represents a serialized set of TLEs with metadata for TTL validation.
 * Used to stabilize the SSIS platform against API rate limiting and network instability.
 */
type CacheEntry struct {
	Timestamp time.Time      `json:"timestamp"`
	Objects   []*orbital.TLE `json:"objects"`
}

/**
 * PersistentCache manages the lifecycle of orbital data on the local filesystem.
 * Adheres to sovereign data persistence standards for field-operational SSA substrates.
 */
type PersistentCache struct {
	CacheDir string
	TTL      time.Duration
}

/**
 * NewPersistentCache initializes the cache directory and sets the default TTL for TLE data.
 * <plan_complexity>Low</plan_complexity><total_lines_estimate>90</total_lines_estimate>
 */
func NewPersistentCache(dir string) *PersistentCache {
	fmt.Printf("DEBUG: Initializing Persistent Cache in %s\n", dir)
	return &PersistentCache{
		CacheDir: dir,
		TTL:      4 * time.Hour,
	}
}

/**
 * Save persists a slice of TLEs to a uniquely named file based on the query name.
 * Uses JSON serialization for the outer envelope to preserve timestamp metadata.
 */
func (c *PersistentCache) Save(name string, objects []*orbital.TLE) error {
	fmt.Printf("DEBUG: Saving %d objects to cache: %s\n", len(objects), name)
	if err := os.MkdirAll(c.CacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	entry := CacheEntry{
		Timestamp: time.Now(),
		Objects:   objects,
	}

	data, err := json.MarshalIndent(entry, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal cache entry: %w", err)
	}

	path := filepath.Join(c.CacheDir, name+".json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write cache file: %w", err)
	}

	fmt.Println("DEBUG: Cache save complete")
	return nil
}

/**
 * Load retrieves a cached entry if it exists and has not exceeded the TTL.
 * Returns the objects or an error if the cache is missing or stale.
 */
func (c *PersistentCache) Load(name string) ([]*orbital.TLE, error) {
	fmt.Printf("DEBUG: Attempting to load from cache: %s\n", name)
	path := filepath.Join(c.CacheDir, name+".json")
	
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("cache mission for %s", name)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to stat cache file: %w", err)
	}

	// Double check TTL against file system and internal timestamp
	if time.Since(info.ModTime()) > c.TTL {
		return nil, fmt.Errorf("cache stale for %s (age > %v)", name, c.TTL)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read cache file: %w", err)
	}

	var entry CacheEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cache entry: %w", err)
	}

	if time.Since(entry.Timestamp) > c.TTL {
		return nil, fmt.Errorf("cache payload stale for %s", name)
	}

	fmt.Printf("DEBUG: Successfully loaded %d objects from cache\n", len(entry.Objects))
	return entry.Objects, nil
}
