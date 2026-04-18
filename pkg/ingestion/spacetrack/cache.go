package spacetrack

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"ssis/pkg/physics"
)

/**
 * CacheEntry represents a serialized set of TLEs with metadata for TTL validation.
 * Used to stabilize the SSIS platform against API rate limiting and network instability.
 */
type CacheEntry struct {
	Timestamp time.Time      `json:"timestamp"`
	Objects   []*physics.TLE `json:"objects"`
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
 */
func NewPersistentCache(dir string) *PersistentCache {
	fmt.Printf("[TRC] Entering NewPersistentCache for %s\n", dir)
	defer fmt.Printf("[TRC] Exiting NewPersistentCache\n")
	return &PersistentCache{
		CacheDir: dir,
		TTL:      4 * time.Hour,
	}
}

/**
 * Save persists a slice of TLEs to a uniquely named file based on the query name.
 * Uses JSON serialization for the outer envelope to preserve timestamp metadata.
 */
func (c *PersistentCache) Save(name string, objects []*physics.TLE) error {
	fmt.Printf("[TRC] Entering PersistentCache.Save for %s (%d objects)\n", name, len(objects))
	defer fmt.Printf("[TRC] Exiting PersistentCache.Save\n")

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

	return nil
}

/**
 * Load retrieves a cached entry if it exists and has not exceeded the TTL.
 * Returns the objects or an error if the cache is missing or stale.
 */
func (c *PersistentCache) Load(name string) ([]*physics.TLE, error) {
	fmt.Printf("[TRC] Entering PersistentCache.Load for %s\n", name)
	defer fmt.Printf("[TRC] Exiting PersistentCache.Load\n")

	path := filepath.Join(c.CacheDir, name+".json")
	
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("cache mission for %s", name)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to stat cache file: %w", err)
	}

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

	return entry.Objects, nil
}
