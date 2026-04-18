package onnx

import (
	"crypto/md5"
	"fmt"
	"os"
	"sync"
	"time"
)

// InferenceCache caches recent inference results to reduce redundant computation.
type InferenceCache struct {
	mu        sync.RWMutex
	cache     map[string]CacheEntry
	maxSize   int
	accessQ   []string // LRU queue
	hitCount  int64
	missCount int64
}

// CacheEntry represents a cached inference result with metadata.
type CacheEntry struct {
	Result      InferenceResult
	CreatedAt   time.Time
	AccessAt    time.Time
	AccessCount int64
}

// NewInferenceCache creates a new LRU inference cache.
func NewInferenceCache(maxSize int) *InferenceCache {
	if maxSize <= 0 {
		maxSize = 512
	}
	return &InferenceCache{
		cache:   make(map[string]CacheEntry),
		maxSize: maxSize,
		accessQ: make([]string, 0, maxSize),
	}
}

// Get retrieves a cached result if it exists and is not stale.
func (c *InferenceCache) Get(requestID string) (InferenceResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.cache[requestID]
	if !exists {
		c.missCount++
		return InferenceResult{}, false
	}

	entry.AccessAt = time.Now()
	entry.AccessCount++
	c.cache[requestID] = entry
	c.hitCount++

	return entry.Result, true
}

// Put stores a result in the cache, evicting LRU entries if necessary.
func (c *InferenceCache) Put(requestID string, result InferenceResult) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.cache) >= c.maxSize {
		// Evict LRU entry
		if len(c.accessQ) > 0 {
			oldestKey := c.accessQ[0]
			delete(c.cache, oldestKey)
			c.accessQ = c.accessQ[1:]
		}
	}

	c.cache[requestID] = CacheEntry{
		Result:      result,
		CreatedAt:   time.Now(),
		AccessAt:    time.Now(),
		AccessCount: 1,
	}
	c.accessQ = append(c.accessQ, requestID)
}

// Clear empties the cache.
func (c *InferenceCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string]CacheEntry)
	c.accessQ = make([]string, 0, c.maxSize)
}

// Stats returns cache performance statistics.
func (c *InferenceCache) Stats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	totalAccess := c.hitCount + c.missCount
	hitRate := 0.0
	if totalAccess > 0 {
		hitRate = float64(c.hitCount) / float64(totalAccess)
	}

	return map[string]interface{}{
		"hit_count":    c.hitCount,
		"miss_count":   c.missCount,
		"total_access": totalAccess,
		"hit_rate":     hitRate,
		"current_size": len(c.cache),
		"max_size":     c.maxSize,
		"utilization":  float64(len(c.cache)) / float64(c.maxSize),
	}
}

// --- Model Versioning ---

// ModelVersion represents a specific version of a deployed model.
type ModelVersion struct {
	ID          string    // version identifier (e.g., "1.2.3")
	ModelID     string    // which model (e.g., "maneuver_detector")
	ModelPath   string    // path to ONNX file
	Checksum    string    // MD5 checksum of model file
	DeployedAt  time.Time // when this version was deployed
	Description string    // human-readable version notes
}

// VersionRegistry tracks deployed model versions for validation and rollback.
type VersionRegistry struct {
	mu       sync.RWMutex
	versions map[string][]ModelVersion
}

// NewVersionRegistry creates a new version registry.
func NewVersionRegistry() *VersionRegistry {
	return &VersionRegistry{
		versions: make(map[string][]ModelVersion),
	}
}

// Register adds a new model version to the registry.
func (r *VersionRegistry) Register(version ModelVersion) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if version.ID == "" || version.ModelID == "" {
		return fmt.Errorf("version ID and model ID are required")
	}

	r.versions[version.ModelID] = append(r.versions[version.ModelID], version)
	return nil
}

// GetVersion retrieves a specific model version.
func (r *VersionRegistry) GetVersion(modelID, versionID string) (ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, exists := r.versions[modelID]
	if !exists {
		return ModelVersion{}, fmt.Errorf("model %s not found in registry", modelID)
	}

	for _, v := range versions {
		if v.ID == versionID {
			return v, nil
		}
	}

	return ModelVersion{}, fmt.Errorf("version %s not found for model %s", versionID, modelID)
}

// GetLatestVersion returns the most recently deployed version of a model.
func (r *VersionRegistry) GetLatestVersion(modelID string) (ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, exists := r.versions[modelID]
	if !exists || len(versions) == 0 {
		return ModelVersion{}, fmt.Errorf("model %s not found", modelID)
	}

	return versions[len(versions)-1], nil
}

// ListVersions returns all versions for a model.
func (r *VersionRegistry) ListVersions(modelID string) []ModelVersion {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, exists := r.versions[modelID]
	if !exists {
		return []ModelVersion{}
	}

	result := make([]ModelVersion, len(versions))
	copy(result, versions)
	return result
}

// ComputeChecksum computes the MD5 checksum of a model file.
func ComputeChecksum(filePath string) (string, error) {
	file, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	hash := md5.Sum(file)
	return fmt.Sprintf("%x", hash), nil
}

// VersionMatch checks if a deployed version matches the expected checksum.
func (v ModelVersion) VerifyChecksum(filePath string) (bool, error) {
	actual, err := ComputeChecksum(filePath)
	if err != nil {
		return false, err
	}
	return actual == v.Checksum, nil
}
