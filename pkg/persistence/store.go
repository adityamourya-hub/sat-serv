package persistence

import (
	"fmt"
	"sync"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/crypto"
)

// PersistenceBackend defines the interface for database implementations.
type PersistenceBackend interface {
	Close() error
	UpsertObject(record catalog.SDAObject) error
	DeleteObject(id int) error
	GetLatestSnapshot() ([]catalog.SDAObject, error)
	GetObjectAtTime(id int, t time.Time) (catalog.SDAObject, error)
	QueryObjectsInWindow(start, end time.Time) ([]catalog.SDAObject, error)
	PersistHypothesis(hyp *catalog.GlobalHypothesis) error
}

// Config defines the persistence parameters.
type Config struct {
	Backend      string // "boltdb" or "sqlite"
	Path         string
	SyncInterval time.Duration
	MaxHistory   int
	Signer       *crypto.Signer
}

// PersistentStore wraps the in-memory store with a persistence backend.
type PersistentStore struct {
	*catalog.Store
	backend PersistenceBackend
	config  Config
	mu      sync.RWMutex
}

// NewPersistentStore initializes a new persistent catalog.
func NewPersistentStore(cfg Config, cellSize float64) (*PersistentStore, error) {
	var backend PersistenceBackend
	var err error

	switch cfg.Backend {
	case "boltdb", "":
		backend, err = NewBoltDBBackend(cfg.Path)
	case "sqlite":
		backend, err = NewSQLiteBackend(cfg.Path)
	default:
		return nil, fmt.Errorf("unsupported persistence backend: %s", cfg.Backend)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize backend: %w", err)
	}

	ps := &PersistentStore{
		Store:   catalog.NewStore(cellSize),
		backend: backend,
		config:  cfg,
	}

	// 1. Reload latest snapshot from DB
	records, err := backend.GetLatestSnapshot()
	if err == nil && len(records) > 0 {
		validRecords := []catalog.SDAObject{}
		for _, r := range records {
			// Optional: Verify Ed25519 signature before loading
			// payload := fmt.Sprintf("%d|%s|%f", r.ID, r.TLELine1, r.Timestamp)
			// if cfg.Signer != nil && !cfg.Signer.Verify([]byte(payload), r.Signature) { continue }
			validRecords = append(validRecords, r)
		}
		ps.Store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: validRecords}, true)
	}

	// 2. Setup persistence hook
	ps.Store.PersistFunc = ps.onUpdate

	return ps, nil
}

func (ps *PersistentStore) onUpdate(delta catalog.Delta) {
	// Sync persistence: mirror memory updates to DB
	for _, rec := range delta.Upserts {
		ps.backend.UpsertObject(rec)
	}
	for _, id := range delta.Deletes {
		ps.backend.DeleteObject(id)
	}
}

func (ps *PersistentStore) Close() error {
	return ps.backend.Close()
}

// Historical & Time-Range Query Passthrough
func (ps *PersistentStore) GetObjectAtTime(id int, t time.Time) (catalog.SDAObject, error) {
	return ps.backend.GetObjectAtTime(id, t)
}

func (ps *PersistentStore) QueryObjectsInWindow(start, end time.Time) ([]catalog.SDAObject, error) {
	return ps.backend.QueryObjectsInWindow(start, end)
}
