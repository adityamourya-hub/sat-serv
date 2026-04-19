package persistence

import (
	"fmt"
	"log"
	"sync"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/cryptography"
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
	Signer       *cryptography.Signer
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
	if err != nil {
		log.Printf("[PERSISTENCE WARN] Failed to load snapshot from %s: %v", cfg.Backend, err)
	} else if len(records) > 0 {
		validRecords := []catalog.SDAObject{}
		sigFailures := 0
		for _, r := range records {
			// Optional: Verify Ed25519 signature before loading
			// payload := fmt.Sprintf("%d|%s|%f", r.ID, r.TLELine1, r.Timestamp)
			// if cfg.Signer != nil && !cfg.Signer.Verify([]byte(payload), r.Signature) { sigFailures++; continue }
			validRecords = append(validRecords, r)
		}
		if sigFailures > 0 {
			log.Printf("[PERSISTENCE WARN] Signature verification failed for %d records", sigFailures)
		}
		log.Printf("[PERSISTENCE] Restored %d objects from %s", len(validRecords), cfg.Backend)
		ps.Store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: validRecords}, true)
	}

	// 2. Setup persistence hook
	ps.Store.PersistFunc = ps.onUpdate

	return ps, nil
}

func (ps *PersistentStore) onUpdate(delta catalog.Delta) {
	// Sync persistence: mirror memory updates to DB
	var upsertErrors, deleteErrors int
	for _, rec := range delta.Upserts {
		if err := ps.backend.UpsertObject(rec); err != nil {
			upsertErrors++
			log.Printf("[PERSISTENCE ERROR] Failed to upsert object %d: %v", rec.ID, err)
		}
	}
	for _, id := range delta.Deletes {
		if err := ps.backend.DeleteObject(id); err != nil {
			deleteErrors++
			log.Printf("[PERSISTENCE ERROR] Failed to delete object %d: %v", id, err)
		}
	}
	if upsertErrors > 0 || deleteErrors > 0 {
		log.Printf("[PERSISTENCE WARN] Persistence sync completed with %d upsert errors, %d delete errors",
			upsertErrors, deleteErrors)
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
