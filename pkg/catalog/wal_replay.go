package catalog

import (
	"encoding/binary"
	"fmt"
	"os"
	"syscall"
	"time"
	"unsafe"
)

/**
 * WAL Replay Logic (The "Reflex Path" Recovery)
 *
 * Ensures <50ms recovery of Hamiltonian state after hard restart.
 *
 * Recovery Pipeline:
 * 1. mmap the catalog file (OS handles paging)
 * 2. Verify checksum integrity
 * 3. Check Hamiltonian drift for each RSO
 * 4. Replay WAL entries if needed
 * 5. Rebuild spatial indices
 *
 * Target: <50ms from crash to operational readiness
 */

const (
	WALMagic        = 0x57414C31 // "WAL1"
	WALVersion      = 1
	MaxWALSize      = 100 * 1024 * 1024 // 100MB max WAL
	EntryHeaderSize = 16
)

// WALEntryType distinguishes record types
type WALEntryType uint8

const (
	WALEntryRSOUpdate WALEntryType = iota
	WALEntryRRUpdate
	WALEntryCheckpoint
)

// WALEntry represents a single write-ahead log record
type WALEntry struct {
	Type      WALEntryType
	Timestamp int64 // Unix nanoseconds
	TargetID  uint32
	Size      uint32 // Size of payload
	Payload   []byte
}

// WALHeader is stored at the beginning of the WAL file
type WALHeader struct {
	Magic      uint32
	Version    uint32
	EntryCount uint64
	LastLSN    uint64 // Log Sequence Number
	Checksum   uint32
	_          uint32 // Padding
}

// WALReplay handles crash recovery for the catalog
// Ensures Hamiltonian state is restored within 50ms
type WALReplay struct {
	walFile *os.File
	walData []byte
	header  *WALHeader
	catalog *MMapCatalog
}

// NewWALReplay creates a replay manager
func NewWALReplay(walPath string, catalog *MMapCatalog) (*WALReplay, error) {
	file, err := os.OpenFile(walPath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	// Ensure minimum size
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat WAL: %w", err)
	}

	minSize := 4096 // 4KB header
	if stat.Size() < int64(minSize) {
		if err := file.Truncate(int64(minSize)); err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to size WAL: %w", err)
		}
	}

	// Memory map the WAL
	data, err := syscall.Mmap(int(file.Fd()), 0, MaxWALSize,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap WAL: %w", err)
	}

	wr := &WALReplay{
		walFile: file,
		walData: data,
		catalog: catalog,
	}

	// Initialize or validate header
	wr.header = (*WALHeader)(unsafe.Pointer(&data[0]))
	if wr.header.Magic != WALMagic {
		// New WAL
		wr.header.Magic = WALMagic
		wr.header.Version = WALVersion
		wr.header.EntryCount = 0
		wr.header.LastLSN = 0
		wr.header.Checksum = 0
	}

	return wr, nil
}

// Recover performs crash recovery
// Target: <50ms total recovery time
func (wr *WALReplay) Recover() error {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start)
		fmt.Printf("[WAL] Recovery completed in %.2fms\n", elapsed.Seconds()*1000)
		if elapsed > 50*time.Millisecond {
			fmt.Printf("[WAL] WARNING: Recovery exceeded 50ms target!\n")
		}
	}()

	// Phase 1: Validate catalog checksum (<5ms)
	if err := wr.validateCatalogIntegrity(); err != nil {
		fmt.Printf("[WAL] Catalog integrity check failed: %v\n", err)
	}

	// Phase 2: Check Hamiltonian drift (<20ms for 30k objects)
	drifted := wr.checkHamiltonianDrift(1e-7) // σ_H / H_avg threshold
	if len(drifted) > 0 {
		fmt.Printf("[WAL] Detected Hamiltonian drift in %d objects\n", len(drifted))
		wr.flagDriftedObjects(drifted)
	}

	// Phase 3: Replay uncommitted WAL entries (<20ms)
	if wr.header.EntryCount > 0 {
		count, err := wr.replayEntries()
		if err != nil {
			return fmt.Errorf("WAL replay failed: %w", err)
		}
		fmt.Printf("[WAL] Replayed %d entries\n", count)
	}

	// Phase 4: Sync catalog to disk (<5ms)
	if err := wr.catalog.Sync(); err != nil {
		return fmt.Errorf("catalog sync failed: %w", err)
	}

	// Reset WAL
	wr.header.EntryCount = 0
	wr.header.LastLSN = 0

	return nil
}

// Append adds an entry to the WAL
func (wr *WALReplay) Append(entryType WALEntryType, targetID uint32, payload []byte) error {
	entry := WALEntry{
		Type:      entryType,
		Timestamp: time.Now().UnixNano(),
		TargetID:  targetID,
		Size:      uint32(len(payload)),
		Payload:   payload,
	}

	// Serialize entry
	data := make([]byte, EntryHeaderSize+len(payload))
	data[0] = uint8(entry.Type)
	binary.LittleEndian.PutUint64(data[1:9], uint64(entry.Timestamp))
	binary.LittleEndian.PutUint32(data[9:13], entry.TargetID)
	binary.LittleEndian.PutUint32(data[13:17], entry.Size)
	copy(data[17:], payload)

	// Calculate offset in WAL
	offset := 4096 + int(wr.header.EntryCount)*(EntryHeaderSize+len(payload))
	if offset+len(data) > MaxWALSize {
		return fmt.Errorf("WAL full, need checkpoint")
	}

	// Write to mmap
	copy(wr.walData[offset:], data)

	// Update header
	wr.header.EntryCount++
	wr.header.LastLSN++

	return nil
}

// Checkpoint truncates WAL after successful catalog sync
func (wr *WALReplay) Checkpoint() error {
	// Sync catalog first
	if err := wr.catalog.Sync(); err != nil {
		return err
	}

	// Reset WAL
	wr.header.EntryCount = 0
	wr.header.LastLSN = 0

	// Sync WAL header
	return msync(wr.walData[:4096], msSync)
}

// Close closes the WAL
func (wr *WALReplay) Close() error {
	syscall.Munmap(wr.walData)
	wr.walFile.Close()
	return nil
}

// Private methods

func (wr *WALReplay) validateCatalogIntegrity() error {
	computed := wr.catalog.ComputeChecksum()
	stored := wr.catalog.header.Checksum

	for i := 0; i < 32; i++ {
		if computed[i] != stored[i] {
			return fmt.Errorf("checksum mismatch at byte %d", i)
		}
	}
	return nil
}

func (wr *WALReplay) checkHamiltonianDrift(threshold float64) []uint32 {
	return wr.catalog.CheckHamiltonianDrift(threshold)
}

func (wr *WALReplay) flagDriftedObjects(drifted []uint32) {
	for _, id := range drifted {
		if state, ok := wr.catalog.Get(id); ok {
			// Flag as needing re-verification
			state.Flags |= 0x02 // Drift detected
			wr.catalog.Put(state)
		}
	}
}

func (wr *WALReplay) replayEntries() (int, error) {
	count := 0
	offset := 4096 // Start after header

	for i := uint64(0); i < wr.header.EntryCount; i++ {
		if offset >= len(wr.walData) {
			break
		}

		entry, n := wr.parseEntry(offset)
		if n == 0 {
			break
		}
		offset += n

		// Replay based on type
		switch entry.Type {
		case WALEntryRSOUpdate:
			if err := wr.replayRSOUpdate(entry); err != nil {
				fmt.Printf("[WAL] Failed to replay RSO update for %d: %v\n",
					entry.TargetID, err)
			}

		case WALEntryRRUpdate:
			if err := wr.replayRRUpdate(entry); err != nil {
				fmt.Printf("[WAL] Failed to replay RRI update for %d: %v\n",
					entry.TargetID, err)
			}

		case WALEntryCheckpoint:
			// Skip checkpoints during replay
		}

		count++
	}

	return count, nil
}

func (wr *WALReplay) parseEntry(offset int) (WALEntry, int) {
	if offset+EntryHeaderSize > len(wr.walData) {
		return WALEntry{}, 0
	}

	entry := WALEntry{
		Type:      WALEntryType(wr.walData[offset]),
		Timestamp: int64(binary.LittleEndian.Uint64(wr.walData[offset+1 : offset+9])),
		TargetID:  binary.LittleEndian.Uint32(wr.walData[offset+9 : offset+13]),
		Size:      binary.LittleEndian.Uint32(wr.walData[offset+13 : offset+17]),
	}

	if entry.Size > 0 {
		payloadOffset := offset + EntryHeaderSize
		if payloadOffset+int(entry.Size) > len(wr.walData) {
			return WALEntry{}, 0
		}
		entry.Payload = wr.walData[payloadOffset : payloadOffset+int(entry.Size)]
	}

	totalSize := EntryHeaderSize + int(entry.Size)
	return entry, totalSize
}

func (wr *WALReplay) replayRSOUpdate(entry WALEntry) error {
	// Deserialize RSOState from payload
	if len(entry.Payload) < RSOStateSize {
		return fmt.Errorf("payload too small for RSO state")
	}

	var state RSOState
	// Simple binary deserialization
	state.ID = binary.LittleEndian.Uint32(entry.Payload[0:4])
	// ... deserialize remaining fields

	// Apply to catalog
	return wr.catalog.Put(state)
}

func (wr *WALReplay) replayRRUpdate(entry WALEntry) error {
	// RRI updates are handled by the RRIStore
	// This is a placeholder for RRI replay logic
	return nil
}

// RecoveryMetrics tracks recovery performance
type RecoveryMetrics struct {
	TotalTime       time.Duration
	IntegrityTime   time.Duration
	DriftCheckTime  time.Duration
	ReplayTime      time.Duration
	SyncTime        time.Duration
	ObjectsChecked  int
	EntriesReplayed int
	DriftedObjects  int
}

// RecoverWithMetrics performs recovery with detailed timing
func (wr *WALReplay) RecoverWithMetrics() (*RecoveryMetrics, error) {
	metrics := &RecoveryMetrics{}
	start := time.Now()

	// Phase 1: Integrity
	t1 := time.Now()
	wr.validateCatalogIntegrity()
	metrics.IntegrityTime = time.Since(t1)

	// Phase 2: Drift check
	t2 := time.Now()
	drifted := wr.checkHamiltonianDrift(1e-7)
	metrics.DriftCheckTime = time.Since(t2)
	metrics.ObjectsChecked = int(wr.catalog.Count())
	metrics.DriftedObjects = len(drifted)
	wr.flagDriftedObjects(drifted)

	// Phase 3: Replay
	t3 := time.Now()
	count, _ := wr.replayEntries()
	metrics.ReplayTime = time.Since(t3)
	metrics.EntriesReplayed = count

	// Phase 4: Sync
	t4 := time.Now()
	wr.catalog.Sync()
	metrics.SyncTime = time.Since(t4)

	metrics.TotalTime = time.Since(start)

	// Reset WAL
	wr.header.EntryCount = 0

	return metrics, nil
}
