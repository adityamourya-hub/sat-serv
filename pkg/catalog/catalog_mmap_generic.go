//go:build !linux
// +build !linux

package catalog

// msync is a no-op on platforms without native msync support
// Data is synced automatically on munmap or file close
func msync(data []byte, flags int) error {
	return nil
}

const (
	msAsync   = 0
	msSync    = 0
	msInvalid = 0
)
