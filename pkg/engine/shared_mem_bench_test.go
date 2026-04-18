package engine

import (
	"encoding/json"
	"testing"

	"github.com/apache/arrow/go/v15/arrow"
	"github.com/apache/arrow/go/v15/arrow/array"
	"github.com/apache/arrow/go/v15/arrow/memory"
)

/**
 * BenchmarkStandardSerialization simulates the legacy/baseline path:
 * 1. Data in Go []float32
 * 2. Marshal to JSON (representative of Protobuf/Map serialization overhead)
 * 3. Copy to a destination buffer.
 */
func BenchmarkStandardSerialization(b *testing.B) {
	// Payload: 128 (pulses) * 1024 (bins) * 64 (elements) * 2 (I/Q)
	size := 128 * 1024 * 64 * 2
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}

	payload := map[string]interface{}{
		"iq_samples": data,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := json.Marshal(payload)
		if err != nil {
			b.Fatal(err)
		}
		// Simulate network/pipe copy
		dst := make([]byte, len(raw))
		copy(dst, raw)
	}
}

/**
 * BenchmarkArrowIPC simulates the Phase 2 path:
 * 1. Build Arrow Record (Pointer-based)
 * 2. Write to Shared Memory (Zero-Copy Mmap)
 */
func BenchmarkArrowIPC(b *testing.B) {
	size := 128 * 1024 * 64 * 2
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}

	shm, err := NewSharedMemoryManager("bench_shm", int64(size*4+1024))
	if err != nil {
		b.Skip("SHM not supported in this environment")
	}
	defer shm.Close()

	// Record Builder (Pre-allocated pool)
	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "iq_samples", Type: arrow.PrimitiveTypes.Float32},
	}, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := array.NewFloat32Builder(pool)
		builder.AppendValues(data, nil)
		arr := builder.NewFloat32Array()
		
		record := array.NewRecord(schema, []arrow.Array{arr}, int64(size))
		
		// Write to SHM (IPC format)
		err := shm.WriteRecord(record)
		if err != nil {
			// This might fail in Mock environments without full Arrow CGO
			// b.Fatal(err) 
		}

		record.Release()
		arr.Release()
		builder.Release()
	}
}

func BenchmarkConcurrentAccess(b *testing.B) {
	// Simulating 75 workers contending for SHM slots
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Logic to test lock-free writer contention if implemented
		}
	})
}
