package physics

import (
	"sync"
	"testing"
)

func TestNewStatePool(t *testing.T) {
	pool := NewStatePool()
	if pool == nil {
		t.Fatal("NewStatePool returned nil")
	}
}

func TestStatePool_GetPut(t *testing.T) {
	pool := NewStatePool()
	
	// Get a state
	state := pool.Get()
	if state == nil {
		t.Fatal("Get returned nil")
	}
	
	// Modify state
	state[0] = 6778.0
	state[1] = 0.0
	state[2] = 0.0
	state[3] = 7.5
	state[4] = 0.0
	state[5] = 0.0
	
	// Return to pool
	pool.Put(state)
	
	// Get another state - should be zeroed
	state2 := pool.Get()
	if state2[0] != 0 {
		t.Errorf("State not zeroed after Put: got %.2f, expected 0", state2[0])
	}
	
	pool.Put(state2)
}

func TestStatePool_ConcurrentAccess(t *testing.T) {
	pool := NewStatePool()
	var wg sync.WaitGroup
	
	// Concurrent access test
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			state := pool.Get()
			state[0] = float64(i)
			pool.Put(state)
		}()
	}
	
	wg.Wait()
	t.Log("Concurrent access test passed")
}

func TestStatePool_GetMany(t *testing.T) {
	pool := NewStatePool()
	
	// Get 10 states
	states := pool.GetMany(10)
	if len(states) != 10 {
		t.Errorf("GetMany returned %d states, expected 10", len(states))
	}
	
	// Modify all states
	for i, state := range states {
		state[0] = float64(i)
	}
	
	// Return all states
	pool.PutMany(states)
	
	// Verify states are zeroed
	states2 := pool.GetMany(10)
	for _, state := range states2 {
		if state[0] != 0 {
			t.Errorf("State not zeroed after PutMany: got %.2f, expected 0", state[0])
		}
	}
	
	pool.PutMany(states2)
}

func TestGlobalPool(t *testing.T) {
	state := GetState()
	if state == nil {
		t.Fatal("GetState returned nil")
	}
	
	state[0] = 100.0
	PutState(state)
	
	state2 := GetState()
	if state2[0] != 0 {
		t.Errorf("Global state not zeroed: got %.2f, expected 0", state2[0])
	}
	
	PutState(state2)
}

func BenchmarkStatePool_GetPut(b *testing.B) {
	pool := NewStatePool()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state := pool.Get()
		state[0] = 1.0
		pool.Put(state)
	}
}

func BenchmarkStatePool_GetMany(b *testing.B) {
	pool := NewStatePool()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		states := pool.GetMany(10)
		for _, state := range states {
			state[0] = 1.0
		}
		pool.PutMany(states)
	}
}
