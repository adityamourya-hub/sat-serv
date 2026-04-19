// Package types provides event bus structures for the SSIS engine.
package types

import "sync"

// EventBus provides a simple in-memory pub/sub for alerts.
type EventBus struct {
	subscribers map[chan *Alert]struct{}
	mu          sync.RWMutex
}

// NewEventBus creates a new event bus.
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[chan *Alert]struct{}),
	}
}

// Subscribe adds a channel to receive alerts.
func (b *EventBus) Subscribe(ch chan *Alert) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subscribers[ch] = struct{}{}
}

// Unsubscribe removes a channel from receiving alerts.
func (b *EventBus) Unsubscribe(ch chan *Alert) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.subscribers, ch)
}

// Publish sends an alert to all subscribers.
func (b *EventBus) Publish(alert *Alert) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.subscribers {
		select {
		case ch <- alert:
		default:
			// Channel full, drop alert for this subscriber
		}
	}
}
