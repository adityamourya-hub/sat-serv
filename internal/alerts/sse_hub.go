// Package alerts provides SSE (Server-Sent Events) hub for real-time alert push.
package alerts

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// SSEClient represents a connected dashboard client.
type SSEClient struct {
	ID        string
	Role      Role
	Channel   chan *WSMessage
	CreatedAt time.Time
	LastPing  time.Time
}

// SSEHub manages SSE client connections and broadcasts alerts.
type SSEHub struct {
	mu      sync.RWMutex
	clients map[string]*SSEClient
}

// NewSSEHub creates a new SSE hub.
func NewSSEHub() *SSEHub {
	return &SSEHub{
		clients: make(map[string]*SSEClient),
	}
}

// Register adds a new client and returns the client ID and channel.
func (h *SSEHub) Register(role Role) (string, <-chan *WSMessage) {
	h.mu.Lock()
	defer h.mu.Unlock()

	clientID := fmt.Sprintf("client-%d", time.Now().UnixNano())
	ch := make(chan *WSMessage, 100)

	client := &SSEClient{
		ID:        clientID,
		Role:      role,
		Channel:   ch,
		CreatedAt: time.Now(),
		LastPing:  time.Now(),
	}

	h.clients[clientID] = client

	// Start heartbeat for this client
	go h.heartbeat(clientID)

	return clientID, ch
}

// Unregister removes a client.
func (h *SSEHub) Unregister(clientID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if client, ok := h.clients[clientID]; ok {
		close(client.Channel)
		delete(h.clients, clientID)
		slog.Info("SSE client unregistered", "client_id", clientID)
	}
}

// Broadcast sends an alert to all connected clients (role-filtered).
func (h *SSEHub) Broadcast(alert *EnrichedAlert) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	msg := WSMessage{
		Type:    "ALERT",
		Payload: alert,
		SentAt:  time.Now(),
	}

	for _, client := range h.clients {
		// Role-based filtering: Commanders see all, Operators see INFO+WARNING+CRITICAL
		if !RoleCanView(client.Role, alert.Severity) {
			continue
		}

		select {
		case client.Channel <- &msg:
		default:
			// Channel full, skip this client
			slog.Warn("SSE client channel full, dropping alert", "client_id", client.ID)
		}
	}
}

// heartbeat sends periodic pings to detect stale connections.
func (h *SSEHub) heartbeat(clientID string) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		h.mu.RLock()
		client, ok := h.clients[clientID]
		if !ok {
			h.mu.RUnlock()
			return
		}
		h.mu.RUnlock()

		client.LastPing = time.Now()

		// Send heartbeat message
		heartbeat := WSMessage{
			Type:    "HEARTBEAT",
			Payload: map[string]interface{}{"ping": time.Now().Unix()},
			SentAt:  time.Now(),
		}

		select {
		case client.Channel <- &heartbeat:
		default:
			// Channel full, client may be stale
			h.Unregister(clientID)
			return
		}
	}
}

// ClientCount returns the number of connected clients.
func (h *SSEHub) ClientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// ServeSSE handles the SSE upgrade and streaming for a client.
func (h *SSEHub) ServeSSE(w http.ResponseWriter, r *http.Request, role Role) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Register client
	clientID, ch := h.Register(role)
	defer h.Unregister(clientID)

	slog.Info("SSE client connected", "client_id", clientID, "role", role)

	// Flush headers
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	// Stream alerts
	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			slog.Info("SSE client disconnected", "client_id", clientID)
			return
		case msg := <-ch:
			// Serialize message
			data, err := json.Marshal(msg)
			if err != nil {
				slog.Error("Failed to marshal SSE message", "error", err)
				continue
			}

			// Write SSE format
			fmt.Fprintf(w, "data: %s\n\n", data)

			// Flush immediately
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}
}
