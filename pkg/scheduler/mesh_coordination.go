package scheduler

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/pebbe/zmq4"
)

/**
 * Mesh Coordination Protocol for Distributed Constellations
 * 
 * Implements Reticulum-inspired message passing for sovereign mesh networks.
 * Enables autonomous negotiation between constellation nodes for task assignment.
 */

// Message types for mesh coordination
const (
	MsgTypeTaskOffer    = "TASK_OFFER"
	MsgTypeTaskAccept   = "TASK_ACCEPT"
	MsgTypeTaskReject   = "TASK_REJECT"
	MsgTypeScheduleSync = "SCHEDULE_SYNC"
	MsgTypeHeartbeat    = "HEARTBEAT"
)

// MeshMessage is the base protocol message
type MeshMessage struct {
	Type      string    `json:"type"`
	FromNode  string    `json:"from_node"`
	ToNode    string    `json:"to_node"`  // Empty = broadcast
	Timestamp time.Time `json:"timestamp"`
	Payload   json.RawMessage `json:"payload"`
}

// TaskOfferMessage proposes a task to the mesh
type TaskOfferMessage struct {
	TaskID      int       `json:"task_id"`
	SensorID    string    `json:"sensor_id"`
	TargetID    int       `json:"target_id"`
	StartTime   time.Time `json:"start_time"`
	Duration    float64   `json:"duration_sec"`
	Priority    float64   `json:"priority"`
	MinElevation float64  `json:"min_elevation"`
}

// TaskResponseMessage accepts or rejects a task offer
type TaskResponseMessage struct {
	TaskID   int    `json:"task_id"`
	Accepted bool   `json:"accepted"`
	Reason   string `json:"reason,omitempty"`
}

// MeshCoordinator manages distributed task negotiation
type MeshCoordinator struct {
	nodeID     string
	network    string  // Mesh network ID
	socket     *zmq4.Socket
	solver     *MISSolver
	
	// Local state
	localTasks   map[int]*TaskNode
	acceptedTasks map[int]*TaskNode
	neighborState map[string]time.Time  // Last seen
}

// NewMeshCoordinator creates a mesh-aware coordinator
func NewMeshCoordinator(nodeID, network string, endpoint string) (*MeshCoordinator, error) {
	socket, err := zmq4.NewSocket(zmq4.DEALER)
	if err != nil {
		return nil, fmt.Errorf("failed to create socket: %w", err)
	}
	
	// Set identity
	socket.SetIdentity(nodeID)
	
	// Connect to mesh router (Reticulum-style)
	if err := socket.Connect(endpoint); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to connect to mesh: %w", err)
	}
	
	return &MeshCoordinator{
		nodeID:        nodeID,
		network:       network,
		socket:        socket,
		solver:        NewMISSolver("weighted"),
		localTasks:    make(map[int]*TaskNode),
		acceptedTasks: make(map[int]*TaskNode),
		neighborState: make(map[string]time.Time),
	}, nil
}

// Run starts the mesh coordination loop
func (mc *MeshCoordinator) Run() {
	// Start heartbeat
	go mc.heartbeatLoop()
	
	// Message processing loop
	for {
		msg, err := mc.receiveMessage()
		if err != nil {
			continue
		}
		
		mc.processMessage(msg)
	}
}

// heartbeatLoop sends periodic heartbeats to mesh
func (mc *MeshCoordinator) heartbeatLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		heartbeat := MeshMessage{
			Type:      MsgTypeHeartbeat,
			FromNode:  mc.nodeID,
			Timestamp: time.Now(),
			Payload:   json.RawMessage(`{"status":"alive"}`),
		}
		mc.broadcastMessage(heartbeat)
	}
}

// receiveMessage polls for incoming mesh messages
func (mc *MeshCoordinator) receiveMessage() (*MeshMessage, error) {
	data, err := mc.socket.Recv(0)
	if err != nil {
		return nil, err
	}
	
	var msg MeshMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return nil, err
	}
	
	return &msg, nil
}

// processMessage handles incoming mesh coordination messages
func (mc *MeshCoordinator) processMessage(msg *MeshMessage) {
	// Update neighbor state
	mc.neighborState[msg.FromNode] = time.Now()
	
	switch msg.Type {
	case MsgTypeTaskOffer:
		mc.handleTaskOffer(msg)
	case MsgTypeTaskAccept:
		mc.handleTaskAccept(msg)
	case MsgTypeTaskReject:
		mc.handleTaskReject(msg)
	case MsgTypeScheduleSync:
		mc.handleScheduleSync(msg)
	case MsgTypeHeartbeat:
		// Just updated neighbor state
	}
}

// handleTaskOffer evaluates and responds to task proposals
func (mc *MeshCoordinator) handleTaskOffer(msg *MeshMessage) {
	var offer TaskOfferMessage
	if err := json.Unmarshal(msg.Payload, &offer); err != nil {
		return
	}
	
	// Convert to internal TaskNode
	task := &TaskNode{
		ID:           offer.TaskID,
		Priority:     offer.Priority,
		SensorID:     offer.SensorID,
		TargetID:     offer.TargetID,
		StartTime:    offer.StartTime,
		Duration:     time.Duration(offer.Duration * float64(time.Second)),
		MinElevation: offer.MinElevation,
	}
	
	// Check if we can accept (local capacity, no conflicts)
	canAccept := mc.evaluateTaskAcceptance(task)
	
	response := TaskResponseMessage{
		TaskID:   offer.TaskID,
		Accepted: canAccept,
	}
	
	if !canAccept {
		response.Reason = "capacity_full"
	} else {
		// Reserve the task
		mc.acceptedTasks[offer.TaskID] = task
	}
	
	// Send response
	respMsg := MeshMessage{
		Type:      MsgTypeTaskAccept,
		FromNode:  mc.nodeID,
		ToNode:    msg.FromNode,
		Timestamp: time.Now(),
	}
	respPayload, _ := json.Marshal(response)
	respMsg.Payload = respPayload
	
	mc.sendMessage(respMsg)
}

// evaluateTaskAcceptance checks if this node can take the task
func (mc *MeshCoordinator) evaluateTaskAcceptance(task *TaskNode) bool {
	// Check capacity (max 10 concurrent tasks per node)
	if len(mc.acceptedTasks) >= 10 {
		return false
	}
	
	// Check for conflicts with already accepted tasks
	for _, accepted := range mc.acceptedTasks {
		if timesOverlap(task, accepted) {
			return false
		}
	}
	
	// Check sensor availability (simplified)
	// In production, check actual sensor schedule
	
	return true
}

// handleTaskAccept processes acceptance confirmations
func (mc *MeshCoordinator) handleTaskAccept(msg *MeshMessage) {
	var response TaskResponseMessage
	if err := json.Unmarshal(msg.Payload, &response); err != nil {
		return
	}
	
	if response.Accepted {
		// Task assigned to this node
		fmt.Printf("Task %d accepted by %s\n", response.TaskID, msg.FromNode)
	}
}

// handleTaskReject processes rejections (reassign to other nodes)
func (mc *MeshCoordinator) handleTaskReject(msg *MeshMessage) {
	var response TaskResponseMessage
	if err := json.Unmarshal(msg.Payload, &response); err != nil {
		return
	}
	
	fmt.Printf("Task %d rejected by %s: %s\n", response.TaskID, msg.FromNode, response.Reason)
}

// handleScheduleSync synchronizes global schedule state
func (mc *MeshCoordinator) handleScheduleSync(msg *MeshMessage) {
	// In distributed mode, periodically sync MIS solution across nodes
	// This ensures all nodes agree on the conflict-free schedule
}

// ProposeTasks initiates distributed task negotiation
func (mc *MeshCoordinator) ProposeTasks(tasks []*TaskNode) error {
	for _, task := range tasks {
		offer := TaskOfferMessage{
			TaskID:       task.ID,
			SensorID:     task.SensorID,
			TargetID:     task.TargetID,
			StartTime:    task.StartTime,
			Duration:     task.Duration.Seconds(),
			Priority:     task.Priority,
			MinElevation: task.MinElevation,
		}
		
		payload, _ := json.Marshal(offer)
		msg := MeshMessage{
			Type:      MsgTypeTaskOffer,
			FromNode:  mc.nodeID,
			Timestamp: time.Now(),
			Payload:   payload,
		}
		
		mc.broadcastMessage(msg)
	}
	
	return nil
}

// sendMessage sends to specific node
func (mc *MeshCoordinator) sendMessage(msg MeshMessage) error {
	data, _ := json.Marshal(msg)
	_, err := mc.socket.Send(string(data), 0)
	return err
}

// broadcastMessage sends to all mesh nodes
func (mc *MeshCoordinator) broadcastMessage(msg MeshMessage) error {
	// In Reticulum, broadcast uses special destination
	msg.ToNode = ""  // Broadcast
	return mc.sendMessage(msg)
}

// GetMeshState returns current mesh topology
func (mc *MeshCoordinator) GetMeshState() map[string]time.Time {
	return mc.neighborState
}

// Close shuts down mesh connection
func (mc *MeshCoordinator) Close() {
	mc.socket.Close()
}
