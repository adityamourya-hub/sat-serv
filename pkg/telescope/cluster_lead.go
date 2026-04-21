// pkg/telescope/cluster_lead.go
// ADITI 4.0: Multi-Mount ZMQ/INDI Cluster Lead
//
// One M2 node (CommandBridge) acting as cluster lead for 3-5 mounts.
// Creates synthetic aperture optical array from cheap COTS telescopes.
//
// Physics: Synthetic aperture effective resolution scales with baseline separation
// (meters to km) while fractional Caputo memory kernel absorbs O(10s) wake persistence.
//
// Architecture:
//   - ClusterLead: ZMQ PUB/SUB for synchronized tasking (3-5 slave mounts)
//   - SyntheticApertureTask: FOV/role assignment per mount
//   - Sensor 1: Narrow FOV → DMD Spectroscopy (material ID)
//   - Sensor 2: Wide FOV → Active Intent Discovery (shadow goal)
//   - Sensor 3: High-cadence → ASR/MFBD reconstruction
//
// Target: <50ms sync latency, synthetic aperture > single GEODSS dish

package telescope

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/pebbe/zmq4"
)

// ClusterRole defines mount function in synthetic aperture array
type ClusterRole int

const (
	RoleUnassigned ClusterRole = iota
	RoleNarrowFOV              // Sensor 1: High-mag DMD spectroscopy
	RoleWideFOV                // Sensor 2: Active intent discovery
	RoleHighCadence            // Sensor 3: ASR/MFBD reconstruction
	RoleBackup                 // Hot standby
)

func (r ClusterRole) String() string {
	switch r {
	case RoleNarrowFOV:
		return "NARROW_FOV"
	case RoleWideFOV:
		return "WIDE_FOV"
	case RoleHighCadence:
		return "HIGH_CADENCE"
	case RoleBackup:
		return "BACKUP"
	default:
		return "UNASSIGNED"
	}
}

// MountConfig defines slave mount connection parameters
type MountConfig struct {
	MountID    string      // Unique identifier
	Endpoint   string      // ZMQ endpoint (tcp://ip:port)
	Role       ClusterRole // Assigned FOV role
	FOVDegrees float64     // FOV size in degrees
	ApertureM  float64     // Aperture in meters
	Location   [3]float64  // ECEF position [km] for baseline calculation
	Enabled    bool        // Active in cluster
	LastSeen   int64       // Unix ms timestamp
}

// SyntheticApertureTask coordinates multi-mount observation
type SyntheticApertureTask struct {
	TaskID        string       // Unique task identifier
	TargetID      string       // NORAD ID or internal designation
	Timestamp     int64        // Unix ms (synchronization anchor)
	LeadTime      float64      // ms predictive lead
	
	// Role assignments
	NarrowMount   string       // MountID for Sensor 1
	WideMount     string       // MountID for Sensor 2
	CadenceMount  string       // MountID for Sensor 3
	
	// Coordinated pointing
	Azimuth       float64      // degrees
	Elevation     float64      // degrees
	AzRate        float64      // deg/s
	ElRate        float64      // deg/s
	
	// Synchronization
	SyncDeadline  int64        // Unix ms - all mounts must ACK by this time
	Acknowledged  []string     // MountIDs that have acknowledged
	
	// Processing triggers
	TriggerDMD    bool         // Fire DMD spectroscopy on Sensor 1
	TriggerMFBD   bool         // Run MFBD on Sensor 3
}

// ClusterLead manages ZMQ PUB/SUB mesh for 3-5 slave mounts
type ClusterLead struct {
	// Identity
	LeadID        string
	NodeID        string
	
	// ZMQ sockets
	pubSocket     *zmq4.Socket  // PUB: Broadcast commands to slaves
	subSocket     *zmq4.Socket  // SUB: Receive ACKs from slaves
	
	// Slave mounts
	mounts        map[string]*MountConfig
	mountMutex    sync.RWMutex
	
	// Tasking
	activeTasks   map[string]*SyntheticApertureTask
	taskMutex     sync.RWMutex
	
	// Performance
	latencyWindow []float64     // Recent sync latencies
	maxMounts     int           // Maximum cluster size (5)
	
	// Gateway integration
	gateway       *NETRAGateway // Sovereign-to-NETRA feed
}

// NewClusterLead creates ZMQ/INDI cluster lead (one M2 node)
//
// ADITI 4.0: Single M2 Air controls synthetic aperture array
// Target: <50ms sync latency for 3-5 mounts
func NewClusterLead(leadID, nodeID string) (*ClusterLead, error) {
	// Create PUB socket for broadcasting
	pubSocket, err := zmq4.NewSocket(zmq4.PUB)
	if err != nil {
		return nil, fmt.Errorf("failed to create PUB socket: %w", err)
	}
	
	// Create SUB socket for receiving ACKs
	subSocket, err := zmq4.NewSocket(zmq4.SUB)
	if err != nil {
		pubSocket.Close()
		return nil, fmt.Errorf("failed to create SUB socket: %w", err)
	}
	
	// Subscribe to all messages
	subSocket.SetSubscribe("")
	
	cl := &ClusterLead{
		LeadID:        leadID,
		NodeID:        nodeID,
		pubSocket:     pubSocket,
		subSocket:     subSocket,
		mounts:        make(map[string]*MountConfig),
		activeTasks:   make(map[string]*SyntheticApertureTask),
		latencyWindow: make([]float64, 0, 100),
		maxMounts:     5, // 3-5 mount cluster
	}
	
	return cl, nil
}

// Bind starts listening on specified endpoints
func (cl *ClusterLead) Bind(pubEndpoint, subEndpoint string) error {
	// Bind PUB socket for broadcasting
	if err := cl.pubSocket.Bind(pubEndpoint); err != nil {
		return fmt.Errorf("failed to bind PUB socket: %w", err)
	}
	
	// Bind SUB socket for receiving ACKs
	if err := cl.subSocket.Bind(subEndpoint); err != nil {
		return fmt.Errorf("failed to bind SUB socket: %w", err)
	}
	
	fmt.Printf("[CLUSTER] Lead %s listening:\n", cl.LeadID)
	fmt.Printf("  PUB: %s (commands to slaves)\n", pubEndpoint)
	fmt.Printf("  SUB: %s (ACKs from slaves)\n", subEndpoint)
	
	// Start ACK listener
	go cl.ackListener()
	
	return nil
}

// RegisterMount adds slave mount to cluster
func (cl *ClusterLead) RegisterMount(config *MountConfig) error {
	cl.mountMutex.Lock()
	defer cl.mountMutex.Unlock()
	
	// Check cluster capacity
	if len(cl.mounts) >= cl.maxMounts {
		return fmt.Errorf("cluster at capacity (%d/%d mounts)", len(cl.mounts), cl.maxMounts)
	}
	
	// Check for duplicate
	if _, exists := cl.mounts[config.MountID]; exists {
		return fmt.Errorf("mount %s already registered", config.MountID)
	}
	
	// Connect SUB socket to mount's PUB endpoint
	if err := cl.subSocket.Connect(config.Endpoint); err != nil {
		return fmt.Errorf("failed to connect to mount %s: %w", config.MountID, err)
	}
	
	// Store config
	config.LastSeen = time.Now().UnixMilli()
	cl.mounts[config.MountID] = config
	
	fmt.Printf("[CLUSTER] Registered mount %s at %s (role: %s)\n",
		config.MountID, config.Endpoint, config.Role.String())
	
	return nil
}

// UnregisterMount removes mount from cluster
func (cl *ClusterLead) UnregisterMount(mountID string) {
	cl.mountMutex.Lock()
	defer cl.mountMutex.Unlock()
	
	if config, exists := cl.mounts[mountID]; exists {
		config.Enabled = false
		delete(cl.mounts, mountID)
		fmt.Printf("[CLUSTER] Unregistered mount %s\n", mountID)
	}
}

// AssignRoles optimally distributes FOV roles across mounts
//
// Strategy:
//   - Highest aperture → Narrow FOV (Sensor 1)
//   - Widest FOV → Wide FOV (Sensor 2)
//   - Fastest camera → High-cadence (Sensor 3)
func (cl *ClusterLead) AssignRoles() {
	cl.mountMutex.Lock()
	defer cl.mountMutex.Unlock()
	
	if len(cl.mounts) < 3 {
		fmt.Printf("[CLUSTER] WARNING: Only %d mounts, need 3+ for optimal roles\n", len(cl.mounts))
	}
	
	var mounts []*MountConfig
	for _, m := range cl.mounts {
		if m.Enabled {
			mounts = append(mounts, m)
		}
	}
	
	// Reset roles
	for _, m := range mounts {
		m.Role = RoleUnassigned
	}
	
	// Assign based on capabilities
	if len(mounts) >= 1 {
		// Largest aperture → Narrow FOV
		largestAperture := 0.0
		var narrowMount *MountConfig
		for _, m := range mounts {
			if m.ApertureM > largestAperture {
				largestAperture = m.ApertureM
				narrowMount = m
			}
		}
		if narrowMount != nil {
			narrowMount.Role = RoleNarrowFOV
		}
	}
	
	if len(mounts) >= 2 {
		// Widest FOV → Wide FOV
		widestFOV := 0.0
		var wideMount *MountConfig
		for _, m := range mounts {
			if m.Role == RoleUnassigned && m.FOVDegrees > widestFOV {
				widestFOV = m.FOVDegrees
				wideMount = m
			}
		}
		if wideMount != nil {
			wideMount.Role = RoleWideFOV
		}
	}
	
	if len(mounts) >= 3 {
		// Remaining → High-cadence
		for _, m := range mounts {
			if m.Role == RoleUnassigned {
				m.Role = RoleHighCadence
				break // One high-cadence mount
			}
		}
	}
	
	// Print assignments
	for _, m := range mounts {
		fmt.Printf("[CLUSTER] %s assigned role: %s\n", m.MountID, m.Role.String())
	}
}

// SynchronizedSlew broadcasts coordinated task to all mounts
//
// ADITI 4.0: All mounts receive same timestamp + leadtime for coherent imaging
// Returns: latency, error
func (cl *ClusterLead) SynchronizedSlew(task *SyntheticApertureTask) (float64, error) {
	start := time.Now()
	
	cl.taskMutex.Lock()
	defer cl.taskMutex.Unlock()
	
	// Assign mounts to task
	cl.assignMountsToTask(task)
	
	// Set synchronization parameters
	task.Timestamp = time.Now().UnixMilli()
	task.SyncDeadline = task.Timestamp + 50 // 50ms deadline
	
	// Serialize task
	data, err := json.Marshal(task)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal task: %w", err)
	}
	
	// Broadcast to all mounts
	if _, err := cl.pubSocket.Send(string(data), 0); err != nil {
		return 0, fmt.Errorf("failed to broadcast: %w", err)
	}
	
	// Store task
	cl.activeTasks[task.TaskID] = task
	
	// Calculate latency
	latency := float64(time.Since(start).Milliseconds())
	cl.recordLatency(latency)
	
	fmt.Printf("[CLUSTER] Sync slew %s: Az=%.2f°, El=%.2f°, latency=%.2fms\n",
		task.TaskID, task.Azimuth, task.Elevation, latency)
	
	return latency, nil
}

// assignMountsToTask populates mount assignments based on roles
func (cl *ClusterLead) assignMountsToTask(task *SyntheticApertureTask) {
	cl.mountMutex.RLock()
	defer cl.mountMutex.RUnlock()
	
	for id, mount := range cl.mounts {
		if !mount.Enabled {
			continue
		}
		
		switch mount.Role {
		case RoleNarrowFOV:
			task.NarrowMount = id
		case RoleWideFOV:
			task.WideMount = id
		case RoleHighCadence:
			task.CadenceMount = id
		}
	}
}

// ackListener processes acknowledgments from slave mounts
func (cl *ClusterLead) ackListener() {
	for {
		// Receive ACK
		data, err := cl.subSocket.Recv(0)
		if err != nil {
			continue
		}
		
		// Parse ACK
		var ack MountACK
		if err := json.Unmarshal([]byte(data), &ack); err != nil {
			continue
		}
		
		// Update task
		cl.taskMutex.Lock()
		if task, exists := cl.activeTasks[ack.TaskID]; exists {
			if !contains(task.Acknowledged, ack.MountID) {
				task.Acknowledged = append(task.Acknowledged, ack.MountID)
				
				// Check if all mounts acknowledged
				if cl.allMountsAcknowledged(task) {
					fmt.Printf("[CLUSTER] Task %s fully acknowledged (%d mounts)\n",
						ack.TaskID, len(task.Acknowledged))
					
					// Trigger processing pipelines
					cl.triggerProcessing(task)
				}
			}
		}
		cl.taskMutex.Unlock()
		
		// Update mount heartbeat
		cl.mountMutex.Lock()
		if mount, exists := cl.mounts[ack.MountID]; exists {
			mount.LastSeen = time.Now().UnixMilli()
		}
		cl.mountMutex.Unlock()
	}
}

// MountACK represents acknowledgment from slave mount
type MountACK struct {
	TaskID    string  `json:"task_id"`
	MountID   string  `json:"mount_id"`
	Timestamp int64   `json:"timestamp"`
	Azimuth   float64 `json:"az"`
	Elevation float64 `json:"el"`
	Status    string  `json:"status"` // "ACK", "ERROR", "TIMEOUT"
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// allMountsAcknowledged checks if all assigned mounts have ACKed
func (cl *ClusterLead) allMountsAcknowledged(task *SyntheticApertureTask) bool {
	expected := 0
	if task.NarrowMount != "" {
		expected++
	}
	if task.WideMount != "" {
		expected++
	}
	if task.CadenceMount != "" {
		expected++
	}
	
	return len(task.Acknowledged) >= expected
}

// triggerProcessing fires DMD/MFBD pipelines after full ACK
func (cl *ClusterLead) triggerProcessing(task *SyntheticApertureTask) {
	// Fire DMD spectroscopy on Sensor 1
	if task.TriggerDMD && task.NarrowMount != "" {
		fmt.Printf("[CLUSTER] Triggering DMD spectroscopy on %s\n", task.NarrowMount)
		// TODO: Integrate with DMD tripwire
	}
	
	// Fire MFBD on Sensor 3
	if task.TriggerMFBD && task.CadenceMount != "" {
		fmt.Printf("[CLUSTER] Triggering MFBD reconstruction on %s\n", task.CadenceMount)
		// TODO: Integrate with VisionEngine
	}
	
	// Gateway push to NETRA
	if cl.gateway != nil {
		cl.gateway.PushSyntheticApertureObservation(task)
	}
}

// recordLatency tracks sync latency
func (cl *ClusterLead) recordLatency(latency float64) {
	cl.latencyWindow = append(cl.latencyWindow, latency)
	if len(cl.latencyWindow) > 100 {
		cl.latencyWindow = cl.latencyWindow[1:]
	}
}

// GetAverageLatency returns mean sync latency
func (cl *ClusterLead) GetAverageLatency() float64 {
	if len(cl.latencyWindow) == 0 {
		return 0.0
	}
	
	sum := 0.0
	for _, l := range cl.latencyWindow {
		sum += l
	}
	return sum / float64(len(cl.latencyWindow))
}

// GetClusterStatus returns current cluster health
func (cl *ClusterLead) GetClusterStatus() (int, float64, map[string]ClusterRole) {
	cl.mountMutex.RLock()
	defer cl.mountMutex.RUnlock()
	
	status := make(map[string]ClusterRole)
	active := 0
	
	for id, m := range cl.mounts {
		status[id] = m.Role
		if m.Enabled {
			active++
		}
	}
	
	return active, cl.GetAverageLatency(), status
}

// SetGateway connects NETRA gateway for sovereign feed
func (cl *ClusterLead) SetGateway(gw *NETRAGateway) {
	cl.gateway = gw
}

// Close shuts down cluster lead
func (cl *ClusterLead) Close() {
	cl.pubSocket.Close()
	cl.subSocket.Close()
}
