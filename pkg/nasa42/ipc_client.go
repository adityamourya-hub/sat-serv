package nasa42

/**
 * NASA 42 IPC Client - Go to C Communication Bridge
 *
 * ADITI 4.0: Connects Go orchestration layer to NASA 42 simulator
 * for high-fidelity physics validation.
 *
 * Architecture:
 * - Go: 100k-object catalog, BVH screening, conjunction detection
 * - NASA 42: High-fidelity physics (Kane's Equations, flexible bodies)
 * - IPC: Unix domain sockets for zero-copy state exchange
 *
 * Protocol:
 * - Binary telemetry packets (little-endian, C struct compatible)
 * - Go sends: Initial state vectors for high-interest objects
 * - 42 returns: Propagated states with full 6-DOF dynamics
 *
 * Use Case:
 * 1. Detect high-risk conjunction in Go (Pc > 1e-4)
 * 2. Hand off object pair to NASA 42 via IPC
 * 3. Get NASA-validated high-fidelity trajectory
 * 4. Update ADITI with "ground truth" for critical events
 *
 * References:
 * - NASA 42: github.com/ericstoneking/42
 * - 42 IPC module: InOut/Inp_IPC.txt
 * - Kane's Equations: Multi-body dynamics without Lagrange multipliers
 */

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"sync"
	"time"
)

// IPCConfig defines connection parameters
type IPCConfig struct {
	Host       string        // Server hostname (localhost for local)
	Port       int           // Server port (4242 default)
	SocketType string        // "tcp" or "unix"
	Timeout    time.Duration // Connection timeout
}

// DefaultIPCConfig returns standard configuration
func DefaultIPCConfig() *IPCConfig {
	return &IPCConfig{
		Host:       "localhost",
		Port:       4242,
		SocketType: "tcp",
		Timeout:    5 * time.Second,
	}
}

// TelemetryPacket matches NASA 42 SCType binary layout
// This must be kept in sync with 42's C struct definition
// Size: ~128 bytes (depending on alignment)
//
// Padding assumption: 7-byte padding after GpsValid to align GPS position
// to 8-byte boundary. This assumes C compiler uses default alignment.
// Verify with: pack(pragma) or manual alignment in C code.
type TelemetryPacket struct {
	// Time
	Time float64 // Simulation time (seconds from epoch)

	// Position (km) - GCRF frame
	PosX float64
	PosY float64
	PosZ float64

	// Velocity (km/s) - GCRF frame
	VelX float64
	VelY float64
	VelZ float64

	// Attitude quaternion (q0, q1, q2, q3) - body to inertial
	Q0 float64
	Q1 float64
	Q2 float64
	Q3 float64

	// Angular velocity (rad/s) - body frame
	Wx float64
	Wy float64
	Wz float64

	// Flexible body states (simplified - first 3 modes)
	Flex1 float64
	Flex2 float64
	Flex3 float64

	// Mass properties (kg, kg*m^2)
	Mass float64
	Ixx  float64
	Iyy  float64
	Izz  float64
	Ixy  float64
	Ixz  float64
	Iyz  float64

	// Environmental data
	DragAccel  float64 // m/s^2
	SRPAccel   float64 // m/s^2
	GravityAcc float64 // m/s^2 (magnitude)
	SunAngle   float64 // degrees (solar phase)

	// Control/sensor data
	ThrusterOn uint32 // Bitmask
	GpsValid   uint8
	GpsPosX    float64 // m
	GpsPosY    float64
	GpsPosZ    float64
}

// Size returns binary size of packet
func (tp *TelemetryPacket) Size() int {
	return binary.Size(*tp)
}

// ToBytes serializes packet to binary (little-endian)
func (tp *TelemetryPacket) ToBytes() ([]byte, error) {
	buf := make([]byte, tp.Size())

	// Manual serialization for exact byte alignment
	offset := 0

	// Helper to write float64
	writeFloat64 := func(f float64) {
		binary.LittleEndian.PutUint64(buf[offset:], uint64(f))
		offset += 8
	}

	// Helper to write uint32
	writeUint32 := func(u uint32) {
		binary.LittleEndian.PutUint32(buf[offset:], u)
		offset += 4
	}

	// Helper to write uint8
	writeUint8 := func(u uint8) {
		buf[offset] = u
		offset += 1
	}

	// Time
	writeFloat64(tp.Time)

	// Position
	writeFloat64(tp.PosX)
	writeFloat64(tp.PosY)
	writeFloat64(tp.PosZ)

	// Velocity
	writeFloat64(tp.VelX)
	writeFloat64(tp.VelY)
	writeFloat64(tp.VelZ)

	// Quaternion
	writeFloat64(tp.Q0)
	writeFloat64(tp.Q1)
	writeFloat64(tp.Q2)
	writeFloat64(tp.Q3)

	// Angular velocity
	writeFloat64(tp.Wx)
	writeFloat64(tp.Wy)
	writeFloat64(tp.Wz)

	// Flexible body
	writeFloat64(tp.Flex1)
	writeFloat64(tp.Flex2)
	writeFloat64(tp.Flex3)

	// Mass properties
	writeFloat64(tp.Mass)
	writeFloat64(tp.Ixx)
	writeFloat64(tp.Iyy)
	writeFloat64(tp.Izz)
	writeFloat64(tp.Ixy)
	writeFloat64(tp.Ixz)
	writeFloat64(tp.Iyz)

	// Environmental
	writeFloat64(tp.DragAccel)
	writeFloat64(tp.SRPAccel)
	writeFloat64(tp.GravityAcc)
	writeFloat64(tp.SunAngle)

	// Control/sensor
	writeUint32(tp.ThrusterOn)
	writeUint8(tp.GpsValid)

	// Padding for alignment
	offset += 7 // Align to 8-byte boundary

	// GPS position
	writeFloat64(tp.GpsPosX)
	writeFloat64(tp.GpsPosY)
	writeFloat64(tp.GpsPosZ)

	return buf, nil
}

// FromBytes deserializes packet from binary
func (tp *TelemetryPacket) FromBytes(data []byte) error {
	if len(data) < tp.Size() {
		return fmt.Errorf("insufficient data: got %d bytes, need %d", len(data), tp.Size())
	}

	offset := 0

	readFloat64 := func() float64 {
		f := binary.LittleEndian.Uint64(data[offset:])
		offset += 8
		return float64(f)
	}

	readUint32 := func() uint32 {
		u := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		return u
	}

	readUint8 := func() uint8 {
		u := data[offset]
		offset += 1
		return u
	}

	// Time
	tp.Time = readFloat64()

	// Position
	tp.PosX = readFloat64()
	tp.PosY = readFloat64()
	tp.PosZ = readFloat64()

	// Velocity
	tp.VelX = readFloat64()
	tp.VelY = readFloat64()
	tp.VelZ = readFloat64()

	// Quaternion
	tp.Q0 = readFloat64()
	tp.Q1 = readFloat64()
	tp.Q2 = readFloat64()
	tp.Q3 = readFloat64()

	// Angular velocity
	tp.Wx = readFloat64()
	tp.Wy = readFloat64()
	tp.Wz = readFloat64()

	// Flexible body
	tp.Flex1 = readFloat64()
	tp.Flex2 = readFloat64()
	tp.Flex3 = readFloat64()

	// Mass properties
	tp.Mass = readFloat64()
	tp.Ixx = readFloat64()
	tp.Iyy = readFloat64()
	tp.Izz = readFloat64()
	tp.Ixy = readFloat64()
	tp.Ixz = readFloat64()
	tp.Iyz = readFloat64()

	// Environmental
	tp.DragAccel = readFloat64()
	tp.SRPAccel = readFloat64()
	tp.GravityAcc = readFloat64()
	tp.SunAngle = readFloat64()

	// Control/sensor
	tp.ThrusterOn = readUint32()
	tp.GpsValid = readUint8()

	// Skip padding
	offset += 7

	// GPS position
	tp.GpsPosX = readFloat64()
	tp.GpsPosY = readFloat64()
	tp.GpsPosZ = readFloat64()

	return nil
}

// IPCClient manages connection to NASA 42
type IPCClient struct {
	config *IPCConfig
	conn   net.Conn
	active bool
	mu     sync.RWMutex
}

// NewIPCClient creates client with configuration
func NewIPCClient(config *IPCConfig) *IPCClient {
	return &IPCClient{
		config: config,
	}
}

// readFull reads exactly n bytes from connection with timeout
func (c *IPCClient) readFull(buf []byte) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.active || c.conn == nil {
		return fmt.Errorf("not connected")
	}

	// Set read deadline
	err := c.conn.SetReadDeadline(time.Now().Add(c.config.Timeout))
	if err != nil {
		return fmt.Errorf("set read deadline failed: %v", err)
	}

	_, err = io.ReadFull(c.conn, buf)
	if err != nil {
		return fmt.Errorf("read failed: %v", err)
	}

	// Clear deadline
	c.conn.SetReadDeadline(time.Time{})
	return nil
}

// writeAll writes all bytes to connection with timeout
func (c *IPCClient) writeAll(data []byte) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.active || c.conn == nil {
		return fmt.Errorf("not connected")
	}

	// Set write deadline
	err := c.conn.SetWriteDeadline(time.Now().Add(c.config.Timeout))
	if err != nil {
		return fmt.Errorf("set write deadline failed: %v", err)
	}

	_, err = c.conn.Write(data)
	if err != nil {
		return fmt.Errorf("write failed: %v", err)
	}

	// Clear deadline
	c.conn.SetWriteDeadline(time.Time{})
	return nil
}

// Connect establishes connection to NASA 42
func (c *IPCClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	addr := net.JoinHostPort(c.config.Host, fmt.Sprintf("%d", c.config.Port))

	conn, err := net.DialTimeout(c.config.SocketType, addr, c.config.Timeout)
	if err != nil {
		return fmt.Errorf("failed to connect to NASA 42 at %s: %v", addr, err)
	}

	c.conn = conn
	c.active = true

	return nil
}

// Disconnect closes connection
func (c *IPCClient) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.active = false
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// SendTelemetry transmits state to NASA 42
func (c *IPCClient) SendTelemetry(packet *TelemetryPacket) error {
	data, err := packet.ToBytes()
	if err != nil {
		return fmt.Errorf("serialization failed: %v", err)
	}

	err = c.writeAll(data)
	if err != nil {
		return fmt.Errorf("send failed: %v", err)
	}

	return nil
}

// ReceiveTelemetry reads propagated state from NASA 42
func (c *IPCClient) ReceiveTelemetry() (*TelemetryPacket, error) {
	packet := &TelemetryPacket{}
	buf := make([]byte, packet.Size())

	err := c.readFull(buf)
	if err != nil {
		return nil, fmt.Errorf("receive failed: %v", err)
	}

	err = packet.FromBytes(buf)
	if err != nil {
		return nil, fmt.Errorf("deserialization failed: %v", err)
	}

	return packet, nil
}

// HandoffObject sends object to 42 for high-fidelity propagation
func (c *IPCClient) HandoffObject(
	id string,
	pos [3]float64, // km
	vel [3]float64, // km/s
	quat [4]float64, // attitude
	mass float64, // kg
	duration float64, // seconds
) (*TelemetryPacket, error) {

	// Create initial state packet
	initial := &TelemetryPacket{
		Time: 0.0,
		PosX: pos[0],
		PosY: pos[1],
		PosZ: pos[2],
		VelX: vel[0],
		VelY: vel[1],
		VelZ: vel[2],
		Q0:   quat[0],
		Q1:   quat[1],
		Q2:   quat[2],
		Q3:   quat[3],
		Mass: mass,
		Wx:   0.0,
		Wy:   0.0,
		Wz:   0.0,
	}

	// Send to 42
	err := c.SendTelemetry(initial)
	if err != nil {
		return nil, fmt.Errorf("handoff failed for object %s: %v", id, err)
	}

	// Wait for propagated result
	result, err := c.ReceiveTelemetry()
	if err != nil {
		return nil, fmt.Errorf("receive failed for object %s: %v", id, err)
	}

	return result, nil
}

// ValidateAgainstADITI compares 42 output with ADITI propagation
func (c *IPCClient) ValidateAgainstADITI(
	aditiPos, aditiVel [3]float64,
	fortyTwoPacket *TelemetryPacket,
) (positionError, velocityError float64) {

	// Position error (meters)
	dx := (aditiPos[0] - fortyTwoPacket.PosX) * 1000.0
	dy := (aditiPos[1] - fortyTwoPacket.PosY) * 1000.0
	dz := (aditiPos[2] - fortyTwoPacket.PosZ) * 1000.0
	positionError = sqrt(dx*dx + dy*dy + dz*dz)

	// Velocity error (mm/s)
	dvx := (aditiVel[0] - fortyTwoPacket.VelX) * 1000.0
	dvy := (aditiVel[1] - fortyTwoPacket.VelY) * 1000.0
	dvz := (aditiVel[2] - fortyTwoPacket.VelZ) * 1000.0
	velocityError = sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

	return
}

// sqrt helper uses math.Sqrt for accurate distance calculations
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	return math.Sqrt(x)
}

// HighFidelityPropagator wraps 42 as a physics engine
type HighFidelityPropagator struct {
	client *IPCClient
}

// NewHighFidelityPropagator creates 42-backed propagator (not connected)
func NewHighFidelityPropagator(config *IPCConfig) *HighFidelityPropagator {
	client := NewIPCClient(config)
	return &HighFidelityPropagator{client: client}
}

// Connect establishes the IPC connection
func (hfp *HighFidelityPropagator) Connect() error {
	return hfp.client.Connect()
}

// Propagate uses NASA 42 for high-fidelity orbit propagation
func (hfp *HighFidelityPropagator) Propagate(
	initialState [6]float64, // [x,y,z,vx,vy,vz] km, km/s
	mass float64, // kg - spacecraft mass for drag/perturbation modeling
	duration float64, // seconds
) ([6]float64, error) {

	pos := [3]float64{initialState[0], initialState[1], initialState[2]}
	vel := [3]float64{initialState[3], initialState[4], initialState[5]}
	quat := [4]float64{1, 0, 0, 0} // Identity quaternion

	result, err := hfp.client.HandoffObject("temp", pos, vel, quat, mass, duration)
	if err != nil {
		return [6]float64{}, err
	}

	finalState := [6]float64{
		result.PosX, result.PosY, result.PosZ,
		result.VelX, result.VelY, result.VelZ,
	}

	return finalState, nil
}

// Close shuts down propagator
func (hfp *HighFidelityPropagator) Close() error {
	return hfp.client.Disconnect()
}
