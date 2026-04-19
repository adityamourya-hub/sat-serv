// Package comms provides tactical messaging protocols for air-gapped defense systems
// Implements C.R.A.P.S.H.O.O.T. protocol and hierarchical communications
package comms

import (
	"fmt"
	"strings"
	"time"
)

const (
	// Transmission timeouts
	KeyupDelaySeconds     = 1.0  // 1 second delay to prevent clipping
	MaxMessageLength      = 160  // Character limit for tactical messages
	CallSignMaxLength    = 8    // Maximum callsign length
	TransmissionTimeout  = 10 * time.Second
)

// Prowords (procedural words) for tactical communication
var Prowords = map[string]string{
	"OVER":    "I have finished my transmission and expect a reply",
	"OUT":     "I have finished my transmission and do not expect a reply",
	"TRAFFIC": "I have other messages to send after this one",
	"ROGER":   "I have received and understood your last transmission",
	"WILCO":   "I have received and will comply",
	"COPY":    "I have received and understood",
	"STANDBY": "Wait for further instructions",
	"GO":      "Proceed with your message",
	"SAYAGAIN":"Repeat your last transmission",
	"NEGATIVE":"No",
	"AFFIRM":  "Yes",
	"UNKNOWN": "Unknown",
}

// CallSign represents a tactical callsign
type CallSign struct {
	Unit      string // Unit identifier
	Number    string // Unit number
	Parent    string // Parent unit (for hierarchy)
	IsCommand bool   // Whether this is a command element
}

// String returns the formatted callsign
func (cs *CallSign) String() string {
	return fmt.Sprintf("%s%s", cs.Unit, cs.Number)
}

// FullString returns the full callsign with parent
func (cs *CallSign) FullString() string {
	if cs.Parent != "" {
		return fmt.Sprintf("%s-%s", cs.Parent, cs.String())
	}
	return cs.String()
}

// TacticalMessage represents a tactical radio transmission
type TacticalMessage struct {
	From        *CallSign
	To          *CallSign
	Message     string
	Proword     string // OVER, OUT, TRAFFIC
	Timestamp   time.Time
	IsEncrypted bool
	Priority    int // 1=Low, 2=Medium, 3=High, 4=Critical
}

// CRAPSHOOTState represents the state in the C.R.A.P.S.H.O.O.T. routine
type CRAPSHOOTState string

const (
	StateIdle         CRAPSHOOTState = "IDLE"
	StateCompose      CRAPSHOOTState = "COMPOSE"
	StateRelax        CRAPSHOOTState = "RELAX"
	StateActivate     CRAPSHOOTState = "ACTIVATE"
	StatePause        CRAPSHOOTState = "PAUSE"
	StateSpeak        CRAPSHOOTState = "SPEAK"
	StateHail         CRAPSHOOTState = "HAIL"
	StateTransmitting CRAPSHOOTState = "TRANSMITTING"
	StateComplete     CRAPSHOOTState = "COMPLETE"
)

// CRAPSHOOTProtocol implements the C.R.A.P.S.H.O.O.T. transmission routine
type CRAPSHOOTProtocol struct {
	currentState    CRAPSHOOTState
	callsign        *CallSign
	composedMessage string
	keyupTimer      *time.Timer
	transmitHistory []TacticalMessage
}

// NewCRAPSHOOTProtocol creates a new C.R.A.P.S.H.O.O.T. protocol handler
func NewCRAPSHOOTProtocol(callsign *CallSign) *CRAPSHOOTProtocol {
	return &CRAPSHOOTProtocol{
		currentState:    StateIdle,
		callsign:        callsign,
		composedMessage: "",
		transmitHistory: make([]TacticalMessage, 0),
	}
}

// Compose initiates message composition phase
func (cp *CRAPSHOOTProtocol) Compose() error {
	if cp.currentState != StateIdle {
		return fmt.Errorf("cannot compose in state %s", cp.currentState)
	}
	cp.currentState = StateCompose
	return nil
}

// Relax performs mental rehearsal and message shortening
func (cp *CRAPSHOOTProtocol) Relax(message string) (string, error) {
	if cp.currentState != StateCompose {
		return "", fmt.Errorf("cannot relax in state %s", cp.currentState)
	}

	// Shorten message if needed
	shortened := cp.shortenMessage(message)
	cp.composedMessage = shortened
	cp.currentState = StateRelax
	return shortened, nil
}

// Activate initiates keyup sequence
func (cp *CRAPSHOOTProtocol) Activate() error {
	if cp.currentState != StateRelax {
		return fmt.Errorf("cannot activate in state %s", cp.currentState)
	}
	cp.currentState = StateActivate
	
	// Start keyup delay timer
	cp.keyupTimer = time.AfterFunc(time.Second, func() {
		cp.currentState = StateSpeak
	})
	
	return nil
}

// Pause waits for keyup delay to prevent clipping
func (cp *CRAPSHOOTProtocol) Pause() error {
	if cp.currentState != StateActivate {
		return fmt.Errorf("cannot pause in state %s", cp.currentState)
	}
	
	// Wait for keyup delay
	time.Sleep(time.Second)
	cp.currentState = StateSpeak
	return nil
}

// Speak initiates transmission with proper callsign format
func (cp *CRAPSHOOTProtocol) Speak(to *CallSign) (string, error) {
	if cp.currentState != StateSpeak {
		return "", fmt.Errorf("cannot speak in state %s", cp.currentState)
	}
	
	// Format: "YOU ARE [TO], I AM [FROM]"
	hail := fmt.Sprintf("YOU ARE %s, I AM %s", to.FullString(), cp.callsign.FullString())
	cp.currentState = StateHail
	return hail, nil
}

// Hail delivers the actual message
func (cp *CRAPSHOOTProtocol) Hail(proword string) (*TacticalMessage, error) {
	if cp.currentState != StateHail {
		return nil, fmt.Errorf("cannot hail in state %s", cp.currentState)
	}
	
	// Validate proword
	if _, exists := Prowords[proword]; !exists {
		return nil, fmt.Errorf("invalid proword: %s", proword)
	}
	
	cp.currentState = StateTransmitting
	
	message := &TacticalMessage{
		From:      cp.callsign,
		Message:   cp.composedMessage,
		Proword:   proword,
		Timestamp: time.Now(),
	}
	
	return message, nil
}

// Complete marks transmission as complete
func (cp *CRAPSHOOTProtocol) Complete() error {
	if cp.currentState != StateTransmitting {
		return fmt.Errorf("cannot complete in state %s", cp.currentState)
	}
	
	cp.currentState = StateComplete
	cp.composedMessage = ""
	return nil
}

// Reset resets the protocol to idle state
func (cp *CRAPSHOOTProtocol) Reset() {
	cp.currentState = StateIdle
	cp.composedMessage = ""
	if cp.keyupTimer != nil {
		cp.keyupTimer.Stop()
		cp.keyupTimer = nil
	}
}

// GetState returns the current state
func (cp *CRAPSHOOTProtocol) GetState() CRAPSHOOTState {
	return cp.currentState
}

// shortenMessage shortens a message to fit within tactical limits
func (cp *CRAPSHOOTProtocol) shortenMessage(message string) string {
	if len(message) <= MaxMessageLength {
		return message
	}
	
	// Truncate and add ellipsis
	return message[:MaxMessageLength-3] + "..."
}

// Transmit performs the complete C.R.A.P.S.H.O.O.T. routine
func (cp *CRAPSHOOTProtocol) Transmit(to *CallSign, message string, proword string) (*TacticalMessage, error) {
	// Compose
	if err := cp.Compose(); err != nil {
		return nil, err
	}
	
	// Relax (shorten)
	_, err := cp.Relax(message)
	if err != nil {
		return nil, err
	}
	
	// Activate
	if err := cp.Activate(); err != nil {
		return nil, err
	}
	
	// Pause (keyup delay)
	if err := cp.Pause(); err != nil {
		return nil, err
	}
	
	// Speak (callsign hail)
	_, err = cp.Speak(to)
	if err != nil {
		return nil, err
	}
	
	// Hail (transmit message)
	msg, err := cp.Hail(proword)
	if err != nil {
		return nil, err
	}
	msg.To = to
	
	// Complete
	if err := cp.Complete(); err != nil {
		return nil, err
	}
	
	// Store in history
	cp.transmitHistory = append(cp.transmitHistory, *msg)
	
	// Reset
	cp.Reset()
	
	return msg, nil
}

// HierarchicalComms implements hierarchical communication structure
type HierarchicalComms struct {
	units         map[string]*CallSign
	parents       map[string]string
	children      map[string][]*CallSign
	commandChain  []*CallSign
}

// NewHierarchicalComms creates a new hierarchical communications system
func NewHierarchicalComms() *HierarchicalComms {
	return &HierarchicalComms{
		units:        make(map[string]*CallSign),
		parents:      make(map[string]string),
		children:     make(map[string][]*CallSign),
		commandChain: make([]*CallSign, 0),
	}
}

// AddUnit adds a unit to the hierarchy
func (hc *HierarchicalComms) AddUnit(callsign *CallSign) {
	hc.units[callsign.String()] = callsign
	
	if callsign.Parent != "" {
		hc.parents[callsign.String()] = callsign.Parent
		hc.children[callsign.Parent] = append(hc.children[callsign.Parent], callsign)
	}
	
	if callsign.IsCommand {
		hc.commandChain = append(hc.commandChain, callsign)
	}
}

// GetParent returns the parent unit of a given unit
func (hc *HierarchicalComms) GetParent(unit string) *CallSign {
	parentID, exists := hc.parents[unit]
	if !exists {
		return nil
	}
	return hc.units[parentID]
}

// GetChildren returns the child units of a given unit
func (hc *HierarchicalComms) GetChildren(unit string) []*CallSign {
	return hc.children[unit]
}

// GetCommandChain returns the command chain from bottom to top
func (hc *HierarchicalComms) GetCommandChain(unit string) []*CallSign {
	chain := make([]*CallSign, 0)
	
	current := hc.units[unit]
	for current != nil {
		chain = append([]*CallSign{current}, chain...) // Prepend
		current = hc.GetParent(current.String())
	}
	
	return chain
}

// FindCommonParent finds the common parent between two units
func (hc *HierarchicalComms) FindCommonParent(unit1, unit2 string) *CallSign {
	chain1 := hc.GetCommandChain(unit1)
	chain2 := hc.GetCommandChain(unit2)
	
	// Find first common element
	for _, u1 := range chain1 {
		for _, u2 := range chain2 {
			if u1.String() == u2.String() {
				return u1
			}
		}
	}
	
	return nil
}

// TacticalNet represents a tactical radio network
type TacticalNet struct {
	name         string
	frequency    string // MHz or channel
	mode         string // FM, AM, SSB, DIGITAL
	encryption   string // NONE, DRYAD, OTP, DIGITAL
	participants map[string]*CallSign
	isActive     bool
}

// NewTacticalNet creates a new tactical network
func NewTacticalNet(name, frequency, mode, encryption string) *TacticalNet {
	return &TacticalNet{
		name:         name,
		frequency:    frequency,
		mode:         mode,
		encryption:   encryption,
		participants: make(map[string]*CallSign),
		isActive:     false,
	}
}

// Join adds a participant to the network
func (tn *TacticalNet) Join(callsign *CallSign) {
	tn.participants[callsign.String()] = callsign
}

// Leave removes a participant from the network
func (tn *TacticalNet) Leave(callsign *CallSign) {
	delete(tn.participants, callsign.String())
}

// Activate activates the network
func (tn *TacticalNet) Activate() {
	tn.isActive = true
}

// Deactivate deactivates the network
func (tn *TacticalNet) Deactivate() {
	tn.isActive = false
}

// IsActive checks if the network is active
func (tn *TacticalNet) IsActive() bool {
	return tn.isActive
}

// Broadcast sends a message to all participants
func (tn *TacticalNet) Broadcast(message string, from *CallSign) error {
	if !tn.isActive {
		return fmt.Errorf("network not active")
	}
	
	// In production, this would send via actual radio interface
	return nil
}

// GetParticipantCount returns the number of participants
func (tn *TacticalNet) GetParticipantCount() int {
	return len(tn.participants)
}

// TransmissionLogger logs all tactical transmissions
type TransmissionLogger struct {
	logs []TacticalMessage
}

// NewTransmissionLogger creates a new transmission logger
func NewTransmissionLogger() *TransmissionLogger {
	return &TransmissionLogger{
		logs: make([]TacticalMessage, 0),
	}
}

// Log logs a transmission
func (tl *TransmissionLogger) Log(message *TacticalMessage) {
	tl.logs = append(tl.logs, *message)
}

// GetLogs returns all logs
func (tl *TransmissionLogger) GetLogs() []TacticalMessage {
	return tl.logs
}

// GetLogsByFrom returns logs from a specific callsign
func (tl *TransmissionLogger) GetLogsByFrom(callsign string) []TacticalMessage {
	filtered := make([]TacticalMessage, 0)
	
	for _, log := range tl.logs {
		if log.From.String() == callsign {
			filtered = append(filtered, log)
		}
	}
	
	return filtered
}

// GetLogsByTo returns logs to a specific callsign
func (tl *TransmissionLogger) GetLogsByTo(callsign string) []TacticalMessage {
	filtered := make([]TacticalMessage, 0)
	
	for _, log := range tl.logs {
		if log.To != nil && log.To.String() == callsign {
			filtered = append(filtered, log)
		}
	}
	
	return filtered
}

// GetLogsByPriority returns logs with minimum priority
func (tl *TransmissionLogger) GetLogsByPriority(minPriority int) []TacticalMessage {
	filtered := make([]TacticalMessage, 0)
	
	for _, log := range tl.logs {
		if log.Priority >= minPriority {
			filtered = append(filtered, log)
		}
	}
	
	return filtered
}

// GetRecentLogs returns logs within a time window
func (tl *TransmissionLogger) GetRecentLogs(duration time.Duration) []TacticalMessage {
	cutoff := time.Now().Add(-duration)
	filtered := make([]TacticalMessage, 0)
	
	for _, log := range tl.logs {
		if log.Timestamp.After(cutoff) {
			filtered = append(filtered, log)
		}
	}
	
	return filtered
}

// ClearLogs clears all logs
func (tl *TransmissionLogger) ClearLogs() {
	tl.logs = make([]TacticalMessage, 0)
}

// FormatMessage formats a tactical message for transmission
func FormatMessage(message *TacticalMessage) string {
	var builder strings.Builder
	
	// Callsign hail
	if message.To != nil {
		builder.WriteString(fmt.Sprintf("YOU ARE %s, ", message.To.FullString()))
	}
	builder.WriteString(fmt.Sprintf("I AM %s. ", message.From.FullString()))
	
	// Message body
	builder.WriteString(message.Message)
	builder.WriteString(". ")
	
	// Proword
	builder.WriteString(message.Proword)
	
	return builder.String()
}

// ParseMessage parses a received tactical message
func ParseMessage(raw string) (*TacticalMessage, error) {
	// Simplified parser
	// In production, implement full parsing for all formats
	
	return &TacticalMessage{
		Message:   raw,
		Timestamp: time.Now(),
	}, fmt.Errorf("message parsing not fully implemented")
}
