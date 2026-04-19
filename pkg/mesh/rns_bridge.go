// Package mesh provides Reticulum Network Stack (RNS) integration
// For air-gapped sovereign mesh communication
package mesh

import (
	"context"
	"fmt"
	"time"
	// TODO: Re-enable when dependency available
	// "github.com/pebbe/zmq4"
)

// RNSBridge handles transmission of TargetBrief over Reticulum mesh
// Uses ZeroMQ to interface with Python RNS relay
type RNSBridge struct {
	zmqSocket  interface{} // TODO: *zmq.Socket when available
	endpoint   string
	originNode string
}

// RNSConfig holds bridge configuration
type RNSConfig struct {
	Endpoint   string        // ipc:///tmp/rns_relay.ipc
	OriginNode string        // RNS address hash
	Timeout    time.Duration // Send timeout
}

// DefaultRNSConfig returns standard configuration
func DefaultRNSConfig() *RNSConfig {
	return &RNSConfig{
		Endpoint:   "ipc:///tmp/rns_relay.ipc",
		OriginNode: "iaf-node-alpha",
		Timeout:    5 * time.Second,
	}
}

// NewRNSBridge creates RNS bridge instance
func NewRNSBridge(config *RNSConfig) (*RNSBridge, error) {
	if config == nil {
		config = DefaultRNSConfig()
	}

	// TODO: Re-enable when zmq available
	// zctx, err := zmq.NewContext()
	// if err != nil {
	// 	return nil, fmt.Errorf("zmq context: %w", err)
	// }

	// socket, err := zctx.NewSocket(zmq.REQ)
	// if err != nil {
	// 	return nil, fmt.Errorf("zmq socket: %w", err)
	// }

	// if err := socket.Connect(config.Endpoint); err != nil {
	// 	socket.Close()
	// 	return nil, fmt.Errorf("zmq connect: %w", err)
	// }

	return &RNSBridge{
		zmqSocket:  nil, // TODO: socket when available
		endpoint:   config.Endpoint,
		originNode: config.OriginNode,
	}, nil
}

// TransmitBrief sends TargetBrief over RNS mesh
// Serializes to MessagePack (~450 bytes) for LoRa efficiency
func (rb *RNSBridge) TransmitBrief(brief *TargetBrief) error {
	if err := brief.Validate(); err != nil {
		return fmt.Errorf("invalid brief: %w", err)
	}

	// Add origin node
	brief.OriginNode = rb.originNode

	// Serialize to MessagePack
	// TODO: Re-enable when msgpack available
	data, err := brief.SerializeToMessagePack()
	if err != nil {
		return fmt.Errorf("serialize: %w", err)
	}

	// Send via ZMQ
	// TODO: Re-enable when zmq available
	// if _, err := rb.zmqSocket.SendBytes(data, 0); err != nil {
	// 	return fmt.Errorf("zmq send: %w", err)
	// }

	_ = data // Avoid unused variable

	return nil
}

// TransmitBriefAsync sends brief asynchronously (non-blocking)
func (rb *RNSBridge) TransmitBriefAsync(brief *TargetBrief) <-chan error {
	errChan := make(chan error, 1)

	go func() {
		errChan <- rb.TransmitBrief(brief)
	}()

	return errChan
}

// PushTactical sends binary tactical data to destination hash
// Used for Fire-Control Handover (FCH) packets
func (rb *RNSBridge) PushTactical(destinationHash string, data []byte) error {
	// TODO: Re-enable when zmq available
	// Send via ZMQ to RNS relay
	// if _, err := rb.zmqSocket.SendBytes(data, 0); err != nil {
	// 	return fmt.Errorf("zmq send: %w", err)
	// }

	_ = destinationHash // Avoid unused variable
	_ = data            // Avoid unused variable

	return nil
}

// TransmitFireControl sends FireControlBrief over RNS mesh
// Main entry point for FCH dispatch
func (rb *RNSBridge) TransmitFireControl(brief *FireControlBrief, destinationHash string) error {
	if err := brief.Validate(); err != nil {
		return fmt.Errorf("invalid fire control brief: %w", err)
	}

	// Serialize to binary
	serializer := NewFCHSerializer()
	data, err := serializer.Serialize(brief)
	if err != nil {
		return fmt.Errorf("serialize: %w", err)
	}

	// Push to tactical link
	if err := rb.PushTactical(destinationHash, data); err != nil {
		return fmt.Errorf("tactical push: %w", err)
	}

	return nil
}

// Close terminates the bridge connection
func (rb *RNSBridge) Close() error {
	// TODO: Re-enable when zmq available
	// if rb.zmqSocket != nil {
	// 	return rb.zmqSocket.Close()
	// }
	return nil
}

// GenerateBriefFromDetection creates TargetBrief from detection pipeline
// This is the "Kill-Chain" integration point
func GenerateBriefFromDetection(
	noradID int,
	pos, vel [3]float64,
	materialID string,
	materialConf float64,
	attitudeState string,
	maxElevation float64,
	isAdversary bool,
) *TargetBrief {
	brief := NewTargetBrief(noradID, pos, vel, materialID, materialConf, attitudeState)

	// Set priority based on threat level
	if isAdversary && attitudeState == "STABILIZED_NADIR" && maxElevation > 30.0 {
		brief.Priority = PriorityCritical
		brief.ActionCode = ActionEMCONB
		brief.Classification = ClassAdversaryISR
	} else if isAdversary {
		brief.Priority = PriorityHigh
		brief.ActionCode = ActionTRACK
		brief.Classification = ClassAdversaryISR
	} else {
		brief.Priority = PriorityRoutine
		brief.ActionCode = ActionTRACK
		brief.Classification = ClassNeutral
	}

	// Set Siliguri-specific fields
	brief.TargetVolume = "SILIGURI_LEO_S2_L13"
	brief.MaxElevation = maxElevation

	// Set ISR capability based on material
	if materialID == "GaAs_SOLAR_PANEL" || materialID == "Si_SOLAR_PANEL" {
		brief.ISRCapability = "OPTICAL_HIGH_RES"
	} else if materialID == "STEALTH_COATING" {
		brief.ISRCapability = "STEALTH_RECON"
	}

	// Set slope trend based on attitude
	if attitudeState == "STABILIZED_NADIR" {
		brief.SlopeTrend = SlopeStable
	} else if attitudeState == "MANEUVERING" {
		brief.SlopeTrend = SlopeManeuvering
	}

	return brief
}

// EstimateBandwidth calculates bandwidth requirements
func EstimateBandwidth(briefsPerMinute int) map[string]string {
	bytesPerBrief := 450 // MessagePack serialized size
	bitsPerBrief := bytesPerBrief * 8

	bps := float64(briefsPerMinute) * float64(bitsPerBrief) / 60.0

	return map[string]string{
		"briefs_per_min":  fmt.Sprintf("%d", briefsPerMinute),
		"bytes_per_brief": fmt.Sprintf("%d", bytesPerBrief),
		"bits_per_sec":    fmt.Sprintf("%.1f", bps),
		"loRa_9600_ok":    fmt.Sprintf("%v", bps < 9600),
		"bandwidth_usage": fmt.Sprintf("%.1f%%", bps/9600.0*100),
	}
}

// MockMeshTest simulates two-node mesh (M2 <-> Linux laptop)
func MockMeshTest(ctx context.Context, brief *TargetBrief) error {
	// TestBridgeConnection tests connectivity to Python relay
	// TODO: Re-enable when zmq available
	// Server (receiver)
	// server, err := zmq.NewSocket(zmq.PULL)
	// if err != nil {
	// 	return err
	// }
	// defer server.Close()

	// if err := server.Bind(endpoint); err != nil {
	// 	return err
	// }

	// Client (sender)
	// client, err := zmq.NewSocket(zmq.PUSH)
	// if err != nil {
	// 	return err
	// }
	// defer client.Close()

	return fmt.Errorf("zmq dependency not available - TODO: re-enable when available")

	// if err := client.Connect(endpoint); err != nil {
	// 	return err
	// }

	// Send brief
	// data, err := brief.ToMessagePack()
	// if err != nil {
	// 	return err
	// }

	// if _, err := client.SendBytes(data, 0); err != nil {
	// 	return err
	// }

	// Receive with timeout
	// server.SetRcvTimeout(1000)
	// received, err := server.RecvBytes(0)
	// if err != nil {
	// 	return err
	// }

	// Verify
	// decoded, err := FromMessagePack(received)
	// if err != nil {
	// 	return err
	// }

	// if decoded.NoradID != brief.NoradID {
	// 	return fmt.Errorf("norad_id mismatch: %d != %d", decoded.NoradID, brief.NoradID)
	// }

	// return nil
}

// FireControlDispatcher manages FCH dispatch operations
type FireControlDispatcher struct {
	bridge     *RNSBridge
	telemetry  *FireControlTelemetry
	serializer *FCHSerializer
}

// NewFireControlDispatcher creates a new FCH dispatcher
func NewFireControlDispatcher(bridge *RNSBridge) *FireControlDispatcher {
	return &FireControlDispatcher{
		bridge:     bridge,
		telemetry:  NewFireControlTelemetry(),
		serializer: NewFCHSerializer(),
	}
}

// Dispatch sends a fire control handover to the specified interceptor
func (d *FireControlDispatcher) Dispatch(brief *FireControlBrief, interceptorHash string) error {
	start := time.Now()

	err := d.bridge.TransmitFireControl(brief, interceptorHash)

	latency := time.Since(start).Milliseconds()
	d.telemetry.RecordDispatch(err == nil, float64(latency))

	if err != nil {
		return fmt.Errorf("fire control dispatch failed: %w", err)
	}

	return nil
}

// GetTelemetry returns current dispatch telemetry
func (d *FireControlDispatcher) GetTelemetry() *FireControlTelemetry {
	return d.telemetry
}
