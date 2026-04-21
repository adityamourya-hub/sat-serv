package proximity

import (
	"testing"
)

func TestNewFlatSlewCommand(t *testing.T) {
	cmd := NewFlatSlewCommand(
		1.0,          // azimuth
		0.5,          // elevation
		0.1,          // tilt rate
		0.01,         // tilt accel
		0.001,        // tilt jerk
		1234567890.0, // timestamp
		5,            // priority
	)

	if cmd == nil {
		t.Fatal("NewFlatSlewCommand returned nil")
	}

	if len(cmd.Bytes()) != 53 {
		t.Errorf("Expected 53 bytes, got %d", len(cmd.Bytes()))
	}
}

func TestFlatSlewCommand_Accessors(t *testing.T) {
	azimuth := 1.5
	elevation := 0.8
	tiltRate := 0.2
	tiltAccel := 0.02
	tiltJerk := 0.002
	timestamp := 9876543210.0
	priority := int8(7)

	cmd := NewFlatSlewCommand(azimuth, elevation, tiltRate, tiltAccel, tiltJerk, timestamp, priority)

	// Test zero-copy access
	if cmd.Azimuth() != azimuth {
		t.Errorf("Azimuth: got %.2f, expected %.2f", cmd.Azimuth(), azimuth)
	}

	if cmd.Elevation() != elevation {
		t.Errorf("Elevation: got %.2f, expected %.2f", cmd.Elevation(), elevation)
	}

	if cmd.TiltRate() != tiltRate {
		t.Errorf("TiltRate: got %.2f, expected %.2f", cmd.TiltRate(), tiltRate)
	}

	if cmd.TiltAccel() != tiltAccel {
		t.Errorf("TiltAccel: got %.2f, expected %.2f", cmd.TiltAccel(), tiltAccel)
	}

	if cmd.TiltJerk() != tiltJerk {
		t.Errorf("TiltJerk: got %.2f, expected %.2f", cmd.TiltJerk(), tiltJerk)
	}

	if cmd.Timestamp() != timestamp {
		t.Errorf("Timestamp: got %.2f, expected %.2f", cmd.Timestamp(), timestamp)
	}

	if cmd.Priority() != priority {
		t.Errorf("Priority: got %d, expected %d", cmd.Priority(), priority)
	}
}

func TestFlatSlewCommand_Validate(t *testing.T) {
	// Valid command
	validCmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)
	if !validCmd.Validate() {
		t.Error("Valid command failed validation")
	}

	// Invalid elevation (out of range)
	invalidElev := NewFlatSlewCommand(1.0, -2.0, 0.1, 0.01, 0.001, 1234567890.0, 5)
	if invalidElev.Validate() {
		t.Error("Invalid elevation passed validation")
	}

	// Invalid azimuth (out of range)
	invalidAz := NewFlatSlewCommand(7.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)
	if invalidAz.Validate() {
		t.Error("Invalid azimuth passed validation")
	}

	// Invalid priority (out of range)
	invalidPrio := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 15)
	if invalidPrio.Validate() {
		t.Error("Invalid priority passed validation")
	}
}

func TestFlatSlewCommand_Conversion(t *testing.T) {
	// Test conversion to SlewCommand
	flatCmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)
	cmd := flatCmd.ToSlewCommand()

	if cmd.Azimuth != flatCmd.Azimuth() {
		t.Error("Azimuth mismatch after conversion")
	}

	if cmd.Elevation != flatCmd.Elevation() {
		t.Error("Elevation mismatch after conversion")
	}

	if cmd.TiltRate != flatCmd.TiltRate() {
		t.Error("TiltRate mismatch after conversion")
	}

	if cmd.TiltAccel != flatCmd.TiltAccel() {
		t.Error("TiltAccel mismatch after conversion")
	}

	if cmd.TiltJerk != flatCmd.TiltJerk() {
		t.Error("TiltJerk mismatch after conversion")
	}

	if cmd.Timestamp != flatCmd.Timestamp() {
		t.Error("Timestamp mismatch after conversion")
	}

	if cmd.Priority != int(flatCmd.Priority()) {
		t.Error("Priority mismatch after conversion")
	}

	// Test conversion from SlewCommand
	flatCmd2 := FromSlewCommand(cmd)
	if flatCmd2.Azimuth() != cmd.Azimuth {
		t.Error("Azimuth mismatch after round-trip conversion")
	}
}

func TestFlatSlewAck(t *testing.T) {
	ack := NewFlatSlewAck(true, 45.5)

	if ack == nil {
		t.Fatal("NewFlatSlewAck returned nil")
	}

	if !ack.Success() {
		t.Error("Expected success=true")
	}

	if ack.SettleTimeMs() != 45.5 {
		t.Errorf("SettleTimeMs: got %.2f, expected 45.5", ack.SettleTimeMs())
	}

	if !ack.Validate() {
		t.Error("Valid acknowledgment failed validation")
	}

	// Test failure acknowledgment
	failAck := NewFlatSlewAck(false, 0.0)
	if failAck.Success() {
		t.Error("Expected success=false")
	}

	// Test invalid settle time
	invalidAck := NewFlatSlewAck(true, -10.0)
	if invalidAck.Validate() {
		t.Error("Invalid settle time passed validation")
	}
}

func BenchmarkFlatSlewCommand_New(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)
		_ = cmd.Bytes()
	}
}

func BenchmarkFlatSlewCommand_Access(b *testing.B) {
	cmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cmd.Azimuth()
		_ = cmd.Elevation()
		_ = cmd.TiltRate()
		_ = cmd.Priority()
	}
}

func BenchmarkFlatSlewCommand_Validate(b *testing.B) {
	cmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cmd.Validate()
	}
}

func BenchmarkFlatSlewCommand_Conversion(b *testing.B) {
	flatCmd := NewFlatSlewCommand(1.0, 0.5, 0.1, 0.01, 0.001, 1234567890.0, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := flatCmd.ToSlewCommand()
		_ = cmd
	}
}

func BenchmarkFlatSlewAck_New(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ack := NewFlatSlewAck(true, 45.5)
		_ = ack.Bytes()
	}
}
