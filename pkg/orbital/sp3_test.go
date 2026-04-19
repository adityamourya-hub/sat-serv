package orbital

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSP3Ingestor_Parse(t *testing.T) {
	// Create a minimal test SP3 file
	// SP3 header line 1 format per IGS spec (positions 1-indexed):
	// #cPYYYY MM DD HH MM SS.SSSSSSSSS EEEEEEEE D T AGENCY
	// 12345678901234567890123456789012345678901234567890123456789012345678901234567890
	// |    |   | |  | |  |  |         |       | | |
	testSP3Content := "#cP2024  1  1  0  0  0.00000000       2 D L SSIS                    \n" +
		"## 1485 0.00000000 0.000000000 0.00000000000 0.00000000000000\n" +
		"+   1   L123 L124 L125  0  0  0  0  0  0  0  0  0  0  0  0  0\n" +
		"+    0    0    0    0    0    0    0    0    0    0\n" +
		"+    0    0    0    0    0    0    0    0    0    0\n" +
		"+    0    0    0    0    0    0    0    0    0    0\n" +
		"+    0    0    0    0    0    0    0    0    0    0\n" +
		"++         0    0    0    0    0    0    0    0    0\n" +
		"++         0    0    0    0    0    0    0    0    0\n" +
		"++         0    0    0    0    0    0    0    0    0\n" +
		"++         0    0    0    0    0    0    0    0    0\n" +
		"++         0    0    0    0    0    0    0    0    0\n" +
		"%%c cc cc c c c c c c c c c c c c c c c c c c c c c c c c c c c c cc cc\n" +
		"%%c cc cc c c c c c c c c c c c c c c c c c c c c c c c c c c c c cc cc\n" +
		"%%f  1.2500000 1.025000000 0.00000000 0.000000000000000\n" +
		"%%f  0.0000000 0.000000000 0.00000000 0.000000000000000\n" +
		"%%i    0    0    0    0      0      0      0      0         0\n" +
		"%%i    0    0    0    0      0      0      0      0         0\n" +
		"/* TEST FILE\n" +
		"/* Generated for unit testing\n" +
		"/* END COMMENT\n" +
		"*  2024  1  1  0  0  0.000000\n" +
		"PL123  -1349527.750000  -5553991.750000  -3045976.250000     0.000000\n" +
		"PL124  -1349528.750000  -5553992.750000  -3045977.250000     0.000000\n" +
		"*  2024  1  1  0 15  0.000000\n" +
		"PL123  -1349526.750000  -5553990.750000  -3045975.250000     0.000000\n" +
		"PL124  -1349527.750000  -5553991.750000  -3045976.250000     0.000000\n" +
		"EOF\n"

	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.sp3")

	err := os.WriteFile(testFile, []byte(testSP3Content), 0644)
	if err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	ingestor := NewSP3Ingestor(false)
	sp3, err := ingestor.ParseFile(testFile)
	if err != nil {
		t.Fatalf("Failed to parse SP3: %v", err)
	}

	// Verify header
	if sp3.Header.Version != SP3VersionC {
		t.Errorf("Expected version c, got %s", sp3.Header.Version)
	}

	expectedTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	if !sp3.Header.StartTime.Equal(expectedTime) {
		t.Errorf("Expected start time %v, got %v", expectedTime, sp3.Header.StartTime)
	}

	if sp3.Header.NumberOfEpochs != 2 {
		t.Errorf("Expected 2 epochs, got %d", sp3.Header.NumberOfEpochs)
	}

	if sp3.Header.OrbitType != SP3OrbitTypeLEO {
		t.Errorf("Expected LEO orbit type, got %s", sp3.Header.OrbitType)
	}
}

func TestSP3Exporter_Export(t *testing.T) {
	tmpDir := t.TempDir()
	exportFile := filepath.Join(tmpDir, "export.sp3")

	sp3 := &SP3File{
		Header: SP3Header{
			Version:        SP3VersionD,
			StartTime:      time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC),
			NumberOfEpochs: 1,
			TimeSystem:     "UTC",
			OrbitType:      SP3OrbitTypeLEO,
			Agency:         "SSIS",
			BaseClockRate:  1.0,
		},
		Satellites: []SP3SatelliteData{
			{
				PRN:      "L12345",
				Epoch:    time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC),
				Position: [3]float64{7000.0, 0.0, 0.0},
				Velocity: &[3]float64{0.0, 7.5, 0.0},
			},
		},
	}

	exporter := NewSP3Exporter(SP3VersionD)
	err := exporter.Export(exportFile, sp3)
	if err != nil {
		t.Fatalf("Failed to export SP3: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(exportFile); os.IsNotExist(err) {
		t.Fatal("Export file was not created")
	}

	// Try to parse it back
	ingestor := NewSP3Ingestor(false)
	parsed, err := ingestor.ParseFile(exportFile)
	if err != nil {
		t.Fatalf("Failed to parse exported SP3: %v", err)
	}

	if len(parsed.Satellites) == 0 {
		t.Error("No satellites parsed from exported file")
	}
}

func TestValidateSP3(t *testing.T) {
	tests := []struct {
		name      string
		sp3       *SP3File
		wantErrs  int
		wantWarns int
	}{
		{
			name: "valid SP3",
			sp3: &SP3File{
				Header: SP3Header{NumberOfEpochs: 2},
				Satellites: []SP3SatelliteData{
					{PRN: "L001", Position: [3]float64{7000, 0, 0}},
					{PRN: "L002", Position: [3]float64{7001, 0, 0}},
				},
			},
			wantErrs:  0,
			wantWarns: 0,
		},
		{
			name: "zero epochs",
			sp3: &SP3File{
				Header:     SP3Header{NumberOfEpochs: 0},
				Satellites: []SP3SatelliteData{},
			},
			wantErrs:  1,
			wantWarns: 0,
		},
		{
			name: "NaN position",
			sp3: &SP3File{
				Header: SP3Header{NumberOfEpochs: 1},
				Satellites: []SP3SatelliteData{
					{PRN: "L001", Position: [3]float64{7000, 0, 0}},
					{PRN: "L002", Position: [3]float64{0, 0, math.NaN()}}, // NaN
				},
			},
			wantErrs:  1,
			wantWarns: 1, // Mismatch warning
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validation := ValidateSP3(tt.sp3)
			if len(validation.Errors) != tt.wantErrs {
				t.Errorf("got %d errors, want %d: %v", len(validation.Errors), tt.wantErrs, validation.Errors)
			}
			if len(validation.Warnings) != tt.wantWarns {
				t.Errorf("got %d warnings, want %d: %v", len(validation.Warnings), tt.wantWarns, validation.Warnings)
			}
		})
	}
}

func TestConvertSP3ToStateVectors(t *testing.T) {
	sp3 := &SP3File{
		Satellites: []SP3SatelliteData{
			{
				PRN:      "L001",
				Epoch:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
				Position: [3]float64{7000.0, 1000.0, 200.0},
				Velocity: &[3]float64{1.0, 7.5, 0.5},
			},
			{
				PRN:      "L002",
				Epoch:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
				Position: [3]float64{7001.0, 1001.0, 201.0},
				// No velocity
			},
		},
	}

	states := ConvertSP3ToStateVectors(sp3)

	if len(states) != 2 {
		t.Fatalf("Expected 2 state vectors, got %d", len(states))
	}

	if states[0].R[0] != 7000.0 {
		t.Errorf("Expected R[0] = 7000, got %f", states[0].R[0])
	}

	if states[0].V[1] != 7.5 {
		t.Errorf("Expected V[1] = 7.5, got %f", states[0].V[1])
	}

	// Second state should have zero velocity (default)
	if states[1].V[0] != 0 || states[1].V[1] != 0 || states[1].V[2] != 0 {
		t.Error("Expected zero velocity for satellite without velocity data")
	}
}
