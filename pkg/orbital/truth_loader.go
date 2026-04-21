// Package orbital provides sovereign truth data loading for residual validation
// Implements OEM (Orbit Ephemeris Message) parsing for air-gapped validation
package orbital

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"ssis/pkg/physics"
)

// OEMFile represents a parsed Orbit Ephemeris Message
type OEMFile struct {
	Header   OEMHeader
	Metadata OEMMetadata
	Segments []OEMSegment
}

// OEMHeader contains the CCSDS OEM header information
type OEMHeader struct {
	Version        string
	Classification string
	Originator     string
	CreationDate   time.Time
}

// OEMMetadata contains ephemeris metadata
type OEMMetadata struct {
	ObjectName string
	ObjectID   string
	CenterName string
	RefFrame   string
	TimeSystem string
	StartTime  time.Time
	StopTime   time.Time
	StepSize   float64 // seconds
}

// OEMSegment contains a single ephemeris segment
type OEMSegment struct {
	Epoch time.Time
	State physics.State
}

// TruthLoader loads and manages OEM truth data for propagator validation
type TruthLoader struct {
	truthData map[string]*OEMFile
}

// NewTruthLoader creates a new sovereign truth data loader
func NewTruthLoader() *TruthLoader {
	return &TruthLoader{
		truthData: make(map[string]*OEMFile),
	}
}

// LoadOEM parses an OEM file from disk
func (tl *TruthLoader) LoadOEM(filepath string) (*OEMFile, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open OEM file: %w", err)
	}
	defer file.Close()

	oem := &OEMFile{
		Segments: make([]OEMSegment, 0),
	}

	scanner := bufio.NewScanner(file)
	section := "none"

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if len(line) == 0 || strings.HasPrefix(line, "#") {
			continue
		}

		// Detect section headers
		if strings.HasPrefix(line, "CCSDS_OEM_VERS") {
			oem.Header.Version = extractValue(line)
			section = "header"
			continue
		}
		if strings.HasPrefix(line, "CREATION_DATE") {
			oem.Header.CreationDate = parseOEMTime(extractValue(line))
			continue
		}
		if strings.HasPrefix(line, "ORIGINATOR") {
			oem.Header.Originator = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "META_START") {
			section = "metadata"
			continue
		}
		if strings.HasPrefix(line, "META_STOP") {
			section = "none"
			continue
		}
		if strings.HasPrefix(line, "OBJECT_NAME") {
			oem.Metadata.ObjectName = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "OBJECT_ID") {
			oem.Metadata.ObjectID = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "CENTER_NAME") {
			oem.Metadata.CenterName = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "REF_FRAME") {
			oem.Metadata.RefFrame = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "TIME_SYSTEM") {
			oem.Metadata.TimeSystem = extractValue(line)
			continue
		}
		if strings.HasPrefix(line, "START_TIME") {
			oem.Metadata.StartTime = parseOEMTime(extractValue(line))
			continue
		}
		if strings.HasPrefix(line, "STOP_TIME") {
			oem.Metadata.StopTime = parseOEMTime(extractValue(line))
			continue
		}
		if strings.HasPrefix(line, "STEP_SIZE") {
			oem.Metadata.StepSize = parseFloat(extractValue(line))
			continue
		}
		if strings.HasPrefix(line, "EPHEMERIS_START") {
			section = "ephemeris"
			continue
		}
		if strings.HasPrefix(line, "EPHEMERIS_STOP") {
			section = "none"
			continue
		}

		// Parse ephemeris data lines
		if section == "ephemeris" {
			segment, err := parseEphemerisLine(line)
			if err == nil {
				oem.Segments = append(oem.Segments, segment)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning OEM file: %w", err)
	}

	tl.truthData[oem.Metadata.ObjectID] = oem
	return oem, nil
}

// GetTruthState retrieves the truth state at a specific time
func (tl *TruthLoader) GetTruthState(objectID string, targetTime time.Time) (*physics.State, error) {
	oem, exists := tl.truthData[objectID]
	if !exists {
		return nil, fmt.Errorf("no truth data loaded for object %s", objectID)
	}

	// Find closest segment (linear interpolation could be added)
	var closest *OEMSegment
	minDelta := time.Hour * 24 * 365 // 1 year

	for i := range oem.Segments {
		delta := targetTime.Sub(oem.Segments[i].Epoch)
		if delta < 0 {
			delta = -delta
		}
		if delta < minDelta {
			minDelta = delta
			closest = &oem.Segments[i]
		}
	}

	if closest == nil {
		return nil, fmt.Errorf("no truth state found for time %v", targetTime)
	}

	return &closest.State, nil
}

// Validator performs propagator validation against truth data
type Validator struct {
	truthLoader *TruthLoader
	propagator  Propagator
}

// NewValidator creates a new propagator validator
func NewValidator(loader *TruthLoader, prop Propagator) *Validator {
	return &Validator{
		truthLoader: loader,
		propagator:  prop,
	}
}

// ComputeRICResiduals calculates L2 norm residuals in RIC frame
// This highlights WHERE the propagator is losing accuracy:
// - Radial errors = gravity model issues
// - In-track errors = drag or maneuver modeling issues
// - Cross-track errors = orbit plane determination issues
func (v *Validator) ComputeRICResiduals(predicted, truth physics.State) (radial, inTrack, crossTrack float64) {
	// ECI position residual
	deltaPos := [3]float64{
		predicted.Pos[0] - truth.Pos[0],
		predicted.Pos[1] - truth.Pos[1],
		predicted.Pos[2] - truth.Pos[2],
	}

	// RIC frame unit vectors
	// R (Radial): Points away from Earth center
	rMag := math.Sqrt(truth.Pos[0]*truth.Pos[0] + truth.Pos[1]*truth.Pos[1] + truth.Pos[2]*truth.Pos[2])
	radialDir := [3]float64{truth.Pos[0] / rMag, truth.Pos[1] / rMag, truth.Pos[2] / rMag}

	// C (Cross-track): Perpendicular to orbital plane (Pos × Vel)
	h := [3]float64{
		truth.Pos[1]*truth.Vel[2] - truth.Pos[2]*truth.Vel[1],
		truth.Pos[2]*truth.Vel[0] - truth.Pos[0]*truth.Vel[2],
		truth.Pos[0]*truth.Vel[1] - truth.Pos[1]*truth.Vel[0],
	}
	hMag := math.Sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2])
	crossTrackDir := [3]float64{h[0] / hMag, h[1] / hMag, h[2] / hMag}

	// I (In-track): Along velocity direction in orbital plane (C × R)
	inTrackDir := [3]float64{
		crossTrackDir[1]*radialDir[2] - crossTrackDir[2]*radialDir[1],
		crossTrackDir[2]*radialDir[0] - crossTrackDir[0]*radialDir[2],
		crossTrackDir[0]*radialDir[1] - crossTrackDir[1]*radialDir[0],
	}

	// Project residuals onto RIC frame
	radial = deltaPos[0]*radialDir[0] + deltaPos[1]*radialDir[1] + deltaPos[2]*radialDir[2]
	inTrack = deltaPos[0]*inTrackDir[0] + deltaPos[1]*inTrackDir[1] + deltaPos[2]*inTrackDir[2]
	crossTrack = deltaPos[0]*crossTrackDir[0] + deltaPos[1]*crossTrackDir[1] + deltaPos[2]*crossTrackDir[2]

	return
}

// ComputeL2Norm calculates total position error (L2 norm)
func (v *Validator) ComputeL2Norm(predicted, truth physics.State) float64 {
	dx := predicted.Pos[0] - truth.Pos[0]
	dy := predicted.Pos[1] - truth.Pos[1]
	dz := predicted.Pos[2] - truth.Pos[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz) // km
}

// ValidatePropagation performs end-to-end validation over time span
func (v *Validator) ValidatePropagation(objectID string, startTime, stopTime time.Time, step time.Duration) (*ValidationReport, error) {
	report := &ValidationReport{
		ObjectID:     objectID,
		StartTime:    startTime,
		StopTime:     stopTime,
		RICResiduals: make([]RICResidual, 0),
	}

	currentTime := startTime
	maxL2Error := 0.0

	for currentTime.Before(stopTime) || currentTime.Equal(stopTime) {
		// Get truth state
		truth, err := v.truthLoader.GetTruthState(objectID, currentTime)
		if err != nil {
			return nil, fmt.Errorf("truth lookup failed at %v: %w", currentTime, err)
		}

		// Propagate to this time
		minutesSinceEpoch := float64(currentTime.Sub(startTime).Minutes())
		predicted, err := v.propagator.Propagate(minutesSinceEpoch)
		if err != nil {
			return nil, fmt.Errorf("propagation failed at %v: %w", currentTime, err)
		}

		// Compute residuals
		radial, inTrack, crossTrack := v.ComputeRICResiduals(predicted, *truth)
		l2Norm := v.ComputeL2Norm(predicted, *truth)

		if l2Norm > maxL2Error {
			maxL2Error = l2Norm
		}

		report.RICResiduals = append(report.RICResiduals, RICResidual{
			Time:       currentTime,
			Radial:     radial,
			InTrack:    inTrack,
			CrossTrack: crossTrack,
			L2Norm:     l2Norm,
		})

		currentTime = currentTime.Add(step)
	}

	report.MaxL2Error = maxL2Error
	report.MeetsThreshold = maxL2Error < 0.01 // 10 meters = 0.01 km

	return report, nil
}

// ValidationReport contains residual analysis results
type ValidationReport struct {
	ObjectID       string
	StartTime      time.Time
	StopTime       time.Time
	RICResiduals   []RICResidual
	MaxL2Error     float64 // km
	MeetsThreshold bool
}

// RICResidual contains frame-specific errors
type RICResidual struct {
	Time       time.Time
	Radial     float64 // km
	InTrack    float64 // km
	CrossTrack float64 // km
	L2Norm     float64 // km
}

// Helper functions
func extractValue(line string) string {
	parts := strings.SplitN(line, "=", 2)
	if len(parts) == 2 {
		return strings.TrimSpace(parts[1])
	}
	return ""
}

func parseOEMTime(value string) time.Time {
	// ISO 8601 format: 2024-01-15T12:00:00.000
	t, _ := time.Parse("2006-01-02T15:04:05.000", value)
	return t
}

func parseFloat(value string) float64 {
	f, _ := strconv.ParseFloat(value, 64)
	return f
}

func parseEphemerisLine(line string) (OEMSegment, error) {
	// Format: EPOCH X Y Z VX VY VZ [AX AY AZ]
	fields := strings.Fields(line)
	if len(fields) < 7 {
		return OEMSegment{}, fmt.Errorf("invalid ephemeris line")
	}

	segment := OEMSegment{
		Epoch: parseOEMTime(fields[0]),
		State: physics.State{
			Pos: [3]float64{
				parseFloat(fields[1]),
				parseFloat(fields[2]),
				parseFloat(fields[3]),
			},
			Vel: [3]float64{
				parseFloat(fields[4]),
				parseFloat(fields[5]),
				parseFloat(fields[6]),
			},
		},
	}

	return segment, nil
}
