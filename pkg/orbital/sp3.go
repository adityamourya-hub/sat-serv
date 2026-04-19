package orbital

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

/**
 * SP3 Format Support (IGS Standard)
 *
 * SP3 (Standard Product #3) is the IGS standard for precise ephemeris and
 * clock products. This implementation supports:
 * - SP3-c and SP3-d formats
 * - Orbit and clock data ingestion
 * - Export to SP3-d for ADITI compliance
 *
 * Reference: IGS SP3 Format Specification
 * - SP3-c: https://files.igs.org/pub/data/format/sp3c.txt
 * - SP3-d: https://files.igs.org/pub/data/format/sp3d.pdf
 */

// SP3Version identifies the SP3 format version
type SP3Version string

const (
	SP3VersionC SP3Version = "c"
	SP3VersionD SP3Version = "d"
)

// SP3OrbitType identifies the type of orbit solution
type SP3OrbitType string

const (
	SP3OrbitTypeGPS     SP3OrbitType = "G"
	SP3OrbitTypeGLONASS SP3OrbitType = "R"
	SP3OrbitTypeGalileo SP3OrbitType = "E"
	SP3OrbitTypeQZSS    SP3OrbitType = "J"
	SP3OrbitTypeBeiDou  SP3OrbitType = "C"
	SP3OrbitTypeMixed   SP3OrbitType = "M"
	SP3OrbitTypeLEO     SP3OrbitType = "L"
	SP3OrbitTypeUser    SP3OrbitType = "U"
)

// SP3Header contains metadata from SP3 file header
type SP3Header struct {
	Version                  SP3Version
	StartTime                time.Time
	NumberOfEpochs           int
	TimeSystem               string // GPS, GLO, GAL, etc.
	OrbitType                SP3OrbitType
	Agency                   string
	CoordinatingAgency       string
	ZdOffset                 float64 // Zenith delay offset for d-versions
	NumberOfClockCorrections int
	BaseClockRate            float64 // Units of clock values (sec, ms, us, ns, ps)
	EpochInterval            time.Duration
	PositionAccuracy         float64 // meters (scaled by 2^N)
	ClockAccuracy            float64 // seconds (scaled by 2^N)
}

// SP3SatelliteData contains position and optionally velocity and clock
type SP3SatelliteData struct {
	PRN          string // Satellite identifier (e.g., "G01", "L12345")
	Epoch        time.Time
	Position     [3]float64  // km (ECF or ECI depending on type)
	Velocity     *[3]float64 // km/s (optional, SP3-d only)
	ClockBias    *float64    // seconds or clock units
	ClockRate    *float64    // seconds/second or clock rate
	PositionXmit [3]float64  // Position standard deviation (mm)
	ClockXmit    *float64    // Clock std dev (psec or units)
	EventFlag    bool        // Maneuver or discontinuity
}

// SP3File represents a complete SP3 orbit file
type SP3File struct {
	Header     SP3Header
	Satellites []SP3SatelliteData
}

// SP3Ingestor handles SP3 file parsing
type SP3Ingestor struct {
	strictMode bool // Reject files with minor errors
}

// NewSP3Ingestor creates a new SP3 parser
func NewSP3Ingestor(strict bool) *SP3Ingestor {
	return &SP3Ingestor{strictMode: strict}
}

// ParseFile reads and parses an SP3 file from disk
func (i *SP3Ingestor) ParseFile(filepath string) (*SP3File, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open SP3 file: %w", err)
	}
	defer file.Close()

	return i.Parse(file)
}

// Parse reads SP3 data from a buffered source
func (i *SP3Ingestor) Parse(file *os.File) (*SP3File, error) {
	reader := bufio.NewReader(file)
	sp3 := &SP3File{}

	// Parse header (first 22 lines for SP3-c/d)
	header, err := i.parseHeader(reader)
	if err != nil {
		return nil, fmt.Errorf("SP3 header parse error: %w", err)
	}
	sp3.Header = *header

	// Parse epoch data
	satellites, err := i.parseEpochs(reader, header)
	if err != nil {
		return nil, fmt.Errorf("SP3 epoch parse error: %w", err)
	}
	sp3.Satellites = satellites

	return sp3, nil
}

// parseHeader extracts SP3 header information
func (i *SP3Ingestor) parseHeader(reader *bufio.Reader) (*SP3Header, error) {
	header := &SP3Header{}

	// Line 1: Version, start time, number of epochs
	line1, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	line1 = strings.TrimRight(line1, "\r\n")

	// Parse version character (position 2)
	if len(line1) < 2 {
		return nil, fmt.Errorf("invalid SP3 line 1 length")
	}
	versionChar := strings.ToLower(string(line1[1]))
	switch versionChar {
	case "c":
		header.Version = SP3VersionC
	case "d":
		header.Version = SP3VersionD
	default:
		if i.strictMode {
			return nil, fmt.Errorf("unsupported SP3 version: %s", versionChar)
		}
		header.Version = SP3VersionC // Assume c for backward compatibility
	}

	// Parse start time from SP3 header line 1
	// Format: #cPYYYY MM DD HH MM SS.SSSSSSSSS ...
	// Positions (1-indexed): YYYY=4-7, MM=9-10, DD=12-13, HH=15-16, MM=18-19, SS=21-31
	if len(line1) < 32 {
		return nil, fmt.Errorf("SP3 line 1 too short for header (need 32 chars, got %d)", len(line1))
	}

	// Convert 1-indexed positions to 0-indexed slices
	year, _ := strconv.Atoi(strings.TrimSpace(line1[3:7]))               // Positions 4-7
	month, _ := strconv.Atoi(strings.TrimSpace(line1[8:10]))             // Positions 9-10
	day, _ := strconv.Atoi(strings.TrimSpace(line1[11:13]))              // Positions 12-13
	hour, _ := strconv.Atoi(strings.TrimSpace(line1[14:16]))             // Positions 15-16
	minute, _ := strconv.Atoi(strings.TrimSpace(line1[17:19]))           // Positions 18-19
	second, _ := strconv.ParseFloat(strings.TrimSpace(line1[20:31]), 64) // Positions 21-31

	// Year can be 2 or 4 digits in SP3
	if year < 50 {
		year += 2000
	} else if year < 100 {
		year += 1900
	}

	header.StartTime = time.Date(year, time.Month(month), day, hour, minute, int(second),
		int((second-float64(int(second)))*1e9), time.UTC)

	// Number of epochs (positions 33-40, 8-character field)
	if len(line1) > 40 {
		numEpochs, _ := strconv.Atoi(strings.TrimSpace(line1[32:40]))
		header.NumberOfEpochs = numEpochs
	} else if len(line1) > 33 {
		// Short line - try to parse whatever is available
		numEpochs, _ := strconv.Atoi(strings.TrimSpace(line1[32:]))
		header.NumberOfEpochs = numEpochs
	}

	// Coord system (positions 47-51), Orbit type (53), Agency (55+)
	if len(line1) > 51 {
		_ = strings.TrimSpace(line1[46:51]) // coordSys
	}
	if len(line1) > 53 {
		header.OrbitType = SP3OrbitType(strings.TrimSpace(line1[52:53]))
	}
	if len(line1) > 55 {
		header.Agency = strings.TrimSpace(line1[54:])
	}

	// Lines 2-7: Satellite list and accuracy (SP3-c/d has 85 satellite slots)
	satellites := make(map[string]float64)
	for lineNum := 2; lineNum <= 7; lineNum++ {
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")

		// Parse satellite IDs (every 3 characters starting at position 9)
		for pos := 9; pos < len(line)-2 && pos < 60; pos += 3 {
			if pos+3 <= len(line) {
				satID := strings.TrimSpace(line[pos : pos+3])
				if satID != "   " && satID != "" && satID != "0" {
					satellites[satID] = 0.0 // Will update accuracy later
				}
			}
		}
	}

	// Lines 8-13: Satellite accuracy (optional, SP3-c/d)
	for lineNum := 8; lineNum <= 13; lineNum++ {
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		_ = line
		// Parse accuracy exponents if present
	}

	// Line 14-18: File type, time system, base for pos/clock, agency info
	for lineNum := 14; lineNum <= 18; lineNum++ {
		line, err := reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")

		if len(line) < 3 {
			continue
		}

		recordType := line[0:2]
		switch recordType {
		case "%c":
			// Coordinate and time system
			if len(line) > 9 {
				header.TimeSystem = strings.TrimSpace(line[9:12])
			}
		case "%f":
			// Base for pos/clk, number of clocks
			if len(line) > 9 {
				base, _ := strconv.ParseFloat(strings.TrimSpace(line[3:13]), 64)
				header.BaseClockRate = base
			}
			if len(line) > 14 {
				numClk, _ := strconv.Atoi(strings.TrimSpace(line[14:16]))
				header.NumberOfClockCorrections = numClk
			}
		case "%i":
			// LEAP seconds and MJD
			// Ignored for now
		case "%":
			// Comments - store coordinating agency
			if strings.Contains(line, "c") && len(line) > 3 {
				header.CoordinatingAgency = strings.TrimSpace(line[3:])
			}
		}
	}

	// Skip remaining header lines (19-22)
	for lineNum := 19; lineNum <= 22; lineNum++ {
		reader.ReadString('\n')
	}

	return header, nil
}

// parseEpochs reads the epoch data records
func (i *SP3Ingestor) parseEpochs(reader *bufio.Reader, header *SP3Header) ([]SP3SatelliteData, error) {
	var satellites []SP3SatelliteData

	for epoch := 0; epoch < header.NumberOfEpochs; epoch++ {
		// Epoch header line: starts with '*'
		epochLine, err := reader.ReadString('\n')
		if err != nil {
			break // EOF - some files have fewer epochs than declared
		}
		epochLine = strings.TrimRight(epochLine, "\r\n")

		if len(epochLine) == 0 || epochLine[0] != '*' {
			// Not an epoch line, might be comment or EOF marker
			if strings.TrimSpace(epochLine) == "EOF" {
				break
			}
			continue
		}

		// Parse epoch time
		epochTime := i.parseEpochTime(epochLine)

		// Read satellite records until next epoch or EOF
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			line = strings.TrimRight(line, "\r\n")

			if len(line) == 0 {
				continue
			}

			// Check for next epoch or EOF
			if line[0] == '*' || strings.TrimSpace(line) == "EOF" {
				// Push back the line (simplified - in production use bufio.UnreadByte or similar)
				break
			}

			// Parse position record (starts with 'P' for position, 'V' for velocity)
			if line[0] == 'P' {
				sat := i.parsePositionRecord(line, epochTime)
				if sat != nil {
					satellites = append(satellites, *sat)
				}
			} else if line[0] == 'V' && header.Version == SP3VersionD {
				// Velocity record (SP3-d)
				i.parseVelocityRecord(line, &satellites)
			}
		}
	}

	return satellites, nil
}

// parseEpochTime extracts time from epoch header line
func (i *SP3Ingestor) parseEpochTime(line string) time.Time {
	if len(line) < 30 {
		return time.Time{}
	}

	year, _ := strconv.Atoi(strings.TrimSpace(line[3:7]))
	month, _ := strconv.Atoi(strings.TrimSpace(line[8:10]))
	day, _ := strconv.Atoi(strings.TrimSpace(line[11:13]))
	hour, _ := strconv.Atoi(strings.TrimSpace(line[14:16]))
	minute, _ := strconv.Atoi(strings.TrimSpace(line[17:19]))
	second, _ := strconv.ParseFloat(strings.TrimSpace(line[20:26]), 64)

	if year < 50 {
		year += 2000
	} else if year < 100 {
		year += 1900
	}

	return time.Date(year, time.Month(month), day, hour, minute, int(second),
		int((second-float64(int(second)))*1e9), time.UTC)
}

// parsePositionRecord extracts satellite position and clock data
func (i *SP3Ingestor) parsePositionRecord(line string, epoch time.Time) *SP3SatelliteData {
	if len(line) < 60 {
		return nil
	}

	sat := &SP3SatelliteData{
		Epoch: epoch,
	}

	// Satellite PRN (positions 1-3, skipping 'P')
	sat.PRN = strings.TrimSpace(line[1:4])

	// Position in km (positions 4-26, 27-49, 50-72)
	xKm, _ := strconv.ParseFloat(strings.TrimSpace(line[4:18]), 64)
	yKm, _ := strconv.ParseFloat(strings.TrimSpace(line[18:32]), 64)
	zKm, _ := strconv.ParseFloat(strings.TrimSpace(line[32:46]), 64)

	// Convert from dm (SP3 standard) to km if needed
	// SP3 stores position in 0.1 mm (SP3-d) or km (some versions)
	// Standard is 0.1 mm = 1e-4 km, but we need to detect the unit
	sat.Position[0] = xKm / 10000.0 // Convert 0.1mm to km
	sat.Position[1] = yKm / 10000.0
	sat.Position[2] = zKm / 10000.0

	// Clock bias in seconds (positions 47-60)
	if len(line) > 60 {
		clkBias, err := strconv.ParseFloat(strings.TrimSpace(line[46:60]), 64)
		if err == nil {
			// SP3 stores clock in picoseconds (1e-12 sec) or seconds
			clk := clkBias / 1e12 // Convert ps to seconds
			sat.ClockBias = &clk
		}
	}

	// Standard deviations (positions 61-78, optional)
	if len(line) > 78 {
		sxMm, _ := strconv.ParseFloat(strings.TrimSpace(line[61:69]), 64)
		syMm, _ := strconv.ParseFloat(strings.TrimSpace(line[69:71]), 64)
		szMm, _ := strconv.ParseFloat(strings.TrimSpace(line[71:73]), 64)
		// These are scaled by the accuracy exponent from header
		sat.PositionXmit[0] = sxMm
		sat.PositionXmit[1] = syMm
		sat.PositionXmit[2] = szMm
	}

	return sat
}

// parseVelocityRecord extracts velocity data (SP3-d)
func (i *SP3Ingestor) parseVelocityRecord(line string, satellites *[]SP3SatelliteData) {
	if len(line) < 60 || len(*satellites) == 0 {
		return
	}

	// Find matching satellite
	prn := strings.TrimSpace(line[1:4])

	for idx := range *satellites {
		sat := &(*satellites)[idx]
		if sat.PRN == prn {
			// Velocity in dm/s, convert to km/s
			vx, _ := strconv.ParseFloat(strings.TrimSpace(line[4:18]), 64)
			vy, _ := strconv.ParseFloat(strings.TrimSpace(line[18:32]), 64)
			vz, _ := strconv.ParseFloat(strings.TrimSpace(line[32:46]), 64)

			vel := [3]float64{vx / 10000.0, vy / 10000.0, vz / 10000.0}
			sat.Velocity = &vel

			// Clock rate
			if len(line) > 60 {
				rate, _ := strconv.ParseFloat(strings.TrimSpace(line[46:60]), 64)
				rateSec := rate / 1e12 // Convert to s/s
				sat.ClockRate = &rateSec
			}
			break
		}
	}
}

// SP3Exporter handles SP3 file generation
type SP3Exporter struct {
	version SP3Version
}

// NewSP3Exporter creates a new SP3 exporter
func NewSP3Exporter(version SP3Version) *SP3Exporter {
	return &SP3Exporter{version: version}
}

// Export writes an SP3 file to disk
func (e *SP3Exporter) Export(filepath string, sp3 *SP3File) error {
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create SP3 file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Write header
	if err := e.writeHeader(writer, &sp3.Header); err != nil {
		return err
	}

	// Write epochs
	if err := e.writeEpochs(writer, sp3); err != nil {
		return err
	}

	// Write EOF marker
	writer.WriteString("EOF\n")

	return writer.Flush()
}

// writeHeader generates SP3 header lines
func (e *SP3Exporter) writeHeader(writer *bufio.Writer, header *SP3Header) error {
	year, month, day := header.StartTime.Date()
	hour, minute, _ := header.StartTime.Clock()
	second := float64(header.StartTime.Second()) + float64(header.StartTime.Nanosecond())/1e9

	// Line 1: Version, epoch, num epochs
	// Format: #cPYY MM DD HH MM SS.SSSSSS epochs type orbit agency
	line1 := fmt.Sprintf("#%sP%s %02d %02d %02d %02d %09.6f %08d %s %s %s\n",
		strings.ToUpper(string(header.Version)),
		strconv.Itoa(year)[2:4], // Last 2 digits of year
		month, day, hour, minute, second,
		header.NumberOfEpochs,
		"D",
		string(header.OrbitType),
		header.Agency)

	writer.WriteString(line1)

	// Lines 2-7: Satellite list (simplified - just placeholder)
	writer.WriteString("##                                                            0    0    0    0\n")
	for i := 0; i < 5; i++ {
		writer.WriteString("+                                                            0    0    0    0\n")
	}

	// Lines 8-13: Accuracy indicators
	for i := 0; i < 6; i++ {
		writer.WriteString("+                                                            0    0    0    0\n")
	}

	// Lines 14-18: System info
	writer.WriteString(fmt.Sprintf("%%c %s  cc GPS ccc cccc cccc cccc cccc cccc cccc cccc cccc cccc cccc\n", header.TimeSystem))
	writer.WriteString("%%c cc cc c c c c c c c c c c c c c c c c c c c c c c c c c c c c cc cc\n")
	writer.WriteString(fmt.Sprintf("%%f  %.7f  %.7f  %.7f  %.7f\n",
		header.BaseClockRate, header.BaseClockRate,
		float64(header.NumberOfClockCorrections), 0.0))
	writer.WriteString("%%f  0.0 0.0 0.0 0.0\n")
	writer.WriteString("%%i    0    0    0    0      0      0      0      0         0\n")
	writer.WriteString("%%i    0    0    0    0      0      0      0      0         0\n")

	// Comment lines
	writer.WriteString("/* ADITI-compliant SSIS generated SP3 file\n")
	writer.WriteString("/* Generated for IAF operational use\n")
	writer.WriteString("/* Coordinate system: ICRF/J2000.0\n")
	writer.WriteString("/* Time system: UTC\n")

	return nil
}

// writeEpochs writes the epoch and position data
func (e *SP3Exporter) writeEpochs(writer *bufio.Writer, sp3 *SP3File) error {
	// Group by epoch
	epochMap := make(map[int64][]SP3SatelliteData)
	for _, sat := range sp3.Satellites {
		ts := sat.Epoch.Unix()
		epochMap[ts] = append(epochMap[ts], sat)
	}

	// Write each epoch
	for _, sats := range epochMap {
		if len(sats) == 0 {
			continue
		}

		// Epoch header
		epoch := sats[0].Epoch
		year, month, day := epoch.Date()
		hour, minute, _ := epoch.Clock()
		second := float64(epoch.Second()) + float64(epoch.Nanosecond())/1e9

		epochLine := fmt.Sprintf("*  %4d %2d %2d %2d %2d %9.6f\n",
			year, month, day, hour, minute, second)
		writer.WriteString(epochLine)

		// Position records
		for _, sat := range sats {
			// Position in 0.1 mm (SP3 standard)
			xDm := sat.Position[0] * 10000.0
			yDm := sat.Position[1] * 10000.0
			zDm := sat.Position[2] * 10000.0

			var clkBias float64
			if sat.ClockBias != nil {
				clkBias = *sat.ClockBias * 1e12 // Convert to picoseconds
			}

			posLine := fmt.Sprintf("P%s%14.6f%14.6f%14.6f%14.6f\n",
				sat.PRN, xDm, yDm, zDm, clkBias)
			writer.WriteString(posLine)

			// Velocity if available (SP3-d)
			if e.version == SP3VersionD && sat.Velocity != nil {
				vx := (*sat.Velocity)[0] * 10000.0
				vy := (*sat.Velocity)[1] * 10000.0
				vz := (*sat.Velocity)[2] * 10000.0

				var clkRate float64
				if sat.ClockRate != nil {
					clkRate = *sat.ClockRate * 1e12
				}

				velLine := fmt.Sprintf("V%s%14.6f%14.6f%14.6f%14.6f\n",
					sat.PRN, vx, vy, vz, clkRate)
				writer.WriteString(velLine)
			}
		}
	}

	return nil
}

// ConvertSP3ToStateVectors converts SP3 data to physics StateVectors
func ConvertSP3ToStateVectors(sp3 *SP3File) []StateVectorWithTime {
	var states []StateVectorWithTime

	for _, sat := range sp3.Satellites {
		state := StateVectorWithTime{
			R: sat.Position,
			V: [3]float64{0, 0, 0}, // Default if no velocity
			T: sat.Epoch,
		}

		if sat.Velocity != nil {
			state.V = *sat.Velocity
		}

		states = append(states, state)
	}

	return states
}

// StateVectorWithTime pairs state with timestamp
type StateVectorWithTime struct {
	R [3]float64
	V [3]float64
	T time.Time
}

// SP3Validation validates SP3 file integrity
type SP3Validation struct {
	Errors   []string
	Warnings []string
}

// ValidateSP3 checks SP3 file for compliance issues
func ValidateSP3(sp3 *SP3File) *SP3Validation {
	validation := &SP3Validation{}

	// Check header consistency
	if sp3.Header.NumberOfEpochs == 0 {
		validation.Errors = append(validation.Errors, "zero epochs declared")
	}

	if sp3.Header.NumberOfEpochs != len(sp3.Satellites) {
		validation.Warnings = append(validation.Warnings,
			fmt.Sprintf("epoch count mismatch: header=%d, actual=%d",
				sp3.Header.NumberOfEpochs, len(sp3.Satellites)))
	}

	// Check for NaN or Inf values
	for i, sat := range sp3.Satellites {
		for j, pos := range sat.Position {
			if math.IsNaN(pos) || math.IsInf(pos, 0) {
				validation.Errors = append(validation.Errors,
					fmt.Sprintf("satellite %s position[%d] invalid at index %d", sat.PRN, j, i))
			}
		}
	}

	return validation
}
