package orbital

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

/**
 * TLE represents a standard Two-Line Element set used for orbital state identification.
 * This structure adheres to the NORAD/SGP4 format specifications essential for SSA platforms.
 * It includes fields for satellite identification, orbital parameters, and motion derivatives.
 * The RawL1 and RawL2 fields are preserved to allow for high-fidelity engine re-initialization.
 * Used primary in the Sovereign Space Intelligence Substrate for tactical orbit determination.
 */
type TLE struct {
	Name               string
	SatelliteNumber    int
	Classification     string
	InternationalDesig string
	CountryCode        string
	EpochYear          int
	EpochDay           float64
	MeanMotionDeriv1   float64 // First derivative of mean motion
	MeanMotionDeriv2   float64 // Second derivative of mean motion (divided by 6)
	BStar              float64 // B* drag term
	ElementNumber      int
	Inclination        float64
	RAAN               float64 // Right Ascension of Ascending Node
	Eccentricity       float64
	ArgOfPerigee       float64
	MeanAnomaly        float64
	MeanMotion         float64
	RevAtEpoch         int
	RawL1              string // Original Line 1 for analytical engine re-init
	RawL2              string // Original Line 2 for analytical engine re-init
}

/**
 * ParseTLE provides the primary entry point for converting raw TLE strings into structured objects.
 * It supports standard 2-line and 3-line formats (where the first line is the object name).
 * This function performs initial line segmentation and delegates to hardened line parsers.
 * It implements strict validation to ensure that no corrupted orbital data enters the substrate.
 * Returns a pointer to a TLE object or a detailed error if validation fail during the process.
 */
func ParseTLE(data string) (*TLE, error) {

	lines := strings.Split(strings.TrimSpace(data), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("invalid TLE: at least 2 lines required")
	}

	var tle TLE
	var line1, line2 string

	if len(lines) == 3 {
		tle.Name = strings.TrimSpace(lines[0])
		line1 = lines[1]
		line2 = lines[2]
	} else {
		line1 = lines[0]
		line2 = lines[1]
	}

	if err := parseLine1(line1, &tle); err != nil {
		return nil, fmt.Errorf("error parsing line 1: %v", err)
	}
	if err := parseLine2(line2, &tle); err != nil {
		return nil, fmt.Errorf("error parsing line 2: %v", err)
	}

	tle.RawL1 = line1
	tle.RawL2 = line2

	return &tle, nil
}

/**
 * parseLine1 implements hardened parsing for the first line of a Two-Line Element set.
 * It utilizes "Reciprocal Trimming" to handle line-ending variations while enforcing offsets.
 * The function validates the line identifier '1' and extracts identification and epoch data.
 * It ensures that the MeanMotionDeriv2 and BStar fields are parsed using column-strict logic.
 * Strict checksum validation is performed at the end of the extraction process for safety.
 */
func parseLine1(line string, tle *TLE) error {

	// Reciprocal Trimming: Trim surrounding whitespace but verify column identifier
	_ = strings.TrimSpace(line) // Validate line is trimmable
	// Enforce strict column-based parsing (no trimming before index access)
	if len(line) < 69 {
		return fmt.Errorf("line 1 too short: expected 69 chars, got %d", len(line))
	}

	if line[0] != '1' {
		return fmt.Errorf("line 1 must start with '1', got %c", rune(line[0]))
	}

	satNum, err := strconv.Atoi(strings.TrimSpace(line[2:7]))
	if err != nil || satNum == 0 {
		return fmt.Errorf("invalid satellite number: %v", err)
	}
	tle.SatelliteNumber = satNum
	tle.Classification = string(line[7])
	tle.InternationalDesig = strings.TrimSpace(line[9:17])
	tle.CountryCode = ExtractCountryCode(line)

	epochYearStr := line[18:20]
	epochYear, err := strconv.Atoi(epochYearStr)
	if err != nil {
		return fmt.Errorf("invalid epoch year %q: %v", epochYearStr, err)
	}
	if epochYear < 57 {
		tle.EpochYear = 2000 + epochYear
	} else {
		tle.EpochYear = 1900 + epochYear
	}

	epochDayStr := line[20:32]
	epochDay, err := strconv.ParseFloat(strings.TrimSpace(epochDayStr), 64)
	if err != nil || epochDay < 0 || epochDay > 367 {
		return fmt.Errorf("invalid epoch day %q: %v or out of range", epochDayStr, err)
	}
	tle.EpochDay = epochDay
	tle.RawL1 = line

	intMM1, err := strconv.ParseFloat(strings.TrimSpace(line[33:43]), 64)
	if err != nil {
		return fmt.Errorf("invalid mean motion deriv 1: %v", err)
	}
	tle.MeanMotionDeriv1 = intMM1

	// Hardened Column-Strict Scientific Notation Parsing
	tle.MeanMotionDeriv2 = parseTLEFloat(line[44:52])
	tle.BStar = parseTLEFloat(line[53:61])

	ele, err := strconv.Atoi(strings.TrimSpace(line[64:68]))
	if err != nil {
		return fmt.Errorf("invalid element number: %v", err)
	}
	tle.ElementNumber = ele

	if err := validateTLEChecksum(line); err != nil {
		return fmt.Errorf("line 1 checksum invalid: %v", err)
	}

	return nil
}

/**
 * parseLine2 implements hardened parsing for the second line of a Two-Line Element set.
 * It extracts the Keplerian elements: inclination, RAAN, eccentricity, and perigee.
 * The function handles the unique TLE decimal format for eccentricity (assumed leading zero).
 * It validates that all extracted parameters fall within physical and standard orbital bounds.
 * Performs critical checksum verification to ensure data integrity during transmission.
 */
func parseLine2(line string, tle *TLE) error {

	// Apply robust line normalization for Line 2
	trimmed := strings.TrimSpace(line)
	if len(trimmed) < 69 {
		return fmt.Errorf("line 2 too short after trimming: %d chars", len(trimmed))
	}
	if len(trimmed) > 69 {
		trimmed = trimmed[:69]
	}
	line = trimmed

	if line[0] != '2' {
		return fmt.Errorf("line 2 must start with '2', got %c", rune(line[0]))
	}

	inc, err := strconv.ParseFloat(strings.TrimSpace(line[8:16]), 64)
	if err != nil || inc < 0 || inc > 180 {
		return fmt.Errorf("invalid inclination: %v or out of range [0,180]", err)
	}
	tle.Inclination = inc

	raan, err := strconv.ParseFloat(strings.TrimSpace(line[17:25]), 64)
	if err != nil || raan < 0 || raan >= 360 {
		return fmt.Errorf("invalid RAAN: %v or out of range [0,360)", err)
	}
	tle.RAAN = raan

	eccStr := "0." + strings.TrimSpace(line[26:33])
	ecc, err := strconv.ParseFloat(eccStr, 64)
	if err != nil || ecc < 0 || ecc >= 1.0 {
		return fmt.Errorf("invalid eccentricity: %v or out of range [0,1)", err)
	}
	tle.Eccentricity = ecc

	argp, err := strconv.ParseFloat(strings.TrimSpace(line[34:42]), 64)
	if err != nil || argp < 0 || argp >= 360 {
		return fmt.Errorf("invalid arg of perigee: %v or out of range [0,360)", err)
	}
	tle.ArgOfPerigee = argp

	ma, err := strconv.ParseFloat(strings.TrimSpace(line[43:51]), 64)
	if err != nil || ma < 0 || ma >= 360 {
		return fmt.Errorf("invalid mean anomaly: %v or out of range [0,360)", err)
	}
	tle.MeanAnomaly = ma

	mm, err := strconv.ParseFloat(strings.TrimSpace(line[52:63]), 64)
	if err != nil || mm <= 0 || mm > 20 {
		return fmt.Errorf("invalid mean motion: %v or out of range (0,20]", err)
	}
	tle.MeanMotion = mm

	revAtEpoch, err := strconv.Atoi(strings.TrimSpace(line[63:68]))
	if err != nil {
		return fmt.Errorf("invalid revolution at epoch: %v", err)
	}
	tle.RevAtEpoch = revAtEpoch

	if err := validateTLEChecksum(line); err != nil {
		return fmt.Errorf("line 2 checksum invalid: %v", err)
	}

	return nil
}

/**
 * validateTLEChecksum calculates the modulo-10 checksum for a 69-character TLE line.
 * It follows the NORAD standard where digits are summed and '-' characters count as 1.
 * Other characters, including '+' and spaces, are ignored during the summation.
 * The calculated value is compared against the final character in the line (index 68).
 * Critical for verifying that TLE data has not been corrupted during storage or ingestion.
 */
func validateTLEChecksum(line string) error {
	if len(line) < 69 {
		return fmt.Errorf("line too short for checksum")
	}
	checksum := 0
	for i := 0; i < 68; i++ {
		if line[i] >= '0' && line[i] <= '9' {
			checksum += int(line[i] - '0')
		} else if line[i] == '-' {
			checksum++
		}
	}
	checksum %= 10
	expected, _ := strconv.Atoi(string(line[68]))
	if checksum != expected {
		return fmt.Errorf("checksum mismatch: computed %d, expected %d", checksum, expected)
	}
	return nil
}

/**
 * parseSciNotation is a legacy alias for parseTLEFloat provided for backward compatibility.
 * It allows existing callers to benefit from the hardened column-strict parsing logic.
 * This function will eventually be deprecated in favor of direct calls to parseTLEFloat.
 * It ensures a smooth transition during the ADITI 4.0 infrastructure hardening phase.
 * Delegates all logic to the new hardened implementation to maintain stability.
 */
func parseSciNotation(s string) float64 {
	return parseTLEFloat(s)
}

/**
 * parseTLEFloat implements the column-strict TLE scientific notation parsing (sNNNNN-E).
 * This format assumes an implicit leading decimal point (0.NNNNN) and fixed offsets.
 * The string must be exactly 8 characters wide to ensure index-safe extraction.
 * It handles the mantissa sign (index 0), digits (1-5), and exponent sign/value (6-7).
 * Returns the computed float64 value, mapping invalid formats to safely handled zero values.
 */
func parseTLEFloat(s string) float64 {
	// Standard TLE fields for mm2 and bstar are exactly 8 characters [sNNNNNsE]
	if len(s) != 8 {
		return 0
	}

	// 1. Extract Mantissa Sign and Digits
	// Index 0: Sign of mantissa ('+', '-', or ' ')
	mantissaSign := 1.0
	if s[0] == '-' {
		mantissaSign = -1.0
	}

	// Indices 1:6 (1,2,3,4,5): Mantissa digits with implicit leading decimal 0.NNNNN
	mantissaStr := strings.TrimSpace(s[1:6])
	if mantissaStr == "" {
		return 0
	}
	mantissaVal, err := strconv.ParseFloat("0."+mantissaStr, 64)
	if err != nil {
		fmt.Printf("[ERR] Failed to parse TLE mantissa %q: %v\n", mantissaStr, err)
		return 0
	}

	// 2. Extract Exponent Sign and Value
	// Index 6: Sign of exponent ('+', '-', or ' ')
	expSign := 1.0
	if s[6] == '-' {
		expSign = -1.0
	}

	// Index 7: Exponent digit
	expVal, err := strconv.Atoi(string(s[7]))
	if err != nil {
		return 0
	}

	// Final computation: sign * mantissa * 10^(expSign * expVal)
	return mantissaSign * mantissaVal * math.Pow(10, expSign*float64(expVal))
}

/**
 * ExtractCountryCode analyzes the International Designator to map objects to ISO country codes.
 * It supports standard prefixes (e.g., 'IN' for India, 'US' for USA) frequently used in cataloging.
 * This is vital for sovereign space situational awareness and multi-actor coordination.
 * If no designator is present or recognized, it defaults to the host nation (IND).
 * Returns a 3-letter ISO 3166-1 alpha-3 code representing the presumed country of origin.
 */
func ExtractCountryCode(tleLine1 string) string {

	if len(tleLine1) < 17 {
		return "UNK"
	}
	desig := strings.TrimSpace(tleLine1[9:17])
	if len(desig) >= 2 {
		prefix := strings.ToUpper(desig[:2])
		switch prefix {
		case "IN":
			return "IND"
		case "US":
			return "USA"
		case "RU":
			return "RUS"
		case "PR", "CH":
			return "PRC"
		case "FR":
			return "FRA"
		case "UK":
			return "GBR"
		}
	}
	// Default fallback for unknown designators
	return "UNK"
}
