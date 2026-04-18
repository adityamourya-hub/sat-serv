package physics

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

/**
 * ParseTLE provides the primary entry point for converting raw TLE strings into structured objects.
 * Moved to physics substrate for architectural decoupling (ADITI 4.0).
 */
func ParseTLE(data string) (*TLE, error) {
	fmt.Printf("[TRC] Entering ParseTLE\n")
	defer fmt.Printf("[TRC] Exiting ParseTLE\n")

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

func parseLine1(line string, tle *TLE) error {
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

	yearStr := line[18:20]
	year, _ := strconv.Atoi(yearStr)
	if year < 57 {
		tle.EpochYear = 2000 + year
	} else {
		tle.EpochYear = 1900 + year
	}

	dayVal, err := strconv.ParseFloat(strings.TrimSpace(line[20:32]), 64)
	if err != nil {
		return fmt.Errorf("invalid epoch day: %v", err)
	}
	tle.EpochDay = dayVal

	tle.MeanMotionDeriv1, _ = strconv.ParseFloat(strings.TrimSpace(line[33:43]), 64)
	tle.MeanMotionDeriv2 = parseTLEFloat(line[44:52])
	tle.BStar = parseTLEFloat(line[53:61])

	tle.ElementNumber, _ = strconv.Atoi(strings.TrimSpace(line[64:68]))

	return nil
}

func parseLine2(line string, tle *TLE) error {
	trimmed := strings.TrimSpace(line)
	if len(trimmed) < 69 {
		return fmt.Errorf("line 2 too short")
	}
	line = trimmed

	if line[0] != '2' {
		return fmt.Errorf("line 2 must start with '2'")
	}

	tle.Inclination, _ = strconv.ParseFloat(strings.TrimSpace(line[8:16]), 64)
	tle.RAAN, _ = strconv.ParseFloat(strings.TrimSpace(line[17:25]), 64)
	tle.Eccentricity, _ = strconv.ParseFloat("0."+strings.TrimSpace(line[26:33]), 64)
	tle.ArgOfPerigee, _ = strconv.ParseFloat(strings.TrimSpace(line[34:42]), 64)
	tle.MeanAnomaly, _ = strconv.ParseFloat(strings.TrimSpace(line[43:51]), 64)
	tle.MeanMotion, _ = strconv.ParseFloat(strings.TrimSpace(line[52:63]), 64)
	tle.RevAtEpoch, _ = strconv.Atoi(strings.TrimSpace(line[63:68]))

	return nil
}

func parseTLEFloat(s string) float64 {
	if len(s) != 8 {
		return 0
	}
	sign := 1.0
	if s[0] == '-' {
		sign = -1.0
	}
	mantissaStr := strings.TrimSpace(s[1:6])
	mantissaVal, _ := strconv.ParseFloat("0."+mantissaStr, 64)
	expSign := 1.0
	if s[6] == '-' {
		expSign = -1.0
	}
	expVal, _ := strconv.Atoi(string(s[7]))
	return sign * mantissaVal * math.Pow(10, expSign*float64(expVal))
}

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
	return "UNK"
}
