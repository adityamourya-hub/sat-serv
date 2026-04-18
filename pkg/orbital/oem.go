package orbital

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
	"time"

	"ssis/pkg/physics"
)

/**
 * OEMDataLine represents a single state measurement line within a CCSDS OEM file.
 * Following the Blue Book standard, it contains the UTC epoch and the physics-substrate state.
 */
type OEMDataLine struct {
	Epoch time.Time
	State physics.State
}

/**
 * OEM represents the parsed CCSDS Orbit Ephemeris Message.
 * It encapsulates metadata and a sequence of data points in the physics schema.
 */
type OEM struct {
	ObjectName string
	ObjectID   string
	RefFrame   string
	TimeSystem string
	StartTime  time.Time
	StopTime   time.Time
	Data       []OEMDataLine
}

/**
 * ParseOEM converts a raw CCSDS OEM KVN (Key-Value Notation) string into a structured object.
 * It implements a multi-stage parser that identifies metadata blocks before stream-reading ephemeris lines.
 */
func ParseOEM(data string) (*OEM, error) {
	fmt.Printf("[TRC] Entering ParseOEM with data length: %d\n", len(data))
	defer fmt.Printf("[TRC] Exiting ParseOEM\n")

	scanner := bufio.NewScanner(strings.NewReader(data))
	oem := &OEM{}
	inData := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "COMMENT") {
			continue
		}

		if line == "META_STOP" {
			inData = true
			continue
		}

		if !inData {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) < 2 {
				continue
			}
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])

			switch key {
			case "OBJECT_NAME":
				oem.ObjectName = val
			case "OBJECT_ID":
				oem.ObjectID = val
			case "REF_FRAME":
				oem.RefFrame = val
			case "TIME_SYSTEM":
				oem.TimeSystem = val
			case "START_TIME":
				oem.StartTime, _ = time.Parse("2006-01-02T15:04:05.000", val)
			case "STOP_TIME":
				oem.StopTime, _ = time.Parse("2006-01-02T15:04:05.000", val)
			}
		} else {
			// Ephemeris Data Line: EPOCH X Y Z VX VY VZ
			fields := strings.Fields(line)
			if len(fields) < 7 {
				continue
			}
			epoch, err := time.Parse("2006-01-02T15:04:05.000", fields[0])
			if err != nil {
				continue
			}
			vals := make([]float64, 6)
			for i := 0; i < 6; i++ {
				vals[i], _ = strconv.ParseFloat(fields[i+1], 64)
			}
			oem.Data = append(oem.Data, OEMDataLine{
				Epoch: epoch,
				State: physics.State{
					Pos: physics.Vector3{vals[0], vals[1], vals[2]},
					Vel: physics.Vector3{vals[3], vals[4], vals[5]},
				},
			})
		}
	}

	return oem, nil
}

/**
 * OEMPropagator implements the Propagator interface using high-fidelity data interpolation.
 */
type OEMPropagator struct {
	oem *OEM
}

/**
 * NewOEMPropagator initializes a propagator from a valid OEM structure.
 */
func NewOEMPropagator(oem *OEM) (*OEMPropagator, error) {
	if len(oem.Data) < 4 {
		return nil, fmt.Errorf("insufficient data points in OEM (min 4 required)")
	}
	return &OEMPropagator{oem: oem}, nil
}

/**
 * Propagate reconstructs the satellite state at a relative offset from the OEM StartTime.
 */
func (p *OEMPropagator) Propagate(minutesSinceEpoch float64) (physics.State, error) {
	fmt.Printf("[TRC] Entering OEMPropagator.Propagate for %.3f min\n", minutesSinceEpoch)
	defer fmt.Printf("[TRC] Exiting OEMPropagator.Propagate\n")

	target := p.oem.StartTime.Add(time.Duration(minutesSinceEpoch * 60.0 * float64(time.Second)))
	if target.Before(p.oem.StartTime) || target.After(p.oem.StopTime) {
		return physics.State{}, fmt.Errorf("requested time %v is outside range [%v, %v]", target, p.oem.StartTime, p.oem.StopTime)
	}

	degree := 7
	if len(p.oem.Data) < 8 {
		degree = len(p.oem.Data) - 1
	}

	idx := p.findNearestIndex(target)
	start := idx - degree/2
	if start < 0 {
		start = 0
	}
	if start+degree >= len(p.oem.Data) {
		start = len(p.oem.Data) - degree - 1
	}

	window := make([]DataPoint, degree+1)
	for i := 0; i <= degree; i++ {
		dp := p.oem.Data[start+i]
		window[i] = DataPoint{
			Time:  dp.Epoch.Sub(p.oem.StartTime).Seconds(),
			State: dp.State,
		}
	}

	targetSecs := target.Sub(p.oem.StartTime).Seconds()
	return LagrangeInterpolate(window, targetSecs), nil
}

func (p *OEMPropagator) findNearestIndex(target time.Time) int {
	low, high := 0, len(p.oem.Data)-1
	for low <= high {
		mid := (low + high) / 2
		if p.oem.Data[mid].Epoch.Before(target) {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return low
}
