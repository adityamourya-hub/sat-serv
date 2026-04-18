package ingestion

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"ssis/pkg/physics"
)

/**
 * OEM (Orbit Ephemeris Message) Parser (CCSDS 502.0-B-2)
 * Hardened for IAF ADITI 4.0 Industry Interoperability.
 */

type OEMDataPoint struct {
	Epoch      time.Time
	X, Y, Z    float64 // km
	VX, VY, VZ float64 // km/s
}

type OEMFile struct {
	ObjectID   string
	ObjectName string
	Frame      string
	Points     []OEMDataPoint
}

/**
 * ParseOEMKVD parses an OEM file in Key-Value Document (KVD) format.
 */
func ParseOEMKVD(path string) (*OEMFile, error) {
	fmt.Printf("[TRC] Entering ParseOEMKVD for %s\n", path)
	defer fmt.Printf("[TRC] Exiting ParseOEMKVD\n")

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	oem := &OEMFile{Points: make([]OEMDataPoint, 0)}
	scanner := bufio.NewScanner(file)
	
	inData := false
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "COMMENT") {
			continue
		}

		if !inData {
			if strings.HasPrefix(line, "OBJECT_ID") {
				oem.ObjectID = strings.TrimSpace(strings.Split(line, "=")[1])
			} else if strings.HasPrefix(line, "OBJECT_NAME") {
				oem.ObjectName = strings.TrimSpace(strings.Split(line, "=")[1])
			} else if strings.HasPrefix(line, "REF_FRAME") {
				oem.Frame = strings.TrimSpace(strings.Split(line, "=")[1])
			} else if line == "DATA_START" || strings.Contains(line, "EPOCH") {
				inData = true
				if line == "DATA_START" { continue }
			}
		}

		if inData {
			parts := strings.Fields(line)
			if len(parts) >= 7 {
				epoch, err := time.Parse(time.RFC3339, parts[0])
				if err != nil {
					epoch, err = time.Parse("2006-002T15:04:05.999", parts[0])
					if err != nil { continue }
				}
				
				x, _ := strconv.ParseFloat(parts[1], 64)
				y, _ := strconv.ParseFloat(parts[2], 64)
				z, _ := strconv.ParseFloat(parts[3], 64)
				vx, _ := strconv.ParseFloat(parts[4], 64)
				vy, _ := strconv.ParseFloat(parts[5], 64)
				vz, _ := strconv.ParseFloat(parts[6], 64)
				
				oem.Points = append(oem.Points, OEMDataPoint{
					Epoch: epoch, X: x, Y: y, Z: z, VX: vx, VY: vy, VZ: vz,
				})
			}
		}
	}

	return oem, nil
}

/**
 * InterpolateState estimates the state at a target time using Hermite Splines.
 * This preserves both position and velocity continuity between points.
 */
func (o *OEMFile) InterpolateState(target time.Time) (physics.State, error) {
	fmt.Printf("[TRC] Entering OEMFile.InterpolateState for target %v\n", target)
	defer fmt.Printf("[TRC] Exiting OEMFile.InterpolateState\n")

	if len(o.Points) < 2 {
		return physics.State{}, fmt.Errorf("OEMFile: insufficient points for interpolation")
	}

	var p1, p2 OEMDataPoint
	found := false
	for i := 0; i < len(o.Points)-1; i++ {
		if (target.Equal(o.Points[i].Epoch) || target.After(o.Points[i].Epoch)) && 
		   (target.Equal(o.Points[i+1].Epoch) || target.Before(o.Points[i+1].Epoch)) {
			p1, p2 = o.Points[i], o.Points[i+1]
			found = true
			break
		}
	}

	if !found {
		return physics.State{}, fmt.Errorf("OEMFile: target time %v out of ephemeris bounds", target)
	}

	t1 := p1.Epoch.UnixNano()
	t2 := p2.Epoch.UnixNano()
	t := target.UnixNano()
	
	dur := float64(t2-t1) / 1e9
	if dur <= 0 {
		return physics.State{
			Pos: physics.Vector3{p1.X, p1.Y, p1.Z},
			Vel: physics.Vector3{p1.VX, p1.VY, p1.VZ},
		}, nil
	}
	s := float64(t-t1) / float64(t2-t1)
	s2 := s * s
	s3 := s2 * s

	h00 := 2*s3 - 3*s2 + 1
	h10 := s3 - 2*s2 + s
	h01 := -2*s3 + 3*s2
	h11 := s3 - s2

	state := physics.State{
		Pos: physics.Vector3{
			h00*p1.X + h10*dur*p1.VX + h01*p2.X + h11*dur*p2.VX,
			h00*p1.Y + h10*dur*p1.VY + h01*p2.Y + h11*dur*p2.VY,
			h00*p1.Z + h10*dur*p1.VZ + h01*p2.Z + h11*dur*p2.VZ,
		},
		Vel: physics.Vector3{
			(6*s2-6*s)*p1.X/dur + (3*s2-4*s+1)*p1.VX + (-6*s2+6*s)*p2.X/dur + (3*s2-2*s)*p2.VX,
			(6*s2-6*s)*p1.Y/dur + (3*s2-4*s+1)*p1.VY + (-6*s2+6*s)*p2.Y/dur + (3*s2-2*s)*p2.VY,
			(6*s2-6*s)*p1.Z/dur + (3*s2-4*s+1)*p1.VZ + (-6*s2+6*s)*p2.Z/dur + (3*s2-2*s)*p2.VZ,
		},
	}

	return state, nil
}
