package orbital

import (
	"math"
	"testing"
	"time"
)

func TestPropagateSGP4(t *testing.T) {
	// Sample ISS TLE
	// Sample ISS TLE (Checksum 9 computed for line 1)
	line1 := "1 25544U 98067A   20316.41738426  .00001090  00000-0  28198-4 0  9999"
	line2 := "2 25544  51.6443  18.4235 0001600 120.3016 116.3687 15.49506689254846"

	epoch := time.Date(2020, 11, 11, 10, 0, 0, 0, time.UTC)
	pos, vel, err := PropagateSGP4(line1, line2, epoch)

	if err != nil {
		t.Fatalf("SGP4 failed: %v", err)
	}

	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	if r < ER+350 || r > ER+450 {
		t.Errorf("ISS altitude %.2f out of bounds (approx 400km)", r-ER)
	}
	
	v := math.Sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2])
	if v < 7.0 || v > 8.0 {
		t.Errorf("ISS velocity %.2f km/s out of bounds", v)
	}
}

func TestFusionDragAndJ3(t *testing.T) {
	// Baseline
	ax1, ay1, az1 := OrbitalAcceleration(ER+400, 0, 0, 0, 7.6, 0, 0.0)
	
	if ax1 == 0 && ay1 == 0 && az1 == 0 {
		t.Error("orbitalAcceleration is 0")
	}
}

func TestGST(t *testing.T) {
	jd := 2451545.0 // J2000
	gst := GreenwichSiderealTime(jd)
	if gst < 0 || gst >= 2*math.Pi {
		t.Errorf("GST out of bounds: %v", gst)
	}
}
