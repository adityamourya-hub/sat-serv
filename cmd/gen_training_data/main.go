package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

type TrajectorySample struct {
	SatID        string      `json:"sat_id"`
	InitialState []float64   `json:"initial_state"`
	TimePoints   []float64   `json:"time_points"`
	States       [][]float64 `json:"states"`
}

func main() {
	fmt.Printf("[TRC] Entering Training Data Generator\n")
	defer fmt.Printf("[TRC] Exiting Training Data Generator\n")

	rand.Seed(time.Now().UnixNano())
	
	numSats := 100
	samplesPerSat := 60
	dt := 60.0 // seconds

	var allSamples []TrajectorySample

	fmt.Printf("Generating training data for %d satellites...\n", numSats)

	for i := 0; i < numSats; i++ {
		regime := rand.Intn(3)
		var rMin, rMax float64
		switch regime {
		case 0: // LEO
			rMin, rMax = physics.EARTH_R_EQUATORIAL+300, physics.EARTH_R_EQUATORIAL+2000
		case 1: // MEO
			rMin, rMax = physics.EARTH_R_EQUATORIAL+2000, physics.EARTH_R_EQUATORIAL+30000
		case 2: // GEO
			rMin, rMax = 42164-500, 42164+500
		}

		rMag := rMin + rand.Float64()*(rMax-rMin)
		phi := rand.Float64() * 2 * math.Pi
		theta := math.Acos(2*rand.Float64() - 1)
		
		x := rMag * math.Sin(theta) * math.Cos(phi)
		y := rMag * math.Sin(theta) * math.Sin(phi)
		z := rMag * math.Cos(theta)
		
		vMag := math.Sqrt(physics.EARTH_MU_IAU / rMag)
		vMag *= (0.9 + rand.Float64()*0.2)

		vx := -vMag * math.Sin(phi)
		vy := vMag * math.Cos(phi)
		vz := 0.0

		init := physics.State{
			Pos: physics.Vector3{x, y, z},
			Vel: physics.Vector3{vx, vy, vz},
		}
		epoch := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
		params := physics.PhysicsParams{
			Mass:     1000.0,
			DragArea: 10.0,
			Cd:       2.2,
			SRPArea:  10.0,
			Cr:       1.2,
		}

		prop := orbital.NewAxiomPropagator(nil, init, epoch, params)
		
		sample := TrajectorySample{
			SatID:        fmt.Sprintf("SAT_%04d", i),
			InitialState: []float64{x, y, z, vx, vy, vz},
			TimePoints:   make([]float64, samplesPerSat),
			States:       make([][]float64, samplesPerSat),
		}

		for j := 0; j < samplesPerSat; j++ {
			tMin := float64(j) * (dt / 60.0)
			state, _ := prop.Propagate(tMin)
			
			sample.TimePoints[j] = tMin * 60.0
			sample.States[j] = []float64{state.Pos[0], state.Pos[1], state.Pos[2], state.Vel[0], state.Vel[1], state.Vel[2]}
		}

		allSamples = append(allSamples, sample)
	}

	outputFile := "training_data_high_fidelity.json"
	fileData, _ := json.MarshalIndent(allSamples, "", "  ")
	_ = os.WriteFile(outputFile, fileData, 0644)

	fmt.Printf("Successfully generated %d samples to %s\n", len(allSamples), outputFile)
}
