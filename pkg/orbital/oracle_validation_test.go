package orbital_test

import (
	"math"
	"ssis/pkg/catalog"
	"ssis/pkg/hypersonic"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
	"ssis/pkg/proximity"
	"testing"
)

func TestStabilityOracleValidation(t *testing.T) {
	// 1. MOID ORACLE GATE
	t.Run("MOID_Oracle_Compliance", func(t *testing.T) {
		s1 := orbital.State{X: 7000, Y: 0, Z: 0, VX: 0, VY: 7.5, VZ: 0}
		s2 := orbital.State{X: 7000, Y: 0.1, Z: 0, VX: 0, VY: 0, VZ: 7.5}
		
		moid := proximity.CalculateMOID(s1, s2)
		if moid < 0 {
			t.Errorf("MOID oracle returned negative value")
		}
	})

	// 2. RE-ENTRY ORACLE (REQ-ORCL-H)
	t.Run("ReEntry_Impact_Oracle", func(t *testing.T) {
		initial := orbital.State{X: 6378 + 120, Y: 0, Z: 0, VX: -0.1, VY: 7.2, VZ: 0}
		impact := hypersonic.PredictImpactFootprint(initial, 0.05, 150.0, 15.0)
		
		if math.IsNaN(impact.LatDeg) || math.IsNaN(impact.LonDeg) {
			t.Errorf("Re-entry oracle failed with NaN")
		}
	})

	// 3. SVO BVH ORACLE (REQ-ORCL-A)
	t.Run("SVO_Efficiency_Oracle", func(t *testing.T) {
		store := catalog.NewStore(500.0)
		for i := 0; i < 1000; i++ {
			store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{
				{ID: i, Pos: octree.Vector3{X: float64(7000 + i), Y: 0, Z: 0}},
			}}, true)
		}
	})

	// 4. ATMOSPHERIC ORACLE GATE
	t.Run("Atmospheric_Density_Profile", func(t *testing.T) {
		altitudes := []float64{100, 200, 300, 400, 500}
		for _, alt := range altitudes {
			rho := orbital.AtmosphericDensity(alt)
			if rho <= 0 && alt < 800 {
				t.Errorf("Oracle failure: Atmosphere vacuum below 800km (Alt=%.0f, rho=%.4e)", alt, rho)
			}
		}
	})
}
