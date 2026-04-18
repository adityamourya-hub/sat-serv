package hypersonic

import (
	"math"
	"time"
	"ssis/pkg/physics"
	"ssis/pkg/orbital"
)

/**
 * Ballistic Impact Engine (BMW Chapter 6 Standards)
 * Hardened for IAF ADITI 4.0 Mission-Critical Threat Assessment.
 */

type BallisticThreat struct {
	ImpactLat      float64
	ImpactLon      float64
	ApogeeKm       float64
	TimeOfFlight   float64
	ImpactVelocity float64
}

/**
 * ComputeBallisticImpact predicts the surface impact coordinate for a sub-orbital track.
 * Input:
 *   launchLat, launchLon: Burnout coordinates (Deg)
 *   vBurnoutKmS: Scaling velocity at burnout (km/s)
 *   gammaDeg: Flight Path Angle (Deg, >0 for ascending)
 *   azimuthDeg: Heading at burnout (Deg, 0=North, 90=East)
 */
func ComputeBallisticImpact(launchLat, launchLon, vBurnoutKmS, gammaDeg, azimuthDeg float64) BallisticThreat {
	// ... BMW Closed form for midcourse (vacuum arc) ...
	latB := launchLat * (math.Pi / 180.0)
	lonB := launchLon * (math.Pi / 180.0)
	gamma := gammaDeg * (math.Pi / 180.0)
	beta := azimuthDeg * (math.Pi / 180.0)
	
	rb := physics.WGS84_A // Burnout near surface
	v2 := vBurnoutKmS * vBurnoutKmS
	Q := (v2 * rb) / physics.EARTH_MU
	a := rb / (2.0 - Q)
	ecc := math.Sqrt(1.0 - (rb/a)*(2.0-rb/a)*math.Pow(math.Cos(gamma), 2))
	
	// Exact Range Angle (Psi) to surface impact (Analytical Estimate)
	tanPsiHalf := (Q * math.Sin(gamma) * math.Cos(gamma)) / (1.0 - Q*math.Pow(math.Cos(gamma), 2))
	psi_total := 2.0 * math.Atan(tanPsiHalf)
	
	// Terminal Handover Baseline: 120km (Entry Interface)
	r_ei := physics.WGS84_A + 120.0
	var tof, finalLat, finalLon, finalV float64
	
	if a*(1+ecc) < r_ei {
		// Lo-alt trajectory, purely numerical
		tof, finalLat, finalLon, finalV = propagateNumerical(latB, lonB, vBurnoutKmS, gamma, beta, 2451545.0)
	} else {
		// Handover point: Find psi where r = r_ei
		cosNuB := (a*(1-ecc*ecc)/rb - 1) / ecc
		if cosNuB > 1 { cosNuB = 1 }
		if cosNuB < -1 { cosNuB = -1 }
		nuB := math.Acos(cosNuB)
		if gamma < 0 { nuB = 2*math.Pi - nuB }
		
		cosNuEI := (a*(1-ecc*ecc)/r_ei - 1) / ecc
		if cosNuEI > 1 { cosNuEI = 1 }
		if cosNuEI < -1 { cosNuEI = -1 }
		nuEI := 2*math.Pi - math.Acos(cosNuEI) // Descending node for impact
		
		psi_ei := nuEI - nuB
		tof_midcourse := physics.CalculateSubOrbitalToF(a, ecc, rb, gamma, psi_ei)
		
		// Project to Entry Interface
		latEI, lonEI := physics.SphericalProject(latB, lonB, psi_ei, beta)
		
		// Terminal Numerical Handover (120km -> 0km)
		tof_terminal, lat_i, lon_i, v_i := propagateNumerical(latEI, lonEI-(physics.EARTH_ROTATION_RATE*tof_midcourse), vBurnoutKmS, -math.Acos(physics.WGS84_A*(1+ecc*math.Cos(nuEI))/(a*(1-ecc*ecc))), beta, 2451545.0+tof_midcourse/86400.0)
		
		tof = tof_midcourse + tof_terminal
		finalLat, finalLon, finalV = lat_i, lon_i, v_i
	}

	return BallisticThreat{
		ImpactLat:      finalLat * (180.0 / math.Pi),
		ImpactLon:      finalLon * (180.0 / math.Pi),
		ApogeeKm:       a*(1.0+ecc) - physics.WGS84_A,
		TimeOfFlight:   tof,
		ImpactVelocity: finalV,
	}
}

func propagateNumerical(lat, lon, v, gamma, beta, jd float64) (tof, finalLat, finalLon, finalV float64) {
	// 1. Initial ECI State (Approximate for terminal phase handover)
	x, y, z := physics.GeodeticToECEFWGS84(lat, lon, 0) // Simplified alt for handover start
	if lat > 1.0 { // Polar adjustment
		x, y, z = physics.GeodeticToECEFWGS84(lat, lon, 120.0)
	}
	
	// Transform beta, gamma to ECI velocity
	// ... (Simplification: integrate in geodetic alt-alpha-beta for sub-orbital) ...
	// To maintain SSIS standards, we use RKF45 with OrbitalAcceleration
	
	rk := orbital.NewIntegratorRKF45()
	rk.AbsError = 1e-8
	rk.RelError = 1e-8
	
	y_state := []float64{x, y, z, v * math.Cos(gamma) * math.Sin(beta), v * math.Cos(gamma) * math.Cos(beta), v * math.Sin(gamma)}
	t := 0.0
	dt := 1.0
	
	f := func(t float64, x_s []float64) []float64 {
		ax, ay, az := orbital.OrbitalAcceleration(x_s[0], x_s[1], x_s[2], x_s[3], x_s[4], x_s[5], jd+t/86400.0, orbital.PhysicsParams{Mass: 1000.0, Cd: 2.2, DragArea: 1.0})
		return []float64{x_s[3], x_s[4], x_s[5], ax, ay, az}
	}

	for i := 0; i < 5000; i++ {
		_, _, alt := orbital.ECIToGeodetic(y_state[0], y_state[1], y_state[2], jd+t/86400.0)
		if alt <= 0 {
			break
		}
		t, y_state, dt = rk.Step(t, dt, y_state, f)
	}
	
	phi, lambda, _ := orbital.ECIToGeodetic(y_state[0], y_state[1], y_state[2], jd+t/86400.0)
	return t, phi * (math.Pi / 180.0), lambda * (math.Pi / 180.0), math.Sqrt(y_state[3]*y_state[3] + y_state[4]*y_state[4] + y_state[5]*y_state[5])
}
