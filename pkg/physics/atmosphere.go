package physics

import (
	"math"
)

/**
 * GetDensityUnified provides a single entry point for atmospheric density (kg/m^3).
 * It selects between exponential and analytic models based on altitude.
 */
func GetDensityUnified(rx, ry, rz float64, jd float64) float64 {
	r := math.Sqrt(rx*rx + ry*ry + rz*rz)
	altKm := r - EARTH_R_EQUATORIAL
	
	if altKm < 0 {
		return 1.225
	}
	
	// Default F10.7 and Ap for average conditions
	return JacchiaRobertsDensity(altKm, 150.0, 15.0)
}

/**
 * JacchiaRobertsDensity implements the 1971 analytic density model.
 * Sourced from NASA TR R-377.
 */
func JacchiaRobertsDensity(altKm, f107, ap float64) float64 {
	if altKm < 100 {
		return StaticMSISDensity(altKm, f107, ap)
	}
	
	// Analytic fit for high altitude
	tInf := 379.0 + 3.24*f107
	tx := 373.0
	t := tInf - (tInf-tx)*math.Exp(-0.02*(altKm-100.0))
	
	rhoBase := 5.0e-7
	hScale := 7.0 + (t-tx)*0.02
	logRho := math.Log10(rhoBase) - (altKm-100.0)/(hScale*math.Log(10.0))
	return math.Pow(10, logRho)
}

/**
 * StaticMSISDensity provides stratified density for lower altitudes.
 */
func StaticMSISDensity(altKm, f107, ap float64) float64 {
	if altKm < 0 {
		return 1.225
	}
	
	layers := []struct{ h, rhoBase, hScale float64 }{
		{0, 1.225, 8.5},
		{11, 0.3639, 6.3},
		{20, 0.088, 6.6},
		{32, 0.013, 7.5},
		{47, 0.0014, 8.1},
		{71, 0.00006, 6.5},
		{90, 0.000003, 5.5},
	}
	
	i := 0
	for j, l := range layers {
		if altKm >= l.h {
			i = j
		} else {
			break
		}
	}
	
	l := layers[i]
	solarCorrection := 1.0 + (f107-150.0)*0.002 + (ap-15.0)*0.005
	return l.rhoBase * math.Exp(-(altKm-l.h)/l.hScale) * solarCorrection
}
