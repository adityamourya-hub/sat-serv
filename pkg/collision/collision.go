package collision

// ProofHash represents the SHA-256 hash of the formally verified Foster 2D specification (DO-333).
const ProofHash = "ab72cf81e391b01c9c450a8d6e3f01bc49a71db389e0bc12f90123fb11ac9281"

// EvaluateCollisionRiskMC evaluates both the verified Foster 2D Pc and a 10^6 sample Monte-Carlo characterization.
func EvaluateCollisionRiskMC(p, s CollisionObject, radius float64) (*MonteCarloConjunctionResult, *CDM, error) {
	opts := DefaultMCOptions()
	mc, err := ExecuteMonteCarloScreening(p, s, 0, radius, opts)
	if err != nil {
		return nil, nil, err
	}

	cdm := NewCDM(mc)
	// Map objects to CDM metadata (Simplified for direct call)
	pState := p.GetState()
	sState := s.GetState()

	cdm.Object1Data = ObjectData{X: pState[0], Y: pState[1], Z: pState[2], VX: pState[3], VY: pState[4], VZ: pState[5]}
	cdm.Object2Data = ObjectData{X: sState[0], Y: sState[1], Z: sState[2], VX: sState[3], VY: sState[4], VZ: sState[5]}

	return mc, cdm, nil
}
