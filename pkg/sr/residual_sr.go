package sr

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"unicode"

	// "github.com/Knetic/govaluate" // Add to go.mod: go get github.com/Knetic/govaluate
	"ssis/pkg/physics"
)

/**
 * Symbolic Regression (SR) Physics Engine Self-Improvement
 *
 * Discovers explicit, human-readable equations from residuals:
 * Δa = a_observed - a_classical
 *
 * These equations are hardcoded into OrbitalAcceleration as perturbation terms,
 * providing traceable, certifiable physics vs black-box HNN.
 *
 * Workflow:
 * 1. Run classical core (J2, drag, third-body, SRP)
 * 2. Compute residuals from UKF or high-fidelity data
 * 3. PySR discovers parsimonious equations (Pareto front)
 * 4. Hardcode best equation into Go via govaluate
 * 5. Re-propagate → new residuals → periodic re-discovery
 *
 * References:
 * - PySR: https://github.com/MilesCranmer/PySR
 * - arXiv:2305.01582 (PySR paper)
 * - govaluate: https://github.com/Knetic/govaluate
 */

/**
 * DiscoveredEquation represents an SR-discovered perturbation law.
 *
 * Example:
 * {
 *   "Name": "solar_drag_correction",
 *   "Expression": "sin(u) * k1 * F107 + k2 * exp(-h/H)",
 *   "Parameters": {"k1": 0.003, "k2": 0.001, "H": 150.0},
 *   "Variables": ["u", "F107", "h"],
 *   "Domain": {"minAlt": 200, "maxAlt": 800},
 *   "ErrorReduction": 0.85,  // 85% error reduction
 *   "Complexity": 8,         // Node count
 *   "Approved": true,        // IAF certification
 *   "Version": "2024.04.18.1"
 * }
 */
type DiscoveredEquation struct {
	Name            string             `json:"name"`
	Expression      string             `json:"expression"`
	Parameters      map[string]float64 `json:"parameters"`
	Variables       []string           `json:"variables"`
	Domain          DomainConstraints  `json:"domain"`
	ErrorReduction  float64            `json:"error_reduction"` // vs classical
	Complexity      int                `json:"complexity"`      // Node count
	Approved        bool               `json:"approved"`        // IAF certified
	Version         string             `json:"version"`
	DiscoveryDate   string             `json:"discovery_date"`
	DiscoverySource string             `json:"discovery_source"` // "PySR", "manual"
}

/**
 * DomainConstraints limits equation applicability.
 */
type DomainConstraints struct {
	MinAlt    float64 `json:"min_alt_km"`
	MaxAlt    float64 `json:"max_alt_km"`
	MinF107   float64 `json:"min_f107"`
	MaxF107   float64 `json:"max_f107"`
	MinIncl   float64 `json:"min_incl_deg"`
	MaxIncl   float64 `json:"max_incl_deg"`
	OrbitType string  `json:"orbit_type"` // "LEO", "GEO", "any"
}

/**
 * ResidualSample represents one training example for SR.
 *
 * Features: orbital state + environment
 * Target: residual acceleration (observed - classical)
 */
type ResidualSample struct {
	// State features
	Alt        float64 // Altitude (km)
	Vel        float64 // Velocity magnitude (km/s)
	U          float64 // Argument of latitude (rad)
	Incl       float64 // Inclination (rad)
	RAAN       float64 // RAAN (rad)
	ArgPerigee float64 // Argument of perigee (rad)

	// Environment features
	F107      float64 // Solar flux
	F107Avg   float64 // 81-day average
	Ap        float64 // Geomagnetic index
	Kp        float64 // Planetary K-index
	LocalTime float64 // Hours (0-24)
	Latitude  float64 // Sub-satellite latitude (rad)
	Density   float64 // NRLMSISE-00 density (kg/m³)

	// Target: residual acceleration (km/s²)
	ResA      [3]float64 // Residual acceleration vector
	ResScalar float64    // Scalar residual magnitude

	// Metadata
	Epoch    float64 // Julian Date
	SatID    int     // NORAD ID
	Maneuver bool    // Known maneuver in this window
}

/**
 * SRDatabase manages discovered equations.
 *
 * Air-gapped, versioned, IAF-certifiable equation storage.
 */
type SRDatabase struct {
	mu        sync.RWMutex
	Equations map[string]*DiscoveredEquation
	Path      string
}

/**
 * NewSRDatabase loads equation database from JSON.
 */
func NewSRDatabase(path string) (*SRDatabase, error) {
	db := &SRDatabase{
		Equations: make(map[string]*DiscoveredEquation),
		Path:      path,
	}

	// Load existing equations
	if _, err := os.Stat(path); err == nil {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read SR database: %w", err)
		}

		var equations []*DiscoveredEquation
		if err := json.Unmarshal(data, &equations); err != nil {
			return nil, fmt.Errorf("failed to parse SR database: %w", err)
		}

		for _, eq := range equations {
			if eq.Approved {
				db.Equations[eq.Name] = eq
			}
		}
	}

	return db, nil
}

/**
 * Save persists approved equations to disk.
 */
func (db *SRDatabase) Save() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var equations []*DiscoveredEquation
	for _, eq := range db.Equations {
		equations = append(equations, eq)
	}

	data, err := json.MarshalIndent(equations, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(db.Path, data, 0644)
}

/**
 * AddEquation adds a new discovered equation (pending approval).
 */
func (db *SRDatabase) AddEquation(eq *DiscoveredEquation) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.Equations[eq.Name] = eq
}

/**
 * GetEquation retrieves an approved equation by name.
 */
func (db *SRDatabase) GetEquation(name string) (*DiscoveredEquation, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	eq, exists := db.Equations[name]
	return eq, exists
}

/**
 * ListEquations returns all approved equations.
 */
func (db *SRDatabase) ListEquations() []*DiscoveredEquation {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var list []*DiscoveredEquation
	for _, eq := range db.Equations {
		list = append(list, eq)
	}
	return list
}

// =============================================================================
// Runtime Equation Evaluator
// =============================================================================

/**
 * EquationEvaluator executes discovered equations at runtime.
 *
 * Uses a simple arithmetic parser for basic expressions.
 * Supports: +, -, *, /, ^, parentheses, and variables.
 */
type EquationEvaluator struct {
	Equation  *DiscoveredEquation
	variables []string // Extracted variable names
}

/**
 * NewEquationEvaluator compiles an equation for execution.
 */
func NewEquationEvaluator(eq *DiscoveredEquation) (*EquationEvaluator, error) {
	if eq.Expression == "" {
		return nil, fmt.Errorf("empty expression")
	}

	// Extract variables from expression
	vars := extractVariables(eq.Expression)

	return &EquationEvaluator{
		Equation:  eq,
		variables: vars,
	}, nil
}

// extractVariables pulls out variable names from expression
func extractVariables(expr string) []string {
	// Simple extraction: find alphabetic sequences that aren't functions
	vars := make(map[string]bool)
	var current strings.Builder

	for i, ch := range expr {
		if unicode.IsLetter(ch) || (current.Len() > 0 && (ch == '_' || unicode.IsDigit(ch))) {
			current.WriteRune(ch)
		} else {
			if current.Len() > 0 {
				word := current.String()
				// Skip known functions
				if !isFunction(word) {
					vars[word] = true
				}
				current.Reset()
			}
		}
		_ = i
	}

	// Don't forget last word
	if current.Len() > 0 {
		word := current.String()
		if !isFunction(word) {
			vars[word] = true
		}
	}

	result := make([]string, 0, len(vars))
	for v := range vars {
		result = append(result, v)
	}
	return result
}

// isFunction checks if word is a known math function
func isFunction(word string) bool {
	functions := []string{"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pow"}
	for _, f := range functions {
		if word == f {
			return true
		}
	}
	return false
}

/**
 * Evaluate computes perturbation from equation.
 *
 * @param params: Variable values (u, F107, h, etc.)
 * @return: Scalar perturbation value
 */
func (eval *EquationEvaluator) Evaluate(params map[string]interface{}) (float64, error) {
	return eval.evaluateExpression(eval.Equation.Expression, params)
}

// evaluateExpression evaluates a simple arithmetic expression
func (eval *EquationEvaluator) evaluateExpression(expr string, params map[string]interface{}) (float64, error) {
	// Substitute variables with their values
	substituted := expr
	for name, val := range params {
		var strVal string
		switch v := val.(type) {
		case float64:
			strVal = fmt.Sprintf("%g", v)
		case float32:
			strVal = fmt.Sprintf("%g", float64(v))
		case int:
			strVal = fmt.Sprintf("%d", v)
		default:
			strVal = fmt.Sprintf("%v", v)
		}
		substituted = strings.ReplaceAll(substituted, name, strVal)
	}

	// Simple recursive descent parser
	result, err := parseExpression(substituted)
	if err != nil {
		return 0, fmt.Errorf("failed to evaluate '%s': %w", expr, err)
	}

	return result, nil
}

// parseExpression parses and evaluates a mathematical expression
func parseExpression(expr string) (float64, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return 0, fmt.Errorf("empty expression")
	}

	// Handle parentheses first
	for strings.Contains(expr, "(") {
		start := strings.LastIndex(expr, "(")
		end := strings.Index(expr[start:], ")")
		if end == -1 {
			return 0, fmt.Errorf("unmatched parenthesis")
		}
		end += start

		inner, err := parseExpression(expr[start+1 : end])
		if err != nil {
			return 0, err
		}
		expr = expr[:start] + fmt.Sprintf("%g", inner) + expr[end+1:]
	}

	// Parse + and - (lowest precedence)
	if idx := findOperator(expr, '+'); idx != -1 {
		left, err := parseExpression(expr[:idx])
		if err != nil {
			return 0, err
		}
		right, err := parseExpression(expr[idx+1:])
		if err != nil {
			return 0, err
		}
		return left + right, nil
	}

	if idx := findOperator(expr, '-'); idx != -1 {
		left, err := parseExpression(expr[:idx])
		if err != nil {
			return 0, err
		}
		right, err := parseExpression(expr[idx+1:])
		if err != nil {
			return 0, err
		}
		return left - right, nil
	}

	// Parse * and /
	if idx := findOperator(expr, '*'); idx != -1 {
		left, err := parseExpression(expr[:idx])
		if err != nil {
			return 0, err
		}
		right, err := parseExpression(expr[idx+1:])
		if err != nil {
			return 0, err
		}
		return left * right, nil
	}

	if idx := findOperator(expr, '/'); idx != -1 {
		left, err := parseExpression(expr[:idx])
		if err != nil {
			return 0, err
		}
		right, err := parseExpression(expr[idx+1:])
		if err != nil {
			return 0, err
		}
		if right == 0 {
			return 0, fmt.Errorf("division by zero")
		}
		return left / right, nil
	}

	// Parse ^ (exponentiation)
	if idx := findOperator(expr, '^'); idx != -1 {
		left, err := parseExpression(expr[:idx])
		if err != nil {
			return 0, err
		}
		right, err := parseExpression(expr[idx+1:])
		if err != nil {
			return 0, err
		}
		return math.Pow(left, right), nil
	}

	// Must be a number
	val, err := strconv.ParseFloat(expr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number: %s", expr)
	}
	return val, nil
}

// findOperator finds the operator at the current precedence level
func findOperator(expr string, op rune) int {
	depth := 0
	for i, ch := range expr {
		switch ch {
		case '(':
			depth++
		case ')':
			depth--
		}
		if depth == 0 && ch == op {
			return i
		}
	}
	return -1
}

/**
 * CheckDomain verifies if state is within equation applicability.
 */
func (eval *EquationEvaluator) CheckDomain(sample ResidualSample) bool {
	domain := eval.Equation.Domain

	if sample.Alt < domain.MinAlt || sample.Alt > domain.MaxAlt {
		return false
	}
	if sample.F107 < domain.MinF107 || sample.F107 > domain.MaxF107 {
		return false
	}

	inclDeg := sample.Incl * 180.0 / math.Pi
	if inclDeg < domain.MinIncl || inclDeg > domain.MaxIncl {
		return false
	}

	return true
}

// =============================================================================
// SR-Enhanced Physics Engine Integration
// =============================================================================

/**
 * SRPerturbationEngine applies discovered equations as perturbations.
 *
 * This integrates with OrbitalAcceleration to add SR-discovered terms.
 */
type SRPerturbationEngine struct {
	Database   *SRDatabase
	Evaluators map[string]*EquationEvaluator
	mu         sync.RWMutex
}

/**
 * NewSRPerturbationEngine creates SR-enhanced perturbation system.
 */
func NewSRPerturbationEngine(db *SRDatabase) (*SRPerturbationEngine, error) {
	engine := &SRPerturbationEngine{
		Database:   db,
		Evaluators: make(map[string]*EquationEvaluator),
	}

	// Pre-compile all approved equations
	for name, eq := range db.Equations {
		eval, err := NewEquationEvaluator(eq)
		if err != nil {
			fmt.Printf("[WARN] Failed to compile equation '%s': %v\n", name, err)
			continue
		}
		engine.Evaluators[name] = eval
	}

	return engine, nil
}

/**
 * ComputePerturbation calculates SR-discovered acceleration correction.
 *
 * Called from OrbitalAcceleration with current state + environment.
 *
 * @param x, y, z: Position ECI (km)
 * @param vx, vy, vz: Velocity ECI (km/s)
 * @param jd: Julian Date
 * @param env: Space weather data
 * @return: Additional acceleration (ax, ay, az) km/s²
 */
func (engine *SRPerturbationEngine) ComputePerturbation(
	x, y, z float64,
	vx, vy, vz float64,
	jd float64,
	env physics.SpaceWeatherData,
) (ax, ay, az float64) {

	// Build feature vector
	sample := engine.buildSample(x, y, z, vx, vy, vz, jd, env)

	// Evaluate applicable equations
	engine.mu.RLock()
	defer engine.mu.RUnlock()

	for _, eval := range engine.Evaluators {
		// Check domain applicability
		if !eval.CheckDomain(sample) {
			continue
		}

		// Build parameter map
		params := engine.buildParams(sample)

		// Evaluate equation
		delta, err := eval.Evaluate(params)
		if err != nil {
			continue
		}

		// Apply perturbation (equation-specific direction)
		// For scalar corrections, apply along velocity or as drag
		switch eval.Equation.Name {
		case "drag_correction_scalar":
			// Apply as drag along velocity vector
			vMag := math.Sqrt(vx*vx + vy*vy + vz*vz)
			if vMag > 1e-9 {
				dragFactor := delta / vMag
				ax -= dragFactor * vx
				ay -= dragFactor * vy
				az -= dragFactor * vz
			}

		case "maneuver_thrust_profile":
			// Apply along specific direction (e.g., tangential)
			ax += delta * vx / sample.Vel
			ay += delta * vy / sample.Vel
			az += delta * vz / sample.Vel

		case "density_tide_correction":
			// Apply as vertical perturbation
			r := math.Sqrt(x*x + y*y + z*z)
			ax += delta * x / r
			ay += delta * y / r
			az += delta * z / r

		default:
			// Generic: add to all components (conservative)
			ax += delta * 0.1
			ay += delta * 0.1
			az += delta * 0.1
		}

		// Debug logging
		// fmt.Printf("[SR] Applied '%s': Δa=%.3e km/s²\n", name, delta)
	}

	return ax, ay, az
}

/**
 * buildSample creates ResidualSample from current state.
 */
func (engine *SRPerturbationEngine) buildSample(
	x, y, z float64,
	vx, vy, vz float64,
	jd float64,
	env physics.SpaceWeatherData,
) ResidualSample {

	r := math.Sqrt(x*x + y*y + z*z)
	alt := r - physics.EARTH_R_EQUATORIAL
	vel := math.Sqrt(vx*vx + vy*vy + vz*vz)

	// Compute argument of latitude (simplified)
	// Real: would use proper orbital elements
	u := math.Atan2(y, x) // Simplified

	// Local time from longitude + JD
	longitude := math.Atan2(y, x)
	localTime := (jd-2451545.0)*24.0 + longitude*12.0/math.Pi
	localTime = math.Mod(localTime, 24.0)
	if localTime < 0 {
		localTime += 24.0
	}

	// Latitude
	latitude := math.Asin(z / r)

	return ResidualSample{
		Alt:       alt,
		Vel:       vel,
		U:         u,
		F107:      env.F107,
		F107Avg:   env.F107Avg,
		Ap:        env.Ap,
		Kp:        env.Kp,
		LocalTime: localTime,
		Latitude:  latitude,
		Density:   0, // Would compute from NRLMSISE
	}
}

/**
 * buildParams creates evaluation parameter map.
 */
func (engine *SRPerturbationEngine) buildParams(sample ResidualSample) map[string]interface{} {
	return map[string]interface{}{
		"h":     sample.Alt,
		"v":     sample.Vel,
		"u":     sample.U,
		"F107":  sample.F107,
		"F107a": sample.F107Avg,
		"Ap":    sample.Ap,
		"Kp":    sample.Kp,
		"LT":    sample.LocalTime,
		"lat":   sample.Latitude,
		"rho":   sample.Density,
		"sin":   math.Sin,
		"cos":   math.Cos,
		"exp":   math.Exp,
		"log":   math.Log,
		"sqrt":  math.Sqrt,
		"abs":   math.Abs,
		"pow":   math.Pow,
	}
}

/**
 * Reload updates evaluators after database changes.
 */
func (engine *SRPerturbationEngine) Reload() error {
	engine.mu.Lock()
	defer engine.mu.Unlock()

	engine.Evaluators = make(map[string]*EquationEvaluator)

	for _, eq := range engine.Database.Equations {
		if !eq.Approved {
			continue
		}

		eval, err := NewEquationEvaluator(eq)
		if err != nil {
			fmt.Printf("[WARN] Failed to reload equation: %v\n", err)
			continue
		}
		engine.Evaluators[eq.Name] = eval
	}

	return nil
}

// =============================================================================
// Residual Computation for Training
// =============================================================================

/**
 * ResidualComputer calculates Δa = observed - classical.
 *
 * Generates training data for PySR.
 */
type ResidualComputer struct {
	ClassicalAccel func(x, y, z, vx, vy, vz, jd float64, params physics.PhysicsParams) (ax, ay, az float64)
}

/**
 * ComputeResidual calculates acceleration residual.
 *
 * @param observed: Measured/orbit-determined acceleration
 * @param state: Current orbital state
 * @param jd: Julian Date
 * @param params: Physics parameters
 * @return: ResidualSample with features + target
 */
func (rc *ResidualComputer) ComputeResidual(
	observed [3]float64,
	state physics.State,
	jd float64,
	params physics.PhysicsParams,
	env physics.SpaceWeatherData,
) ResidualSample {

	// Classical prediction
	axClassical, ayClassical, azClassical := rc.ClassicalAccel(
		state.Pos[0], state.Pos[1], state.Pos[2],
		state.Vel[0], state.Vel[1], state.Vel[2],
		jd, params,
	)

	// Residual
	resA := [3]float64{
		observed[0] - axClassical,
		observed[1] - ayClassical,
		observed[2] - azClassical,
	}

	resScalar := math.Sqrt(resA[0]*resA[0] + resA[1]*resA[1] + resA[2]*resA[2])

	// Build sample
	sample := ResidualSample{
		ResA:      resA,
		ResScalar: resScalar,
		Epoch:     jd,
		F107:      env.F107,
		F107Avg:   env.F107Avg,
		Ap:        env.Ap,
		Kp:        env.Kp,
	}

	// Compute orbital elements from state (simplified)
	r := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	sample.Alt = r - physics.EARTH_R_EQUATORIAL
	sample.Vel = math.Sqrt(state.Vel[0]*state.Vel[0] + state.Vel[1]*state.Vel[1] + state.Vel[2]*state.Vel[2])

	return sample
}

/**
 * ExportTrainingData saves residuals for PySR training.
 */
func ExportTrainingData(samples []ResidualSample, path string) error {
	// Convert to format PySR expects
	type PySRRow struct {
		Features map[string]float64 `json:"features"`
		Target   float64            `json:"target"`
	}

	var rows []PySRRow
	for _, s := range samples {
		row := PySRRow{
			Features: map[string]float64{
				"h":     s.Alt,
				"v":     s.Vel,
				"u":     s.U,
				"F107":  s.F107,
				"F107a": s.F107Avg,
				"Ap":    s.Ap,
				"Kp":    s.Kp,
				"LT":    s.LocalTime,
				"lat":   s.Latitude,
				"rho":   s.Density,
			},
			Target: s.ResScalar,
		}
		rows = append(rows, row)
	}

	data, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// =============================================================================
// Example Discovered Equations (for documentation)
// =============================================================================

/**
 * GetExampleEquations returns sample discovered laws.
 *
 * These demonstrate the SR capability - replace with actual PySR discoveries.
 */
func GetExampleEquations() []*DiscoveredEquation {
	return []*DiscoveredEquation{
		{
			Name:       "solar_drag_tide",
			Expression: "sin(u) * k1 * F107 * exp(-h/H)",
			Parameters: map[string]float64{
				"k1": 0.003,
				"H":  150.0,
			},
			Variables:       []string{"u", "F107", "h"},
			ErrorReduction:  0.72,
			Complexity:      8,
			Approved:        true,
			Version:         "2024.04.18.1",
			DiscoveryDate:   "2024-04-18",
			DiscoverySource: "PySR",
			Domain: DomainConstraints{
				MinAlt: 200, MaxAlt: 800,
				MinF107: 100, MaxF107: 250,
				OrbitType: "LEO",
			},
		},
		{
			Name:       "station_keeping_rhythmic",
			Expression: "k1 * cos(u) + k2 * sin(2*u)",
			Parameters: map[string]float64{
				"k1": 0.0005,
				"k2": 0.0002,
			},
			Variables:       []string{"u"},
			ErrorReduction:  0.91,
			Complexity:      6,
			Approved:        true,
			Version:         "2024.04.18.2",
			DiscoveryDate:   "2024-04-18",
			DiscoverySource: "PySR",
			Domain: DomainConstraints{
				MinAlt: 35000, MaxAlt: 45000,
				OrbitType: "GEO",
			},
		},
		{
			Name:       "geomagnetic_storm_drag",
			Expression: "k1 * Ap * exp(-(h-300)/H) * pow(v, 2)",
			Parameters: map[string]float64{
				"k1": 1e-8,
				"H":  50.0,
			},
			Variables:       []string{"Ap", "h", "v"},
			ErrorReduction:  0.68,
			Complexity:      10,
			Approved:        false, // Pending IAF review
			Version:         "2024.04.18.3-draft",
			DiscoveryDate:   "2024-04-18",
			DiscoverySource: "PySR",
			Domain: DomainConstraints{
				MinAlt: 150, MaxAlt: 600,
				MinF107: 50, MaxF107: 300,
				OrbitType: "LEO",
			},
		},
	}
}
