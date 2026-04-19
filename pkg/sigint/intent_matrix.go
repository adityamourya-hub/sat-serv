// Package sigint provides intent inference matrix combining EM and optical signatures
// Enables threat characterization through multi-modal sensor fusion
package sigint

import (
	"math"
)

const (
	// Intent confidence thresholds
	IntentConfidenceThreshold = 0.7
	EMWeight                  = 0.5
	OpticalWeight             = 0.5
)

// EMSignature represents electromagnetic signature characteristics
type EMSignature struct {
	SignalType   string // JAMMER, COMMS_RELAY, ELINT, UNKNOWN
	Modulation   string // QPSK, 16-QAM, FHSS, etc.
	PowerDBm     float64
	Bandwidth    float64 // Hz
	IsBurst      bool
	IsNarrowBeam bool
	IsBacklit    bool
	DopplerShift float64 // Hz
}

// OpticalSignature represents optical/attitude signature
type OpticalSignature struct {
	AttitudeMode   string // NADIR, STAR_POINTING, OFFSET, UNKNOWN
	IsStabilized   bool
	IsManeuvering  bool
	RelativeMotion bool
	StreakDetected bool
	ThermalSig     float64 // Kelvin
	RCS            float64 // m²
}

// IntentClassification represents inferred payload intent
type IntentClassification struct {
	Intent              string
	Confidence          float64
	EMContribution      float64
	OpticalContribution float64
	ThreatLevel         string
	Action              string
}

// IntentInferenceMatrix combines EM and optical signatures to infer intent
type IntentInferenceMatrix struct {
	rules map[string]IntentRule
}

// IntentRule defines a rule for intent inference
type IntentRule struct {
	EMCondition      func(EMSignature) bool
	OpticalCondition func(OpticalSignature) bool
	Intent           string
	ThreatLevel      string
	Action           string
}

// NewIntentInferenceMatrix creates a new intent inference matrix
func NewIntentInferenceMatrix() *IntentInferenceMatrix {
	matrix := &IntentInferenceMatrix{
		rules: make(map[string]IntentRule),
	}

	matrix.initializeRules()
	return matrix
}

// initializeRules sets up the intent inference rules
func (iim *IntentInferenceMatrix) initializeRules() {
	// Rule 1: Jamming (Wide-band noise + Nadir-pointing)
	iim.rules["jamming"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			return em.SignalType == SignalTypeJammer && em.IsNarrowBeam
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.AttitudeMode == "NADIR" && opt.IsStabilized
		},
		Intent:      "JAMMING",
		ThreatLevel: "CRITICAL",
		Action:      "RF_COUNTERMEASURE",
	}

	// Rule 2: Comms Relay (Periodic pulses + Star-pointing)
	iim.rules["comms_relay"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			return em.SignalType == SignalTypeComms && !em.IsBurst
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.AttitudeMode == "STAR_POINTING" && opt.IsStabilized
		},
		Intent:      "COMMS_RELAY",
		ThreatLevel: "LOW",
		Action:      "MONITOR",
	}

	// Rule 3: ELINT (High heat + Offset-pointing)
	iim.rules["elint"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			return em.SignalType == SignalTypeELINT && em.PowerDBm > -60.0
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.AttitudeMode == "OFFSET" && opt.IsStabilized
		},
		Intent:      "ELINT",
		ThreatLevel: "HIGH",
		Action:      "EMCON_INITIATE",
	}

	// Rule 4: Kinetic/Robotic Threat (No EM signature + Rapid proximity closure)
	iim.rules["kinetic_threat"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			return em.SignalType == SignalTypeUnknown && em.PowerDBm < -80.0
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.RelativeMotion && opt.StreakDetected && !opt.IsStabilized
		},
		Intent:      "KINETIC_THREAT",
		ThreatLevel: "CRITICAL",
		Action:      "EVASIVE_MANEUVER",
	}

	// Rule 5: Co-orbital Inspector (Low power + Station-keeping)
	iim.rules["inspector"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			return em.SignalType == SignalTypeComms && em.PowerDBm < -50.0
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.RelativeMotion && opt.IsStabilized && !opt.IsManeuvering
		},
		Intent:      "COORBITAL_INSPECTOR",
		ThreatLevel: "HIGH",
		Action:      "EMCON_B_INITIATE",
	}

	// Rule 6: Strategic Troposcatter (Sever/Gorizont signature)
	iim.rules["strategic_tropo"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			// 4.4 GHz QPSK with "Hollow" tropo signature
			return em.SignalType == SignalTypeELINT &&
				em.Modulation == "QPSK" &&
				em.Bandwidth >= 315e6 && em.Bandwidth <= 320e6
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			return opt.IsStabilized // Fixed ground station
		},
		Intent:      "STRATEGIC_TROPO",
		ThreatLevel: "HIGH",
		Action:      "TACTICAL_ALERT",
	}

	// Rule 7: Electronic Warfare (850-930 MHz + high energy, no ADS-B)
	iim.rules["electronic_warfare"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			// Orlan-10 frequency range with high power but no ADS-B correlation
			return em.SignalType == SignalTypeUnknown &&
				em.PowerDBm > -60.0 &&
				!em.IsBacklit // No ADS-B registration
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			// UAV-like behavior
			return opt.AttitudeMode == "NADIR" || opt.IsManeuvering
		},
		Intent:      "ELECTRONIC_WARFARE",
		ThreatLevel: "CRITICAL",
		Action:      "RF_COUNTERMEASURE",
	}

	// Rule 8: Kinetic Threat (PROFORMA + HCW residuals)
	iim.rules["kinetic_threat_sigint"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			// PROFORMA X/Ka-band signature
			return em.SignalType == SignalTypeELINT &&
				em.Bandwidth >= 4e9 && // X-band or Ka-band width
				em.IsNarrowBeam // High-gain radar dish
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			// Rapid proximity closure with radar tracking
			return opt.RelativeMotion && opt.StreakDetected && opt.IsManeuvering
		},
		Intent:      "KINETIC_THREAT_SIGINT",
		ThreatLevel: "CRITICAL",
		Action:      "EVASIVE_MANEUVER",
	}

	// Rule 9: Deception Detected (Clear COMINT + MASINT mismatch)
	iim.rules["deception_detected"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			// Clear voice comms without background noise
			return em.SignalType == SignalTypeComms &&
				em.Modulation == "FM_VOICE" &&
				em.IsNarrowBeam // Directed at our receiver
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			// MASINT mismatch: no thermal signature despite claimed position
			return opt.ThermalSig < 250.0 && opt.IsStabilized
		},
		Intent:      "DECEPTION_DETECTED",
		ThreatLevel: "HIGH",
		Action:      "EMCON_INITIATE",
	}

	// Rule 10: Ballistic Missile Activity (Telemetry detection)
	iim.rules["ballistic_missile"] = IntentRule{
		EMCondition: func(em EMSignature) bool {
			// L-band or S-band ballistic telemetry (using bandwidth as proxy)
			// L-band: 80-100 MHz bandwidth in 1.4-1.6 GHz range
			// S-band: 180-200 MHz bandwidth in 2.1-2.5 GHz range
			isLBandTelem := em.Bandwidth >= 80e6 && em.Bandwidth <= 100e6 &&
				em.PowerDBm > -70.0 // Telemetry typically low power
			isSBandTelem := em.Bandwidth >= 180e6 && em.Bandwidth <= 200e6 &&
				em.PowerDBm > -70.0
			return em.SignalType == SignalTypeUnknown && (isLBandTelem || isSBandTelem)
		},
		OpticalCondition: func(opt OpticalSignature) bool {
			// Ascent trajectory thermal signature
			return opt.ThermalSig > 600.0 && opt.IsManeuvering
		},
		Intent:      "BALLISTIC_MISSILE",
		ThreatLevel: "CRITICAL",
		Action:      "TACTICAL_ALERT",
	}
}

// InferIntent combines EM and optical signatures to infer intent
func (iim *IntentInferenceMatrix) InferIntent(em EMSignature, opt OpticalSignature) IntentClassification {
	bestMatch := IntentClassification{
		Intent:      "UNKNOWN",
		Confidence:  0.0,
		ThreatLevel: "LOW",
		Action:      "MONITOR",
	}

	for _, rule := range iim.rules {
		emMatch := rule.EMCondition(em)
		optMatch := rule.OpticalCondition(opt)

		if emMatch && optMatch {
			// Both conditions match - high confidence
			confidence := 0.9
			if confidence > bestMatch.Confidence {
				emContrib := EMWeight
				optContrib := OpticalWeight
				bestMatch = IntentClassification{
					Intent:              rule.Intent,
					Confidence:          confidence,
					EMContribution:      emContrib,
					OpticalContribution: optContrib,
					ThreatLevel:         rule.ThreatLevel,
					Action:              rule.Action,
				}
			}
		} else if emMatch || optMatch {
			// Partial match - lower confidence
			confidence := 0.6
			if confidence > bestMatch.Confidence {
				emContrib := 0.0
				optContrib := 0.0
				if emMatch {
					emContrib = EMWeight
				}
				if optMatch {
					optContrib = OpticalWeight
				}
				bestMatch = IntentClassification{
					Intent:              rule.Intent,
					Confidence:          confidence,
					EMContribution:      emContrib,
					OpticalContribution: optContrib,
					ThreatLevel:         rule.ThreatLevel,
					Action:              rule.Action,
				}
			}
		}
	}

	return bestMatch
}

// MultiModalFusion performs sensor fusion for enhanced intent inference
type MultiModalFusion struct {
	intentMatrix *IntentInferenceMatrix
	history      []IntentClassification
}

// NewMultiModalFusion creates a new multi-modal fusion system
func NewMultiModalFusion() *MultiModalFusion {
	return &MultiModalFusion{
		intentMatrix: NewIntentInferenceMatrix(),
		history:      make([]IntentClassification, 0),
	}
}

// FuseSignatures combines multiple observations for robust intent inference
func (mmf *MultiModalFusion) FuseSignatures(em EMSignature, opt OpticalSignature) IntentClassification {
	classification := mmf.intentMatrix.InferIntent(em, opt)

	// Add to history
	mmf.history = append(mmf.history, classification)

	// Perform temporal smoothing if we have history
	if len(mmf.history) >= 3 {
		classification = mmf.smoothClassification(classification)
	}

	return classification
}

// smoothClassification performs temporal smoothing of classifications
func (mmf *MultiModalFusion) smoothClassification(current IntentClassification) IntentClassification {
	// Weighted average of recent classifications
	recent := mmf.history[len(mmf.history)-3:]

	intentCounts := make(map[string]int)
	threatCounts := make(map[string]int)

	for _, hist := range recent {
		intentCounts[hist.Intent]++
		threatCounts[hist.ThreatLevel]++
	}

	// Find most common intent
	maxCount := 0
	dominantIntent := "UNKNOWN"
	for intent, count := range intentCounts {
		if count > maxCount {
			maxCount = count
			dominantIntent = intent
		}
	}

	// Find highest threat level
	highestThreat := "LOW"
	for threat := range threatCounts {
		if mmf.compareThreatLevel(threat, highestThreat) > 0 {
			highestThreat = threat
		}
	}

	// Boost confidence if consistent
	confidence := current.Confidence
	if maxCount >= 2 {
		confidence = math.Min(confidence+0.2, 1.0)
	}

	return IntentClassification{
		Intent:              dominantIntent,
		Confidence:          confidence,
		EMContribution:      current.EMContribution,
		OpticalContribution: current.OpticalContribution,
		ThreatLevel:         highestThreat,
		Action:              current.Action,
	}
}

// compareThreatLevel compares two threat levels
func (mmf *MultiModalFusion) compareThreatLevel(a, b string) int {
	order := map[string]int{
		"LOW":      0,
		"MEDIUM":   1,
		"HIGH":     2,
		"CRITICAL": 3,
	}

	return order[a] - order[b]
}

// GenerateIntentReport creates a comprehensive intent analysis report
func (mmf *MultiModalFusion) GenerateIntentReport(em EMSignature, opt OpticalSignature) IntentReport {
	classification := mmf.FuseSignatures(em, opt)

	report := IntentReport{
		Timestamp:           0, // Will be set by caller
		EMSignature:         em,
		OpticalSignature:    opt,
		Intent:              classification.Intent,
		Confidence:          classification.Confidence,
		ThreatLevel:         classification.ThreatLevel,
		Action:              classification.Action,
		EMContribution:      classification.EMContribution,
		OpticalContribution: classification.OpticalContribution,
		HistoryLength:       len(mmf.history),
	}

	return report
}

// IntentReport represents a comprehensive intent analysis
type IntentReport struct {
	Timestamp           float64
	EMSignature         EMSignature
	OpticalSignature    OpticalSignature
	Intent              string
	Confidence          float64
	ThreatLevel         string
	Action              string
	EMContribution      float64
	OpticalContribution float64
	HistoryLength       int
}

// CalculateDopplerVelocity calculates radial velocity from Doppler shift
func CalculateDopplerVelocity(dopplerShift, carrierFreq float64) float64 {
	// f_d = (v / c) * f_c
	// v = (f_d / f_c) * c
	return (dopplerShift / carrierFreq) * 299792458.0 // Speed of light in m/s
}

// EstimateSignalRange estimates range based on signal strength
func EstimateSignalRange(powerDBm, frequencyHz float64) float64 {
	// Friis transmission equation (simplified)
	// P_r = P_t + G_t + G_r - 20*log10(d) - 20*log10(f) - 32.44
	// Assuming 0 dBm transmit power and 0 dBi antennas
	// d = 10^((P_t - P_r - 32.44 - 20*log10(f)) / 20)

	pt := 0.0 // Assume 0 dBm transmit power
	pathLoss := pt - powerDBm - 32.44 - 20*math.Log10(frequencyHz/1e6)
	distanceKm := math.Pow(10, pathLoss/20.0)

	return distanceKm
}

// DetectAnomalousBehavior detects unusual patterns in EM or optical signatures
func (mmf *MultiModalFusion) DetectAnomalousBehavior(em EMSignature, opt OpticalSignature) []Anomaly {
	anomalies := make([]Anomaly, 0)

	// Check for sudden power increase
	if em.PowerDBm > -40.0 && em.SignalType != SignalTypeJammer {
		anomalies = append(anomalies, Anomaly{
			Type:        "POWER_SPIKE",
			Severity:    "HIGH",
			Description: "Unexpected high power signal detected",
		})
	}

	// Check for unexpected maneuver
	if opt.IsManeuvering && opt.AttitudeMode == "STAR_POINTING" {
		anomalies = append(anomalies, Anomaly{
			Type:        "UNEXPECTED_MANEUVER",
			Severity:    "MEDIUM",
			Description: "Maneuver detected during star-pointing mode",
		})
	}

	// Check for thermal anomaly
	if opt.ThermalSig > 400.0 && em.SignalType == SignalTypeUnknown {
		anomalies = append(anomalies, Anomaly{
			Type:        "THERMAL_ANOMALY",
			Severity:    "HIGH",
			Description: "High thermal signature without EM activity",
		})
	}

	return anomalies
}

// Anomaly represents detected anomalous behavior
type Anomaly struct {
	Type        string
	Severity    string
	Description string
}
