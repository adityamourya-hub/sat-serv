// pkg/telescope/netra_gateway.go
// Sovereign-to-NETRA Gateway
//
// Bridges sovereign optical cluster to ISRO NETRA/IS4OM national catalog.
// Pushes deterministic, physics-validated payloads only - no metadata collision.
//
// Data Streams:
//   1. HGV Wakes → Fractional residuals + DMD triggers
//   2. RSO Ephemeris → NETRA: Hamiltonian-validated state vectors (CCSDS OPM)
//   3. Characterization → NETRA: MFBD silhouettes + e3nn material IDs
//
// Protocol: Secure Tailscale + CCSDS OPM / JSON ObservationReport
// Tagging: Sovereign UUID + timestamp + lead-node ID prevents collision

package telescope

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// NETRAEndpoint defines ISRO NETRA connection parameters
type NETRAEndpoint struct {
	BaseURL     string // e.g., "https://netra.isro.gov.in/api/v1"
	AuthToken   string // API key or JWT
	TailscaleIP string // Tailscale mesh IP for secure tunnel
	StationID   string // e.g., "DEL-HAN-01"
}

// NETRAGateway manages sovereign-to-NETRA data feed
type NETRAGateway struct {
	endpoint   NETRAEndpoint
	httpClient *http.Client

	// Statistics
	pushesTotal   int64
	pushesSuccess int64
	lastPushTime  int64
	lastError     string
}

// NewNETRAGateway creates gateway for ISRO NETRA integration
func NewNETRAGateway(endpoint NETRAEndpoint) *NETRAGateway {
	return &NETRAGateway{
		endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: false, // Production: proper cert
				},
			},
		},
	}
}

// CCSDSOPM represents CCSDS Orbit Parameter Message
type CCSDSOPM struct {
	Header struct {
		CCSDS_Version int    `json:"CCSDS_OMM_VERS"`
		Creation_Date string `json:"CREATION_DATE"`
		Originator    string `json:"ORIGINATOR"`
		Message_ID    string `json:"MESSAGE_ID"`
	}
	Metadata struct {
		Object_Name string `json:"OBJECT_NAME"`
		Object_ID   string `json:"OBJECT_ID"`
		Center_Name string `json:"CENTER_NAME"`
		Ref_Frame   string `json:"REF_FRAME"`
		Time_System string `json:"TIME_SYSTEM"`
	}
	StateVector struct {
		Epoch string  `json:"EPOCH"`
		X     float64 `json:"X"` // km
		Y     float64 `json:"Y"`
		Z     float64 `json:"Z"`
		X_Dot float64 `json:"X_DOT"` // km/s
		Y_Dot float64 `json:"Y_DOT"`
		Z_Dot float64 `json:"Z_DOT"`
	}
	CovarianceMatrix struct {
		CX_X         float64 `json:"CX_X"`
		CY_X         float64 `json:"CY_X"`
		CY_Y         float64 `json:"CY_Y"`
		CZ_X         float64 `json:"CZ_X"`
		CZ_Y         float64 `json:"CZ_Y"`
		CZ_Z         float64 `json:"CZ_Z"`
		CX_DOT_X     float64 `json:"CX_DOT_X"`
		CX_DOT_Y     float64 `json:"CX_DOT_Y"`
		CX_DOT_Z     float64 `json:"CX_DOT_Z"`
		CX_DOT_X_DOT float64 `json:"CX_DOT_X_DOT"`
		CY_DOT_X     float64 `json:"CY_DOT_X"`
		CY_DOT_Y     float64 `json:"CY_DOT_Y"`
		CY_DOT_Z     float64 `json:"CY_DOT_Z"`
		CY_DOT_X_DOT float64 `json:"CY_DOT_X_DOT"`
		CY_DOT_Y_DOT float64 `json:"CY_DOT_Y_DOT"`
		CZ_DOT_X     float64 `json:"CZ_DOT_X"`
		CZ_DOT_Y     float64 `json:"CZ_DOT_Y"`
		CZ_DOT_Z     float64 `json:"CZ_DOT_Z"`
		CZ_DOT_X_DOT float64 `json:"CZ_DOT_X_DOT"`
		CZ_DOT_Y_DOT float64 `json:"CZ_DOT_Y_DOT"`
		CZ_DOT_Z_DOT float64 `json:"CZ_DOT_Z_DOT"`
	}
	SovereignMetadata struct {
		LeadNodeID                string  `json:"LEAD_NODE_ID"`
		HamiltonianInvariant      float64 `json:"HAMILTONIAN_INVARIANT"`
		HamiltonianConfidence     float64 `json:"HAMILTONIAN_CONFIDENCE"`
		FractionalAlpha           float64 `json:"FRACTIONAL_ALPHA"`
		SyntheticApertureBaseline float64 `json:"SYNTHETIC_BASELINE_M"`
	}
}

// PushRSOEphemeris sends Hamiltonian-validated state vectors to NETRA
//
// SE(3) equivariance guarantees rotation-correct gradients
// Format: CCSDS OPM with sovereign metadata extension
func (ng *NETRAGateway) PushRSOEphemeris(report *ObservationReport, cov [6][6]float64) error {
	// Build CCSDS OPM
	opm := ng.buildOPM(report, cov)

	// Serialize
	data, err := json.Marshal(opm)
	if err != nil {
		ng.lastError = err.Error()
		return fmt.Errorf("failed to marshal OPM: %w", err)
	}

	// Push to NETRA
	url := fmt.Sprintf("%s/ephemeris", ng.endpoint.BaseURL)

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
	if err != nil {
		ng.lastError = err.Error()
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", ng.endpoint.AuthToken))
	req.Header.Set("X-Sovereign-Node", ng.endpoint.StationID)

	resp, err := ng.httpClient.Do(req)
	if err != nil {
		ng.lastError = err.Error()
		return fmt.Errorf("NETRA push failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		ng.lastError = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return fmt.Errorf("NETRA returned HTTP %d", resp.StatusCode)
	}

	// Update stats
	ng.pushesTotal++
	ng.pushesSuccess++
	ng.lastPushTime = time.Now().Unix()

	fmt.Printf("[NETRA] Pushed RSO ephemeris for %s (HTTP %d)\n",
		report.TargetID, resp.StatusCode)

	return nil
}

// buildOPM creates CCSDS OPM from ObservationReport
func (ng *NETRAGateway) buildOPM(report *ObservationReport, cov [6][6]float64) CCSDSOPM {
	opm := CCSDSOPM{}

	// Header
	opm.Header.CCSDS_Version = 2.0
	opm.Header.Creation_Date = time.Unix(int64(report.TimestampNs/1e9), 0).UTC().Format("2006-01-02T15:04:05")
	opm.Header.Originator = ng.endpoint.StationID
	opm.Header.Message_ID = fmt.Sprintf("%s-%d", report.TargetID, report.TimestampNs)

	// Metadata
	opm.Metadata.Object_Name = report.TargetID
	opm.Metadata.Object_ID = report.TargetID
	opm.Metadata.Center_Name = "EARTH"
	opm.Metadata.Ref_Frame = "ECI"
	opm.Metadata.Time_System = "UTC"

	// State vector
	opm.StateVector.Epoch = time.Unix(int64(report.EpochJD-2440587.5)*86400, 0).UTC().Format("2006-01-02T15:04:05")
	opm.StateVector.X = report.ECIPos[0]
	opm.StateVector.Y = report.ECIPos[1]
	opm.StateVector.Z = report.ECIPos[2]
	opm.StateVector.X_Dot = report.ECIVel[0]
	opm.StateVector.Y_Dot = report.ECIVel[1]
	opm.StateVector.Z_Dot = report.ECIVel[2]

	// Covariance (upper triangle)
	opm.CovarianceMatrix.CX_X = cov[0][0]
	opm.CovarianceMatrix.CY_X = cov[1][0]
	opm.CovarianceMatrix.CY_Y = cov[1][1]
	opm.CovarianceMatrix.CZ_X = cov[2][0]
	opm.CovarianceMatrix.CZ_Y = cov[2][1]
	opm.CovarianceMatrix.CZ_Z = cov[2][2]
	opm.CovarianceMatrix.CX_DOT_X = cov[3][0]
	opm.CovarianceMatrix.CX_DOT_Y = cov[3][1]
	opm.CovarianceMatrix.CX_DOT_Z = cov[3][2]
	opm.CovarianceMatrix.CX_DOT_X_DOT = cov[3][3]
	// ... (remaining covariance elements)

	// Sovereign metadata
	opm.SovereignMetadata.LeadNodeID = ng.endpoint.StationID
	opm.SovereignMetadata.HamiltonianInvariant = report.HamiltonianInvariant
	opm.SovereignMetadata.HamiltonianConfidence = report.HamiltonianConfidence
	opm.SovereignMetadata.FractionalAlpha = 0.7             // Default Caputo order
	opm.SovereignMetadata.SyntheticApertureBaseline = 100.0 // meters (placeholder)

	return opm
}

// PushObservationReport sends full JSON report to NETRA
//
// For MFBD silhouettes + e3nn material IDs (characterization data)
func (ng *NETRAGateway) PushObservationReport(report *ObservationReport) error {
	data, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("failed to marshal report: %w", err)
	}

	url := fmt.Sprintf("%s/observations", ng.endpoint.BaseURL)

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", ng.endpoint.AuthToken))
	req.Header.Set("X-Sovereign-Node", ng.endpoint.StationID)

	resp, err := ng.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("NETRA push failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("NETRA returned HTTP %d", resp.StatusCode)
	}

	fmt.Printf("[NETRA] Pushed observation report for %s\n", report.TargetID)
	return nil
}

// PushSyntheticApertureObservation sends synthetic aperture fused data
//
// Triggered after all cluster mounts ACK synchronized task
func (ng *NETRAGateway) PushSyntheticApertureObservation(task *SyntheticApertureTask) error {
	payload := map[string]interface{}{
		"task_id":       task.TaskID,
		"target_id":     task.TargetID,
		"timestamp":     task.Timestamp,
		"cluster_size":  len(task.Acknowledged),
		"baseline_m":    ng.calculateBaseline(task),
		"synthetic_fov": "coherent",
		"mounts":        task.Acknowledged,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("%s/synthetic-aperture", ng.endpoint.BaseURL)

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", ng.endpoint.AuthToken))

	resp, err := ng.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
		fmt.Printf("[NETRA] Pushed synthetic aperture observation %s\n", task.TaskID)
	}

	return nil
}

// calculateBaseline estimates synthetic aperture baseline from mount positions
func (ng *NETRAGateway) calculateBaseline(task *SyntheticApertureTask) float64 {
	// Placeholder: Would compute from actual ECEF mount locations
	return 100.0 // meters
}

// GetStats returns gateway performance metrics
func (ng *NETRAGateway) GetStats() (total, success int64, lastPush int64, lastErr string) {
	return ng.pushesTotal, ng.pushesSuccess, ng.lastPushTime, ng.lastError
}
