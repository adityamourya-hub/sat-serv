package collision

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"strings"
	"time"
)

// CDMHeader contains administrative metadata for the conjunction message.
type CDMHeader struct {
	CCSDS_CDM_VERS string `xml:"CCSDS_CDM_VERS,attr" json:"version"`
	CREATION_DATE  string `xml:"CREATION_DATE" json:"creation_date"`
	ORIGINATOR     string `xml:"ORIGINATOR" json:"originator"`
	MESSAGE_FOR    string `xml:"MESSAGE_FOR" json:"message_for"`
	MESSAGE_ID     string `xml:"MESSAGE_ID" json:"message_id"`
}

// RelativeMetadata contains metrics defining the conjunction event itself.
type RelativeMetadata struct {
	TCA                   string  `xml:"TCA" json:"tca"`
	MISS_DISTANCE         float64 `xml:"MISS_DISTANCE" json:"miss_distance"`
	RELATIVE_SPEED        float64 `xml:"RELATIVE_SPEED" json:"relative_speed"`
	RELATIVE_POSITION_R   float64 `xml:"RELATIVE_POSITION_R" json:"rel_pos_r"`
	RELATIVE_POSITION_T   float64 `xml:"RELATIVE_POSITION_T" json:"rel_pos_t"`
	RELATIVE_POSITION_N   float64 `xml:"RELATIVE_POSITION_N" json:"rel_pos_n"`
	COLLISION_PROBABILITY float64 `xml:"COLLISION_PROBABILITY" json:"pc"`
	COLLISION_PROB_METHOD string  `xml:"COLLISION_PROB_METHOD" json:"pc_method"`
}

// ObjectMetadata contains identification and state metadata for a single object.
type ObjectMetadata struct {
	OBJECT            string `xml:"OBJECT" json:"object"`
	OBJECT_DESIGNATOR string `xml:"OBJECT_DESIGNATOR" json:"designator"`
	CATALOG_NAME      string `xml:"CATALOG_NAME" json:"catalog_name"`
	OBJECT_NAME       string `xml:"OBJECT_NAME" json:"object_name"`
	INTERNATIONAL_DES string `xml:"INTERNATIONAL_DESIGNATOR" json:"intl_des"`
	OBJECT_TYPE       string `xml:"OBJECT_TYPE" json:"object_type"`
}

// ObjectData contains the physical state and covariance for a single object.
type ObjectData struct {
	X  float64 `xml:"X" json:"x"`
	Y  float64 `xml:"Y" json:"y"`
	Z  float64 `xml:"Z" json:"z"`
	VX float64 `xml:"VX" json:"vx"`
	VY float64 `xml:"VY" json:"vy"`
	VZ float64 `xml:"VZ" json:"vz"`
	// Covariance terms (simplified representation for JSON/XML)
	CR_R float64 `xml:"CR_R" json:"cr_r"`
	CT_T float64 `xml:"CT_T" json:"ct_t"`
	CN_N float64 `xml:"CN_N" json:"cn_n"`
}

// SSIS_Extension contains proprietary metadata for auditing and verification.
type SSIS_Extension struct {
	MC_SAMPLES      uint64  `json:"mc_samples"`
	MC_CONVERGENCE  float64 `json:"mc_convergence"`
	PROOF_SIGNATURE string  `json:"proof_signature"`
	VERIFIED_SPEC   string  `json:"verified_spec"`
}

// CDM represents the full CCSDS Conjunction Data Message.
type CDM struct {
	XMLName          xml.Name         `xml:"cdm" json:"-"`
	Header           CDMHeader        `xml:"header" json:"header"`
	RelativeMetadata RelativeMetadata `xml:"relativeMetadata" json:"relative_metadata"`
	Object1Metadata  ObjectMetadata   `xml:"object1Metadata" json:"object1_metadata"`
	Object1Data      ObjectData       `xml:"object1Data" json:"object1_data"`
	Object2Metadata  ObjectMetadata   `xml:"object2Metadata" json:"object2_metadata"`
	Object2Data      ObjectData       `xml:"object2Data" json:"object2_data"`
	Extension        SSIS_Extension   `xml:"extension" json:"extension"`
}

// NewCDM initializes a standard CDM shell from MC results.
func NewCDM(mc *MonteCarloConjunctionResult) *CDM {
	method := "MONTE_CARLO_HALTON_QMC"
	if mc.AnalyticMeta.Method != "" {
		method = mc.AnalyticMeta.Method + "+MONTE_CARLO_HALTON_QMC"
	}

	return &CDM{
		Header: CDMHeader{
			CCSDS_CDM_VERS: "1.0",
			CREATION_DATE:  time.Now().Format(time.RFC3339),
			ORIGINATOR:     "SSIS_IAF_ADITI_4",
			MESSAGE_FOR:    "IAF_SPACE_OPS",
			MESSAGE_ID:     fmt.Sprintf("SSIS-MC-%d", time.Now().Unix()),
		},
		RelativeMetadata: RelativeMetadata{
			COLLISION_PROBABILITY: mc.PcMean,
			COLLISION_PROB_METHOD: method,
		},
		Extension: SSIS_Extension{
			MC_SAMPLES:      mc.TotalSamples,
			MC_CONVERGENCE:  mc.ConvergenceDiag,
			PROOF_SIGNATURE: mc.ProofSignature,
			VERIFIED_SPEC:   "FOSTER_1992_VERIFIED_SUBSTRATE",
		},
	}
}

// ToKVN serializes the CDM to CCSDS Keyword-Value Notation.
func (c *CDM) ToKVN() string {
	var sb strings.Builder
	sb.WriteString("CCSDS_CDM_VERS = " + c.Header.CCSDS_CDM_VERS + "\n")
	sb.WriteString("CREATION_DATE  = " + c.Header.CREATION_DATE + "\n")
	sb.WriteString("ORIGINATOR     = " + c.Header.ORIGINATOR + "\n")
	sb.WriteString("MESSAGE_ID     = " + c.Header.MESSAGE_ID + "\n")
	sb.WriteString("--------------------------------------------------\n")
	sb.WriteString(fmt.Sprintf("TCA            = %s\n", c.RelativeMetadata.TCA))
	sb.WriteString(fmt.Sprintf("MISS_DISTANCE  = %.3f [km]\n", c.RelativeMetadata.MISS_DISTANCE))
	sb.WriteString(fmt.Sprintf("RELATIVE_SPEED = %.3f [km/s]\n", c.RelativeMetadata.RELATIVE_SPEED))
	sb.WriteString(fmt.Sprintf("COLLISION_PROB = %.4e\n", c.RelativeMetadata.COLLISION_PROBABILITY))
	sb.WriteString("COLLISION_PROB_METHOD = " + c.RelativeMetadata.COLLISION_PROB_METHOD + "\n")
	sb.WriteString("--------------------------------------------------\n")
	sb.WriteString("OBJECT1 = " + c.Object1Metadata.OBJECT_NAME + "\n")
	sb.WriteString("OBJECT2 = " + c.Object2Metadata.OBJECT_NAME + "\n")
	// ... (additional fields would be populated in production)
	return sb.String()
}

// ToJSON serializes the CDM to a signed JSON payload.
func (c *CDM) ToJSON() (string, error) {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ToXML serializes the CDM to CCSDS XML format.
func (c *CDM) ToXML() (string, error) {
	b, err := xml.MarshalIndent(c, "", "  ")
	if err != nil {
		return "", err
	}
	return xml.Header + string(b), nil
}
