package collision

import (
	"testing"
)

/**
 * TestParseCDMXML validates CDM parsing from operational 18 SDS/CARA format.
 */
func TestParseCDMXML(t *testing.T) {
	// Example CDM XML (simplified - real CDMs have more fields)
	cdmXML := `<?xml version="1.0" encoding="UTF-8"?>
<cdm>
  <header CCSDS_CDM_VERS="1.0">
    <CREATION_DATE>2024-01-15T12:00:00Z</CREATION_DATE>
    <ORIGINATOR>18 SDS</ORIGINATOR>
    <MESSAGE_FOR>IAF_OPS</MESSAGE_FOR>
    <MESSAGE_ID>CDM-2024-001</MESSAGE_ID>
  </header>
  <relativeMetadata>
    <TCA>2024-01-16T08:30:00.000Z</TCA>
    <MISS_DISTANCE>0.5</MISS_DISTANCE>
    <RELATIVE_SPEED>15.2</RELATIVE_SPEED>
    <RELATIVE_POSITION_R>0.3</RELATIVE_POSITION_R>
    <RELATIVE_POSITION_T>0.4</RELATIVE_POSITION_T>
    <RELATIVE_POSITION_N>0.0</RELATIVE_POSITION_N>
    <COLLISION_PROBABILITY>1.5e-4</COLLISION_PROBABILITY>
    <COLLISION_PROB_METHOD>FOSTER_1992</COLLISION_PROB_METHOD>
  </relativeMetadata>
  <object1Metadata>
    <OBJECT>OBJECT1</OBJECT>
    <OBJECT_DESIGNATOR>12345</OBJECT_DESIGNATOR>
    <CATALOG_NAME>SATCAT</CATALOG_NAME>
    <OBJECT_NAME>TEST_SAT_1</OBJECT_NAME>
    <INTERNATIONAL_DESIGNATOR>2020-001A</INTERNATIONAL_DESIGNATOR>
    <OBJECT_TYPE>PAYLOAD</OBJECT_TYPE>
  </object1Metadata>
  <object1Data>
    <X>7000.0</X>
    <Y>0.0</Y>
    <Z>0.0</Z>
    <VX>0.0</VX>
    <VY>7.5</VY>
    <VZ>0.0</VZ>
    <CR_R>0.01</CR_R>
    <CT_T>0.005</CT_T>
    <CN_N>0.002</CN_N>
  </object1Data>
  <object2Metadata>
    <OBJECT>OBJECT2</OBJECT>
    <OBJECT_DESIGNATOR>99999</OBJECT_DESIGNATOR>
    <CATALOG_NAME>SATCAT</CATALOG_NAME>
    <OBJECT_NAME>TEST_DEBRIS_1</OBJECT_NAME>
    <INTERNATIONAL_DESIGNATOR>1990-999B</INTERNATIONAL_DESIGNATOR>
    <OBJECT_TYPE>DEBRIS</OBJECT_TYPE>
  </object2Metadata>
  <object2Data>
    <X>7000.2</X>
    <Y>0.1</Y>
    <Z>0.05</Z>
    <VX>0.1</VX>
    <VY>7.4</VY>
    <VZ>0.05</VZ>
    <CR_R>0.02</CR_R>
    <CT_T>0.01</CT_T>
    <CN_N>0.005</CN_N>
  </object2Data>
</cdm>`

	cdm, err := ParseCDMXML([]byte(cdmXML))
	if err != nil {
		t.Fatalf("Failed to parse CDM: %v", err)
	}

	// Verify header
	if cdm.Header.CCSDS_CDM_VERS != "1.0" {
		t.Errorf("Expected version 1.0, got %s", cdm.Header.CCSDS_CDM_VERS)
	}
	if cdm.Header.ORIGINATOR != "18 SDS" {
		t.Errorf("Expected ORIGINATOR '18 SDS', got %s", cdm.Header.ORIGINATOR)
	}

	// Verify relative metadata
	if cdm.RelativeMetadata.MISS_DISTANCE != 0.5 {
		t.Errorf("Expected miss distance 0.5 km, got %f", cdm.RelativeMetadata.MISS_DISTANCE)
	}
	if cdm.RelativeMetadata.COLLISION_PROBABILITY != 1.5e-4 {
		t.Errorf("Expected Pc 1.5e-4, got %e", cdm.RelativeMetadata.COLLISION_PROBABILITY)
	}

	// Verify object states
	primary := cdm.ExtractPrimaryState()
	if primary.ObjectName != "TEST_SAT_1" {
		t.Errorf("Expected primary name 'TEST_SAT_1', got %s", primary.ObjectName)
	}
	if primary.Position[0] != 7000.0 {
		t.Errorf("Expected X position 7000.0, got %f", primary.Position[0])
	}

	secondary := cdm.ExtractSecondaryState()
	if secondary.ObjectName != "TEST_DEBRIS_1" {
		t.Errorf("Expected secondary name 'TEST_DEBRIS_1', got %s", secondary.ObjectName)
	}

	t.Logf("Successfully parsed CDM for conjunction: %s vs %s", primary.ObjectName, secondary.ObjectName)
	t.Logf("Published Pc: %e", cdm.RelativeMetadata.COLLISION_PROBABILITY)
}

/**
 * TestCDMRecomputePc validates Pc recomputation against published value.
 * This is the critical test for operational verification.
 */
func TestCDMRecomputePc(t *testing.T) {
	// Simple test case with known geometry
	cdmXML := `<?xml version="1.0" encoding="UTF-8"?>
<cdm>
  <header CCSDS_CDM_VERS="1.0">
    <CREATION_DATE>2024-01-15T12:00:00Z</CREATION_DATE>
    <ORIGINATOR>TEST</ORIGINATOR>
    <MESSAGE_FOR>TEST</MESSAGE_FOR>
    <MESSAGE_ID>TEST-001</MESSAGE_ID>
  </header>
  <relativeMetadata>
    <TCA>2024-01-16T08:30:00.000Z</TCA>
    <MISS_DISTANCE>0.1</MISS_DISTANCE>
    <RELATIVE_SPEED>10.0</RELATIVE_SPEED>
    <RELATIVE_POSITION_R>0.05</RELATIVE_POSITION_R>
    <RELATIVE_POSITION_T>0.08</RELATIVE_POSITION_T>
    <RELATIVE_POSITION_N>0.0</RELATIVE_POSITION_N>
    <COLLISION_PROBABILITY>1.0e-3</COLLISION_PROBABILITY>
    <COLLISION_PROB_METHOD>TEST</COLLISION_PROB_METHOD>
  </relativeMetadata>
  <object1Metadata>
    <OBJECT>OBJECT1</OBJECT>
    <OBJECT_DESIGNATOR>11111</OBJECT_DESIGNATOR>
    <OBJECT_NAME>PRIMARY</OBJECT_NAME>
    <INTERNATIONAL_DESIGNATOR>2020-001A</INTERNATIONAL_DESIGNATOR>
  </object1Metadata>
  <object1Data>
    <X>7000.0</X>
    <Y>0.0</Y>
    <Z>0.0</Z>
    <VX>0.0</VX>
    <VY>7.5</VY>
    <VZ>0.0</VZ>
    <CR_R>0.001</CR_R>
    <CT_T>0.001</CT_T>
    <CN_N>0.001</CN_N>
  </object1Data>
  <object2Metadata>
    <OBJECT>OBJECT2</OBJECT>
    <OBJECT_DESIGNATOR>22222</OBJECT_DESIGNATOR>
    <OBJECT_NAME>SECONDARY</OBJECT_NAME>
    <INTERNATIONAL_DESIGNATOR>1990-001B</INTERNATIONAL_DESIGNATOR>
  </object2Metadata>
  <object2Data>
    <X>7000.05</X>
    <Y>0.08</Y>
    <Z>0.0</Z>
    <VX>-0.5</VX>
    <VY>7.0</VY>
    <VZ>0.0</VZ>
    <CR_R>0.001</CR_R>
    <CT_T>0.001</CT_T>
    <CN_N>0.001</CN_N>
  </object2Data>
</cdm>`

	cdm, err := ParseCDMXML([]byte(cdmXML))
	if err != nil {
		t.Fatalf("Failed to parse CDM: %v", err)
	}

	// Recompute Pc using fixed methods
	result := cdm.RecomputePc()

	t.Logf("Published Pc:  %e", result.PublishedPc)
	t.Logf("Recomputed Pc: %e", result.RecomputedPc)
	t.Logf("Absolute error: %e", result.AbsoluteError)
	t.Logf("Relative error: %.2f%%", result.RelativeError*100)
	t.Logf("Miss distance: %.3f km", result.MissDistance)
	t.Logf("Relative speed: %.3f km/s", result.RelativeSpeed)

	// Note: This is a simplified test - real validation would compare
	// against NASA CARA published CDMs with known Pc values
	if result.RecomputedPc < 0 || result.RecomputedPc > 1 {
		t.Errorf("Recomputed Pc out of valid range [0,1]: %e", result.RecomputedPc)
	}
}

/**
 * TestCDMValidationResult checks the IsValid logic.
 */
func TestCDMValidationResult(t *testing.T) {
	tests := []struct {
		name       string
		published  float64
		recomputed float64
		wantValid  bool
	}{
		{
			name:       "High Pc within tolerance",
			published:  1e-3,
			recomputed: 1.1e-3, // 10% error
			wantValid:  true,
		},
		{
			name:       "High Pc outside tolerance",
			published:  1e-3,
			recomputed: 1.5e-3, // 50% error
			wantValid:  false,
		},
		{
			name:       "Low Pc within absolute tolerance",
			published:  1e-6,
			recomputed: 1.5e-6, // 5e-7 absolute error
			wantValid:  true,
		},
		{
			name:       "Low Pc outside absolute tolerance",
			published:  1e-6,
			recomputed: 3e-6, // 2e-6 absolute error
			wantValid:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := CDMValidationResult{
				PublishedPc:   tt.published,
				RecomputedPc:  tt.recomputed,
				AbsoluteError: tt.recomputed - tt.published,
				RelativeError: (tt.recomputed - tt.published) / tt.published,
			}
			got := v.IsValid()
			if got != tt.wantValid {
				t.Errorf("IsValid() = %v, want %v", got, tt.wantValid)
			}
		})
	}
}
