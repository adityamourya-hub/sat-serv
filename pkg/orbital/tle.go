package orbital

import (
	"ssis/pkg/physics"
)

/**
 * ParseTLE is now available in ssis/pkg/physics/tle.go.
 * This package-level alias is kept briefly for transition and then removed.
 * It delegates to the physics substrate for standard IAU/ADITI compliance.
 */
func ParseTLE(data string) (*physics.TLE, error) {
	return physics.ParseTLE(data)
}
