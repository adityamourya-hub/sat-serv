// Package comms provides DRYAD authentication for tactical communications
// Implements paper-based two-factor authentication for air-gapped systems
package comms

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"time"
)

const (
	// DRYAD constants
	AuthenticatorLength = 6   // Number of characters in authenticator
	GridSize            = 25  // 5x5 grid
	MaxAuthenticators   = 100 // Maximum authenticators per sheet
	AuthTTL             = 24 * time.Hour
)

// DRYADSheet represents a DRYAD authentication sheet
type DRYADSheet struct {
	SheetID        string
	Authenticators map[string]string // Authenticator -> Grid coordinates
	ExpiryDate     time.Time
	IsActive       bool
	IssuedTo       string
	IssuedBy       string
}

// DRYADAuthenticator represents a single authenticator
type DRYADAuthenticator struct {
	Word      string
	GridCoord string // e.g., "A-3", "B-12"
	Used      bool
	UsedAt    time.Time
}

// DRYADSystem manages DRYAD authentication
type DRYADSystem struct {
	currentSheet       *DRYADSheet
	usedAuthenticators map[string]*DRYADAuthenticator
	authHistory        []AuthRecord
}

// AuthRecord represents an authentication record
type AuthRecord struct {
	Authenticator string
	GridCoord     string
	UsedBy        string
	UsedAt        time.Time
	IsValid       bool
}

// NewDRYADSystem creates a new DRYAD authentication system
func NewDRYADSystem() *DRYADSystem {
	return &DRYADSystem{
		usedAuthenticators: make(map[string]*DRYADAuthenticator),
		authHistory:        make([]AuthRecord, 0),
	}
}

// GenerateSheet generates a new DRYAD sheet
func (ds *DRYADSystem) GenerateSheet(issuedTo, issuedBy string) *DRYADSheet {
	// Generate unique sheet ID
	sheetID := fmt.Sprintf("DRYAD-%d", time.Now().Unix())

	// Generate authenticators
	authenticators := make(map[string]string)

	// Use common DRYAD words (simplified)
	words := []string{
		"APPLE", "BRAVO", "CHARLIE", "DELTA", "ECHO",
		"FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET",
		"KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR",
		"PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO",
		"UNIFORM", "VICTOR", "WHISKEY", "XRAY", "YANKEE", "ZULU",
	}

	for i := 0; i < MaxAuthenticators; i++ {
		wi, _ := rand.Int(rand.Reader, big.NewInt(int64(len(words))))
		word := words[wi.Int64()]
		ri, _ := rand.Int(rand.Reader, big.NewInt(5))
		row := int(ri.Int64()) // A-E
		ci, _ := rand.Int(rand.Reader, big.NewInt(5))
		col := int(ci.Int64()) + 1 // 1-5
		gridCoord := fmt.Sprintf("%c-%d", 'A'+row, col)
		authenticators[word] = gridCoord
	}

	sheet := &DRYADSheet{
		SheetID:        sheetID,
		Authenticators: authenticators,
		ExpiryDate:     time.Now().Add(AuthTTL),
		IsActive:       true,
		IssuedTo:       issuedTo,
		IssuedBy:       issuedBy,
	}

	ds.currentSheet = sheet
	return sheet
}

// LoadSheet loads an existing DRYAD sheet
func (ds *DRYADSystem) LoadSheet(sheet *DRYADSheet) error {
	if time.Now().After(sheet.ExpiryDate) {
		return fmt.Errorf("sheet expired on %s", sheet.ExpiryDate)
	}

	ds.currentSheet = sheet
	return nil
}

// Authenticate performs DRYAD authentication
func (ds *DRYADSystem) Authenticate(authenticator, gridCoord, identity string) (bool, error) {
	if ds.currentSheet == nil {
		return false, fmt.Errorf("no active DRYAD sheet")
	}

	if !ds.currentSheet.IsActive {
		return false, fmt.Errorf("DRYAD sheet not active")
	}

	if time.Now().After(ds.currentSheet.ExpiryDate) {
		return false, fmt.Errorf("DRYAD sheet expired")
	}

	// Check if authenticator exists
	expectedCoord, exists := ds.currentSheet.Authenticators[authenticator]
	if !exists {
		// Log failed authentication
		ds.authHistory = append(ds.authHistory, AuthRecord{
			Authenticator: authenticator,
			GridCoord:     gridCoord,
			UsedBy:        identity,
			UsedAt:        time.Now(),
			IsValid:       false,
		})
		return false, fmt.Errorf("invalid authenticator")
	}

	// Check if grid coordinate matches
	if expectedCoord != gridCoord {
		ds.authHistory = append(ds.authHistory, AuthRecord{
			Authenticator: authenticator,
			GridCoord:     gridCoord,
			UsedBy:        identity,
			UsedAt:        time.Now(),
			IsValid:       false,
		})
		return false, fmt.Errorf("grid coordinate mismatch")
	}

	// Check if already used
	if used, exists := ds.usedAuthenticators[authenticator]; exists && used.Used {
		ds.authHistory = append(ds.authHistory, AuthRecord{
			Authenticator: authenticator,
			GridCoord:     gridCoord,
			UsedBy:        identity,
			UsedAt:        time.Now(),
			IsValid:       false,
		})
		return false, fmt.Errorf("authenticator already used")
	}

	// Mark as used
	ds.usedAuthenticators[authenticator] = &DRYADAuthenticator{
		Word:      authenticator,
		GridCoord: gridCoord,
		Used:      true,
		UsedAt:    time.Now(),
	}

	// Log successful authentication
	ds.authHistory = append(ds.authHistory, AuthRecord{
		Authenticator: authenticator,
		GridCoord:     gridCoord,
		UsedBy:        identity,
		UsedAt:        time.Now(),
		IsValid:       true,
	})

	return true, nil
}

// GetAuthenticator retrieves an unused authenticator
func (ds *DRYADSystem) GetAuthenticator() (string, error) {
	if ds.currentSheet == nil {
		return "", fmt.Errorf("no active DRYAD sheet")
	}

	// Find unused authenticator
	for word, coord := range ds.currentSheet.Authenticators {
		if used, exists := ds.usedAuthenticators[word]; !exists || !used.Used {
			return fmt.Sprintf("%s %s", word, coord), nil
		}
	}

	return "", fmt.Errorf("no unused authenticators available")
}

// InvalidateSheet invalidates the current sheet
func (ds *DRYADSystem) InvalidateSheet(reason string) {
	if ds.currentSheet != nil {
		ds.currentSheet.IsActive = false
		// Log invalidation
		ds.authHistory = append(ds.authHistory, AuthRecord{
			Authenticator: "SHEET_INVALIDATED",
			GridCoord:     reason,
			UsedBy:        "SYSTEM",
			UsedAt:        time.Now(),
			IsValid:       false,
		})
	}
}

// GetAuthHistory returns authentication history
func (ds *DRYADSystem) GetAuthHistory() []AuthRecord {
	return ds.authHistory
}

// GetUsedCount returns the number of used authenticators
func (ds *DRYADSystem) GetUsedCount() int {
	return len(ds.usedAuthenticators)
}

// GetRemainingCount returns the number of remaining authenticators
func (ds *DRYADSystem) GetRemainingCount() int {
	if ds.currentSheet == nil {
		return 0
	}
	return len(ds.currentSheet.Authenticators) - ds.GetUsedCount()
}

// GetSheetStatus returns the status of the current sheet
func (ds *DRYADSystem) GetSheetStatus() *SheetStatus {
	if ds.currentSheet == nil {
		return &SheetStatus{
			HasSheet:       false,
			IsActive:       false,
			IsExpired:      false,
			UsedCount:      0,
			RemainingCount: 0,
		}
	}

	return &SheetStatus{
		HasSheet:       true,
		IsActive:       ds.currentSheet.IsActive,
		IsExpired:      time.Now().After(ds.currentSheet.ExpiryDate),
		UsedCount:      ds.GetUsedCount(),
		RemainingCount: ds.GetRemainingCount(),
		SheetID:        ds.currentSheet.SheetID,
		ExpiryDate:     ds.currentSheet.ExpiryDate,
	}
}

// SheetStatus represents the status of a DRYAD sheet
type SheetStatus struct {
	HasSheet       bool
	IsActive       bool
	IsExpired      bool
	UsedCount      int
	RemainingCount int
	SheetID        string
	ExpiryDate     time.Time
}

// Challenge creates an authentication challenge
func (ds *DRYADSystem) Challenge(identity string) (string, error) {
	authenticator, err := ds.GetAuthenticator()
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("AUTHENTICATE: %s", authenticator), nil
}

// Verify verifies an authentication response
func (ds *DRYADSystem) Verify(identity, response string) (bool, error) {
	// Parse response (format: "WORD GRIDCOORD")
	var word, gridCoord string
	_, err := fmt.Sscanf(response, "%s %s", &word, &gridCoord)
	if err != nil {
		return false, fmt.Errorf("invalid response format")
	}

	return ds.Authenticate(word, gridCoord, identity)
}

// TwoFactorAuth performs two-factor authentication using DRYAD
func (ds *DRYADSystem) TwoFactorAuth(identity, password, authenticator, gridCoord string) (bool, error) {
	// First factor: Password (simplified - in production use proper password hashing)
	if password == "" {
		return false, fmt.Errorf("password required")
	}

	// Second factor: DRYAD
	dryadValid, err := ds.Authenticate(authenticator, gridCoord, identity)
	if err != nil {
		return false, err
	}

	return dryadValid, nil
}
