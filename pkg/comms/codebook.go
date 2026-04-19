// Package comms provides codebook and OTP encryption for tactical communications
// Implements numerical codes, OPCODEs, and One-Time Pads for message security
package comms

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"strings"
	"time"
)

const (
	// Codebook constants
	CodeLength      = 3    // Standard code length
	OPCODELength    = 3    // OPCODE length
	OTPLength       = 1000 // OTP length in digits
	MaxCodebookSize = 1000 // Maximum entries in codebook
)

// CodebookType represents the type of codebook
type CodebookType string

const (
	CodebookNumerical CodebookType = "NUMERICAL" // Numerical codes (e.g., 837)
	CodebookOPCODE   CodebookType = "OPCODE"    // OPCODEs (e.g., DIM)
	CodebookMixed     CodebookType = "MIXED"     // Mixed codes
)

// CodebookEntry represents a single codebook entry
type CodebookEntry struct {
	Code     string
	Meaning  string
	Category string
	Used     bool
	UsedAt   time.Time
}

// Codebook represents a tactical codebook
type Codebook struct {
	BookID    string
	BookType  CodebookType
	Entries   map[string]*CodebookEntry
	IssuedTo  string
	IssuedBy  string
	IsActive  bool
}

// NewCodebook creates a new codebook
func NewCodebook(bookID string, bookType CodebookType, issuedTo, issuedBy string) *Codebook {
	return &Codebook{
		BookID:    bookID,
		BookType:  bookType,
		Entries:   make(map[string]*CodebookEntry),
		IssuedTo:  issuedTo,
		IssuedBy:  issuedBy,
		IsActive:  true,
	}
}

// AddEntry adds an entry to the codebook
func (cb *Codebook) AddEntry(code, meaning, category string) error {
	if len(code) != CodeLength && cb.BookType == CodebookNumerical {
		return fmt.Errorf("code must be %d digits for numerical codebook", CodeLength)
	}
	
	cb.Entries[code] = &CodebookEntry{
		Code:     code,
		Meaning:  meaning,
		Category: category,
		Used:     false,
	}
	return nil
}

// Lookup looks up a code in the codebook
func (cb *Codebook) Lookup(code string) (*CodebookEntry, error) {
	entry, exists := cb.Entries[code]
	if !exists {
		return nil, fmt.Errorf("code not found: %s", code)
	}
	return entry, nil
}

// ReverseLookup finds a code by meaning
func (cb *Codebook) ReverseLookup(meaning string) (string, error) {
	for code, entry := range cb.Entries {
		if entry.Meaning == meaning {
			return code, nil
		}
	}
	return "", fmt.Errorf("meaning not found: %s", meaning)
}

// Encode encodes a message using the codebook
func (cb *Codebook) Encode(message string) (string, error) {
	if !cb.IsActive {
		return "", fmt.Errorf("codebook not active")
	}
	
	words := strings.Fields(message)
	encoded := make([]string, 0)
	
	for _, word := range words {
		code, err := cb.ReverseLookup(word)
		if err != nil {
			// If not found, keep original word
			encoded = append(encoded, word)
		} else {
			encoded = append(encoded, code)
			// Mark as used
			if entry, exists := cb.Entries[code]; exists {
				entry.Used = true
				entry.UsedAt = time.Now()
			}
		}
	}
	
	return strings.Join(encoded, " "), nil
}

// Decode decodes a message using the codebook
func (cb *Codebook) Decode(encoded string) (string, error) {
	if !cb.IsActive {
		return "", fmt.Errorf("codebook not active")
	}
	
	codes := strings.Fields(encoded)
	decoded := make([]string, 0)
	
	for _, code := range codes {
		entry, err := cb.Lookup(code)
		if err != nil {
			// If not found, keep original code
			decoded = append(decoded, code)
		} else {
			decoded = append(decoded, entry.Meaning)
		}
	}
	
	return strings.Join(decoded, " "), nil
}

// GetUsedCount returns the number of used codes
func (cb *Codebook) GetUsedCount() int {
	count := 0
	for _, entry := range cb.Entries {
		if entry.Used {
			count++
		}
	}
	return count
}

// Invalidate invalidates the codebook
func (cb *Codebook) Invalidate(reason string) {
	cb.IsActive = false
}

// OTP represents a One-Time Pad
type OTP struct {
	PadID    string
	Key      []int // Random digits
	Position int    // Current position in the pad
	Expiry   time.Time
	IsActive bool
}

// NewOTP creates a new One-Time Pad
func NewOTP(padID string, length int) (*OTP, error) {
	key := make([]int, length)
	
	// Generate cryptographically secure random digits
	for i := 0; i < length; i++ {
		n, err := rand.Int(rand.Reader, big.NewInt(10))
		if err != nil {
			return nil, fmt.Errorf("failed to generate random digit: %w", err)
		}
		key[i] = int(n.Int64())
	}
	
	return &OTP{
		PadID:    padID,
		Key:      key,
		Position: 0,
		Expiry:   time.Now().Add(24 * time.Hour),
		IsActive: true,
	}, nil
}

// Encrypt encrypts a numerical message using OTP
func (otp *OTP) Encrypt(message string) (string, error) {
	if !otp.IsActive {
		return "", fmt.Errorf("OTP not active")
	}
	
	if time.Now().After(otp.Expiry) {
		return "", fmt.Errorf("OTP expired")
	}
	
	// Remove non-digit characters
	cleaned := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, message)
	
	// Check if we have enough key material
	if otp.Position+len(cleaned) > len(otp.Key) {
		return "", fmt.Errorf("insufficient key material")
	}
	
	// Encrypt: ciphertext = (plaintext + key) mod 10
	encrypted := make([]byte, len(cleaned))
	for i, char := range cleaned {
		plainDigit := int(char - '0')
		keyDigit := otp.Key[otp.Position+i]
		cipherDigit := (plainDigit + keyDigit) % 10
		encrypted[i] = byte(cipherDigit + '0')
	}
	
	otp.Position += len(cleaned)
	return string(encrypted), nil
}

// Decrypt decrypts a ciphertext using OTP
func (otp *OTP) Decrypt(ciphertext string) (string, error) {
	if !otp.IsActive {
		return "", fmt.Errorf("OTP not active")
	}
	
	if time.Now().After(otp.Expiry) {
		return "", fmt.Errorf("OTP expired")
	}
	
	// Remove non-digit characters
	cleaned := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, ciphertext)
	
	// Check if we have enough key material
	if otp.Position+len(cleaned) > len(otp.Key) {
		return "", fmt.Errorf("insufficient key material")
	}
	
	// Decrypt: plaintext = (ciphertext - key) mod 10
	decrypted := make([]byte, len(cleaned))
	for i, char := range cleaned {
		cipherDigit := int(char - '0')
		keyDigit := otp.Key[otp.Position+i]
		plainDigit := (cipherDigit - keyDigit + 10) % 10
		decrypted[i] = byte(plainDigit + '0')
	}
	
	otp.Position += len(cleaned)
	return string(decrypted), nil
}

// GetRemainingKeyMaterial returns the remaining key material
func (otp *OTP) GetRemainingKeyMaterial() int {
	return len(otp.Key) - otp.Position
}

// GetUsagePercentage returns the percentage of key material used
func (otp *OTP) GetUsagePercentage() float64 {
	return float64(otp.Position) / float64(len(otp.Key)) * 100.0
}

// Invalidate invalidates the OTP
func (otp *OTP) Invalidate(reason string) {
	otp.IsActive = false
}

// GenerateStandardCodebook generates a standard military codebook
func GenerateStandardCodebook(bookID, issuedTo, issuedBy string) *Codebook {
	cb := NewCodebook(bookID, CodebookNumerical, issuedTo, issuedBy)
	
	// Add standard military codes
	// Personnel
	cb.AddEntry("001", "SOLDIERS", "PERSONNEL")
	cb.AddEntry("002", "OFFICERS", "PERSONNEL")
	cb.AddEntry("003", "CIVILIANS", "PERSONNEL")
	cb.AddEntry("004", "CASUALTIES", "PERSONNEL")
	cb.AddEntry("005", "PRISONERS", "PERSONNEL")
	
	// Equipment
	cb.AddEntry("100", "WEAPONS", "EQUIPMENT")
	cb.AddEntry("101", "AMMUNITION", "EQUIPMENT")
	cb.AddEntry("102", "VEHICLES", "EQUIPMENT")
	cb.AddEntry("103", "AIRCRAFT", "EQUIPMENT")
	cb.AddEntry("104", "COMMUNICATIONS", "EQUIPMENT")
	
	// Actions
	cb.AddEntry("200", "ATTACK", "ACTION")
	cb.AddEntry("201", "DEFEND", "ACTION")
	cb.AddEntry("202", "MOVE", "ACTION")
	cb.AddEntry("203", "RECON", "ACTION")
	cb.AddEntry("204", "WITHDRAW", "ACTION")
	
	// Status
	cb.AddEntry("300", "OK", "STATUS")
	cb.AddEntry("301", "DANGER", "STATUS")
	cb.AddEntry("302", "UNKNOWN", "STATUS")
	cb.AddEntry("303", "CONFIRMED", "STATUS")
	cb.AddEntry("304", "DENIED", "STATUS")
	
	// Locations
	cb.AddEntry("400", "BASE", "LOCATION")
	cb.AddEntry("401", "FORWARD", "LOCATION")
	cb.AddEntry("402", "REAR", "LOCATION")
	cb.AddEntry("403", "AIRPORT", "LOCATION")
	cb.AddEntry("404", "PORT", "LOCATION")
	
	// Time
	cb.AddEntry("500", "NOW", "TIME")
	cb.AddEntry("501", "SOON", "TIME")
	cb.AddEntry("502", "LATER", "TIME")
	cb.AddEntry("503", "TODAY", "TIME")
	cb.AddEntry("504", "TOMORROW", "TIME")
	
	// Space-specific
	cb.AddEntry("600", "SATELLITE", "SPACE")
	cb.AddEntry("601", "ORBIT", "SPACE")
	cb.AddEntry("602", "LAUNCH", "SPACE")
	cb.AddEntry("603", "DEBRIS", "SPACE")
	cb.AddEntry("604", "INTERCEPT", "SPACE")
	
	return cb
}

// GenerateOPCODECodebook generates an OPCODE codebook
func GenerateOPCODECodebook(bookID, issuedTo, issuedBy string) *Codebook {
	cb := NewCodebook(bookID, CodebookOPCODE, issuedTo, issuedBy)
	
	// OPCODEs are 3-letter codes
	cb.AddEntry("DIM", "DIRECTION OF MOVEMENT", "OPCODE")
	cb.AddEntry("SIT", "SITUATION REPORT", "OPCODE")
	cb.AddEntry("SPOT", "SPOT REPORT", "OPCODE")
	cb.AddEntry("MED", "MEDEVAC REQUEST", "OPCODE")
	cb.AddEntry("LOG", "LOGISTICS REPORT", "OPCODE")
	cb.AddEntry("INT", "INTELLIGENCE REPORT", "OPCODE")
	cb.AddEntry("FIRE", "FIRE MISSION", "OPCODE")
	cb.AddEntry("MOV", "MOVEMENT ORDER", "OPCODE")
	cb.AddEntry("SEC", "SECURITY REPORT", "OPCODE")
	cb.AddEntry("COM", "COMMUNICATIONS REPORT", "OPCODE")
	
	return cb
}

// EncryptionLayer represents a layer of encryption
type EncryptionLayer struct {
	Type     string // CODEBOOK, OTP, DIGITAL
	IsActive bool
	Key      interface{}
}

// EncryptionStack manages multiple layers of encryption
type EncryptionStack struct {
	layers []EncryptionLayer
}

// NewEncryptionStack creates a new encryption stack
func NewEncryptionStack() *EncryptionStack {
	return &EncryptionStack{
		layers: make([]EncryptionLayer, 0),
	}
}

// AddLayer adds an encryption layer
func (es *EncryptionStack) AddLayer(layerType string, key interface{}) error {
	es.layers = append(es.layers, EncryptionLayer{
		Type:     layerType,
		IsActive: true,
		Key:      key,
	})
	return nil
}

// Encrypt encrypts a message through all active layers
func (es *EncryptionStack) Encrypt(message string) (string, error) {
	encrypted := message
	
	for _, layer := range es.layers {
		if !layer.IsActive {
			continue
		}
		
		switch layer.Type {
		case "CODEBOOK":
			if cb, ok := layer.Key.(*Codebook); ok {
				var err error
				encrypted, err = cb.Encode(encrypted)
				if err != nil {
					return "", fmt.Errorf("codebook encryption failed: %w", err)
				}
			}
		case "OTP":
			if otp, ok := layer.Key.(*OTP); ok {
				var err error
				encrypted, err = otp.Encrypt(encrypted)
				if err != nil {
					return "", fmt.Errorf("OTP encryption failed: %w", err)
				}
			}
		default:
			return "", fmt.Errorf("unknown encryption layer: %s", layer.Type)
		}
	}
	
	return encrypted, nil
}

// Decrypt decrypts a message through all active layers (in reverse order)
func (es *EncryptionStack) Decrypt(ciphertext string) (string, error) {
	decrypted := ciphertext
	
	// Reverse order for decryption
	for i := len(es.layers) - 1; i >= 0; i-- {
		layer := es.layers[i]
		if !layer.IsActive {
			continue
		}
		
		switch layer.Type {
		case "CODEBOOK":
			if cb, ok := layer.Key.(*Codebook); ok {
				var err error
				decrypted, err = cb.Decode(decrypted)
				if err != nil {
					return "", fmt.Errorf("codebook decryption failed: %w", err)
				}
			}
		case "OTP":
			if otp, ok := layer.Key.(*OTP); ok {
				var err error
				decrypted, err = otp.Decrypt(decrypted)
				if err != nil {
					return "", fmt.Errorf("OTP decryption failed: %w", err)
				}
			}
		default:
			return "", fmt.Errorf("unknown encryption layer: %s", layer.Type)
		}
	}
	
	return decrypted, nil
}

// RemoveLayer removes an encryption layer
func (es *EncryptionStack) RemoveLayer(layerType string) error {
	for i, layer := range es.layers {
		if layer.Type == layerType {
			es.layers = append(es.layers[:i], es.layers[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("layer not found: %s", layerType)
}

// GetLayerCount returns the number of encryption layers
func (es *EncryptionStack) GetLayerCount() int {
	return len(es.layers)
}

// Clear clears all encryption layers
func (es *EncryptionStack) Clear() {
	es.layers = make([]EncryptionLayer, 0)
}
