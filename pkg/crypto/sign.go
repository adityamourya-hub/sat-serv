package crypto

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Signer struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
}

type AuditEntry struct {
	Timestamp    int64  `json:"timestamp"`
	ObjectID     int    `json:"object_id"`
	EventType    string `json:"event_type"`
	Proof        string `json:"proof"`
	PublicKeyHex string `json:"public_key_hex"`
}

// NewSigner deterministically derives an Ed25519 keypair from a seed string.
// This is retained for unit tests and compatibility with existing call sites.
func NewSigner(seed string) *Signer {
	h := sha256.Sum256([]byte(seed))
	private := ed25519.NewKeyFromSeed(h[:])
	public := private.Public().(ed25519.PublicKey)
	return &Signer{privateKey: private, publicKey: public}
}

// NewPersistentSigner loads or creates an Ed25519 keypair on disk.
func NewPersistentSigner(keyDir string) (*Signer, error) {
	if keyDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("resolve home dir: %w", err)
		}
		keyDir = filepath.Join(home, ".ssis", "keys")
	}
	if err := os.MkdirAll(keyDir, 0700); err != nil {
		return nil, fmt.Errorf("create key directory: %w", err)
	}

	privatePath := filepath.Join(keyDir, "ed25519_private.key")
	publicPath := filepath.Join(keyDir, "ed25519_public.key")

	if b, err := os.ReadFile(privatePath); err == nil {
		privateBytes, err := hex.DecodeString(string(bytesTrimSpace(b)))
		if err != nil {
			return nil, fmt.Errorf("decode persisted private key: %w", err)
		}
		if len(privateBytes) != ed25519.PrivateKeySize {
			return nil, fmt.Errorf("invalid persisted private key length: %d", len(privateBytes))
		}
		private := ed25519.PrivateKey(privateBytes)
		public := private.Public().(ed25519.PublicKey)
		return &Signer{privateKey: private, publicKey: public}, nil
	}

	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate ed25519 keypair: %w", err)
	}
	if err := os.WriteFile(privatePath, []byte(hex.EncodeToString(private)), 0600); err != nil {
		return nil, fmt.Errorf("write private key: %w", err)
	}
	if err := os.WriteFile(publicPath, []byte(hex.EncodeToString(public)), 0644); err != nil {
		return nil, fmt.Errorf("write public key: %w", err)
	}
	return &Signer{privateKey: private, publicKey: public}, nil
}

func NewSignerFromEnv(envKey string) (*Signer, error) {
	key := strings.TrimSpace(os.Getenv(envKey))
	if key == "" {
		return nil, fmt.Errorf("CRITICAL: environment variable %s not set - SSIS must run in Zero-Trust mode", envKey)
	}

	raw, err := hex.DecodeString(key)
	if err != nil {
		return nil, fmt.Errorf("invalid env key format for %s: %w", envKey, err)
	}
	var private ed25519.PrivateKey
	switch len(raw) {
	case ed25519.SeedSize:
		private = ed25519.NewKeyFromSeed(raw)
	case ed25519.PrivateKeySize:
		// Normalize 64-byte payloads by deriving a coherent keypair from the seed half.
		private = ed25519.NewKeyFromSeed(raw[:ed25519.SeedSize])
	default:
		return nil, fmt.Errorf("invalid env private key length for %s: %d (expected 32 or 64)", envKey, len(raw))
	}
	public := private.Public().(ed25519.PublicKey)
	return &Signer{privateKey: private, publicKey: public}, nil
}

func (s *Signer) PublicKeyHex() string {
	return hex.EncodeToString(s.publicKey)
}

func (s *Signer) Sign(data []byte) string {
	sig := ed25519.Sign(s.privateKey, data)
	return hex.EncodeToString(sig)
}

func (s *Signer) Verify(data []byte, signature string) bool {
	sigBytes, err := hex.DecodeString(signature)
	if err != nil {
		return false
	}
	return ed25519.Verify(s.publicKey, data, sigBytes)
}

func (s *Signer) SignJSON(v interface{}) (string, string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", "", err
	}
	sig := s.Sign(data)
	return string(data), sig, nil
}

func (s *Signer) VerifyJSON(jsonData string, signature string) bool {
	return s.Verify([]byte(jsonData), signature)
}

func (s *Signer) GenerateProof(objID int, eventType string, timestamp int64) string {
	payload := fmt.Sprintf("%d:%s:%d", objID, eventType, timestamp)
	return s.Sign([]byte(payload))
}

func (s *Signer) MakeAuditEntry(objID int, eventType string) AuditEntry {
	ts := time.Now().Unix()
	return AuditEntry{
		Timestamp:    ts,
		ObjectID:     objID,
		EventType:    eventType,
		Proof:        s.GenerateProof(objID, eventType, ts),
		PublicKeyHex: s.PublicKeyHex(),
	}
}

func bytesTrimSpace(b []byte) []byte {
	start := 0
	for start < len(b) && (b[start] == ' ' || b[start] == '\n' || b[start] == '\t' || b[start] == '\r') {
		start++
	}
	end := len(b)
	for end > start && (b[end-1] == ' ' || b[end-1] == '\n' || b[end-1] == '\t' || b[end-1] == '\r') {
		end--
	}
	return b[start:end]
}
