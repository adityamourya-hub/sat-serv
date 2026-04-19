# Sovereign Audit Bridge - Cryptographic Implementation Research

## Overview
This document researches the implementation of cryptographic components for the IDEX Sovereign Audit Bridge, providing tamper-evident attestation of space domain awareness (SDA) events.

## 1. Core Cryptographic Primitives

### 1.1 Hash Functions (Merkle Rooting)
**Recommendation:** BLAKE3 or SHA-256

| Algorithm | Speed | Security | Use Case |
|-----------|-------|----------|----------|
| BLAKE3 | ~3x faster than SHA-256 | 256-bit | Merkle tree construction, frame hashing |
| SHA-256 | Standard | 256-bit | Interoperability with legacy systems |

**Go Implementation:**
```go
import "golang.org/x/crypto/blake3"
// or
import "crypto/sha256"
```

**Merkle Tree Design:**
- Leaf nodes: Individual radar frame hashes
- Internal nodes: `Hash(left || right)`
- Root: Final 32-byte attestation anchor
- Tree depth: Adjustable based on frame batch size (typically 10-14 for 1K-16K frames)

### 1.2 Digital Signatures (Ed25519)
**Recommendation:** Ed25519 (Curve25519)

**Rationale:**
- Fast signature generation (~50μs per signature)
- Compact signatures (64 bytes)
- Side-channel resistant
- Widely supported (libsodium, Go standard library)

**Go Implementation:**
```go
import "crypto/ed25519"

// Key generation (in AUM-1 hardware, this uses PUF)
privateKey := ed25519.NewKeyFromSeed(seed) // 32-byte seed
publicKey := privateKey.Public()

// Signing
signature := ed25519.Sign(privateKey, message)

// Verification
valid := ed25519.Verify(publicKey, message, signature)
```

**Key Management (Shadow Sovereign Mode):**
- Seed derived from: `HKDF(environment_entropy, "idex_sovereign_v1")`
- Environment entropy: `/dev/urandom`, TPM if available, or HSM
- **Warning:** Extractable in software mode. Hardware PUF (AUM-1) required for production.

### 1.3 Key Derivation (HKDF)
**Purpose:** Derive multiple keys from single seed

```go
import "golang.org/x/crypto/hkdf"

// Master seed from PUF or environment
masterSeed := []byte{...} // 32 bytes

// Derive signing key
hkdfReader := hkdf.New(sha256.New, masterSeed, nil, []byte("signing_key_v1"))
signingKeySeed := make([]byte, 32)
io.ReadFull(hkdfReader, signingKeySeed)
```

## 2. Attestation Datagram Schema

### 2.1 Binary Encoding (CBOR - Concise Binary Object Representation)
**Why CBOR over JSON:**
- Compact (20-50% smaller than JSON)
- Deterministic encoding (canonical forms)
- Fast parsing
- IETF standard (RFC 8949)

**Go Implementation:**
```go
import "github.com/fxamacker/cbor/v2"

// Encode
data, err := cbor.Marshal(attestation)

// Decode
var att AttestationDatagram
err := cbor.Unmarshal(data, &att)
```

### 2.2 Schema Definition

```go
// AttestationDatagram - the "Proof of Aggression"
type AttestationDatagram struct {
    // Header
    Version     uint8     // Schema version (1)
    Timestamp   int64     // Unix nanoseconds
    AuditorID   [16]byte  // Unique auditor identifier
    
    // Event Data
    ObjectID    string    // Satellite/debris NORAD ID
    EventType   uint8     // 0x01 = MANEUVER_DETECTED, 0x02 = ANOMALY
    
    // Mathematical Proof
    MerkleRoot  [32]byte  // Hash of raw sensor frames
    ResidualRMS float64   // Root mean square of orbit residuals
    Confidence  float64   // Bayesian detection score (0-1)
    Threshold   float64   // Detection threshold used
    
    // Context
    TCA         float64   // Time of Closest Approach (Julian Date)
    Position    [3]float64 // ECI position at event (km)
    Velocity    [3]float64 // ECI velocity at event (km/s)
    
    // Signature (Ed25519)
    Signature   [64]byte  // Signature of serialized header+event+proof
}
```

**Serialized Size:** ~200-300 bytes (compact for blockchain storage)

## 3. Merkle Tree Implementation

### 3.1 Incremental Tree (For Streaming Radar Frames)

```go
package merkle

type IncrementalTree struct {
    leaves   [][32]byte
    hasher   hash.Hash
}

func NewIncrementalTree() *IncrementalTree {
    return &IncrementalTree{
        hasher: sha256.New(), // or blake3.New()
    }
}

func (t *IncrementalTree) AddLeaf(data []byte) {
    t.hasher.Reset()
    t.hasher.Write(data)
    var leaf [32]byte
    copy(leaf[:], t.hasher.Sum(nil))
    t.leaves = append(t.leaves, leaf)
}

func (t *IncrementalTree) Root() [32]byte {
    return computeRoot(t.leaves)
}

// Recursive root computation
func computeRoot(leaves [][32]byte) [32]byte {
    if len(leaves) == 0 {
        return [32]byte{}
    }
    if len(leaves) == 1 {
        return leaves[0]
    }
    
    // Pairwise hash
    var nextLevel [][32]byte
    for i := 0; i < len(leaves); i += 2 {
        if i+1 < len(leaves) {
            // Hash pair
            var combined [64]byte
            copy(combined[:32], leaves[i][:])
            copy(combined[32:], leaves[i+1][:])
            nextLevel = append(nextLevel, hash64(combined))
        } else {
            // Odd leaf - promote
            nextLevel = append(nextLevel, leaves[i])
        }
    }
    return computeRoot(nextLevel)
}
```

### 3.2 Proof Generation (For Selective Disclosure)

```go
type MerkleProof struct {
    LeafIndex int
    LeafHash  [32]byte
    Siblings  [][32]byte // Path from leaf to root
}

func (t *IncrementalTree) GenerateProof(index int) MerkleProof {
    // Implementation: return path of sibling hashes
    // Allows verification: leaf exists in tree without revealing other leaves
}
```

## 4. Sovereign Auditor Interface

### 4.1 Core Interface

```go
package sovereign

// SovereignAuditor provides tamper-evident attestation
type SovereignAuditor interface {
    // HashIngress builds Merkle root from raw sensor frames
    HashIngress(frames [][]byte) ([32]byte, error)
    
    // AttestManeuver generates signed proof of detection
    AttestManeuver(
        detection ManeuverDetection,
        merkleRoot [32]byte,
    ) (AttestationDatagram, error)
    
    // VerifyAttestation checks external attestation
    VerifyAttestation(datagram AttestationDatagram) (bool, error)
    
    // PublicKey returns auditor's verification key
    PublicKey() [32]byte
}

// ManeuverDetection - output from physics engine
type ManeuverDetection struct {
    ObjectID      string
    Timestamp     time.Time
    ResidualRMS   float64
    Confidence    float64
    Threshold     float64
    Position      [3]float64
    Velocity      [3]float64
    TCAs          []float64 // Associated conjunction times
}
```

### 4.2 Software Implementation (Shadow Sovereign)

```go
package sovereign

import (
    "crypto/ed25519"
    "crypto/rand"
    "encoding/binary"
    "time"
)

type SoftwareSovereignAuditor struct {
    privateKey ed25519.PrivateKey
    publicKey  ed25519.PublicKey
    auditorID  [16]byte
}

// NewSoftwareSovereignAuditor creates a software-based auditor
// WARNING: For development/testing only. Use AUM-1 hardware for production.
func NewSoftwareSovereignAuditor() (*SoftwareSovereignAuditor, error) {
    // Generate random seed (in AUM-1, this comes from PUF)
    seed := make([]byte, 32)
    if _, err := rand.Read(seed); err != nil {
        return nil, err
    }
    
    privateKey := ed25519.NewKeyFromSeed(seed)
    
    // Derive auditor ID from public key hash
    var auditorID [16]byte
    copy(auditorID[:], privateKey.Public().(ed25519.PublicKey)[:16])
    
    return &SoftwareSovereignAuditor{
        privateKey: privateKey,
        publicKey:  privateKey.Public().(ed25519.PublicKey),
        auditorID:  auditorID,
    }, nil
}

func (s *SoftwareSovereignAuditor) HashIngress(frames [][]byte) ([32]byte, error) {
    tree := NewIncrementalTree()
    for _, frame := range frames {
        tree.AddLeaf(frame)
    }
    return tree.Root(), nil
}

func (s *SoftwareSovereignAuditor) AttestManeuver(
    detection ManeuverDetection,
    merkleRoot [32]byte,
) (AttestationDatagram, error) {
    
    att := AttestationDatagram{
        Version:     1,
        Timestamp:   time.Now().UnixNano(),
        AuditorID:   s.auditorID,
        ObjectID:    detection.ObjectID,
        EventType:   0x01, // MANEUVER_DETECTED
        MerkleRoot:  merkleRoot,
        ResidualRMS: detection.ResidualRMS,
        Confidence:  detection.Confidence,
        Threshold:   detection.Threshold,
        TCA:         detection.TCAs[0], // Primary TCA
        Position:    detection.Position,
        Velocity:    detection.Velocity,
    }
    
    // Serialize for signing (exclude signature field)
    message := att.serializeForSigning()
    
    // Sign with Ed25519
    sig := ed25519.Sign(s.privateKey, message)
    copy(att.Signature[:], sig)
    
    return att, nil
}

func (s *SoftwareSovereignAuditor) VerifyAttestation(att AttestationDatagram) (bool, error) {
    // Serialize same way as during signing
    message := att.serializeForSigning()
    
    // Verify signature
    sig := att.Signature[:]
    return ed25519.Verify(s.publicKey, message, sig), nil
}

func (s *SoftwareSovereignAuditor) PublicKey() [32]byte {
    var pk [32]byte
    copy(pk[:], s.publicKey)
    return pk
}

// serializeForSigning creates deterministic encoding of attestation fields
func (a AttestationDatagram) serializeForSigning() []byte {
    // Use CBOR or custom binary encoding (must be deterministic)
    // Exclude Signature field from serialization
    buf := make([]byte, 0, 200)
    
    buf = append(buf, a.Version)
    buf = binary.LittleEndian.AppendUint64(buf, uint64(a.Timestamp))
    buf = append(buf, a.AuditorID[:]...)
    buf = append(buf, []byte(a.ObjectID)...)
    buf = append(buf, a.EventType)
    buf = append(buf, a.MerkleRoot[:]...)
    buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.ResidualRMS))
    buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Confidence))
    buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Threshold))
    
    return buf
}
```

## 5. Blockchain Integration Options

### 5.1 Option A: Solana Devnet (Public)

**Pros:**
- Fast finality (400ms)
- Low cost (~$0.00025 per transaction)
- Existing 9Ganga program structure

**Cons:**
- Public ledger (sensitive SDA data exposed)
- External dependency (RPC availability)
- Not defense-grade for kinetic attestation

**Go SDK:**
```go
import "github.com/gagliardetto/solana-go"

// Submit attestation
client := rpc.New("https://api.devnet.solana.com")
tx, err := client.SendTransaction(ctx, attestationData)
```

### 5.2 Option B: Private Sovereign Ledger (Recommended)

**Architecture:**
- Permissioned blockchain (Hyperledger Fabric or R3 Corda)
- IAF-controlled validator set (5-7 nodes)
- Air-gapped deployment option
- BFT consensus (PBFT or Raft)

**Go Integration:**
- Fabric: `github.com/hyperledger/fabric-sdk-go`
- Corda: REST API via `net/http`

**Attestation Storage:**
```go
// Store attestation hash on-chain, data off-chain
type LedgerEntry struct {
    AttestationHash [32]byte    // Blake3 hash of full attestation
    Timestamp       int64       // Block timestamp
    AuditorPubKey   [32]byte    // Ed25519 verifying key
    Signature       [64]byte    // Auditor's signature of hash
    // Full attestation stored in encrypted object storage
}
```

### 5.3 Option C: Merkle Log (Blockchain-Optional)

**Design:**
- Append-only signed log (Trillian-style transparency)
- Periodic batch submission to blockchain (if required)
- Local verification without external dependency

```go
type SignedLog struct {
    Entries   []AttestationDatagram
    TreeRoot  [32]byte
    Signature [64]byte // Signed tree root
    Timestamp int64
}
```

## 6. Security Considerations

### 6.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Key extraction (software) | AUM-1 PUF hardware in production |
| Replay attacks | Timestamp + unique ObjectID per attestation |
| Frame tampering | Merkle root commits to all frames |
| Auditor impersonation | Ed25519 signature verification |
| Log tampering | Blockchain finality or signed Merkle tree |

### 6.2 Shadow Sovereign Mode Limitations

**Current (Software):**
- Private key in memory (extractable via debuggers)
- No hardware-bound identity
- Suitable for development, demos, non-kinetic attestation

**Production (AUM-1):**
- PUF-derived keys (unclonable, device-bound)
- Secure enclave for signing operations
- Tamper-evident packaging

### 6.3 Key Rotation

```go
// Hierarchical deterministic keys for rotation
// Master key -> Season keys -> Event keys

func (s *SoftwareSovereignAuditor) DeriveSeasonKey(season uint32) ed25519.PrivateKey {
    hkdfReader := hkdf.New(sha256.New, s.masterSeed, nil, 
        []byte(fmt.Sprintf("season_%d", season)))
    seasonSeed := make([]byte, 32)
    io.ReadFull(hkdfReader, seasonSeed)
    return ed25519.NewKeyFromSeed(seasonSeed)
}
```

## 7. Implementation Roadmap

### Phase 1: Core Cryptography (Week 1)
- [ ] Implement `pkg/cryptography/merkle/` tree
- [ ] Implement `pkg/cryptography/ed25519/` wrappers
- [ ] Implement `AttestationDatagram` with CBOR encoding
- [ ] Unit tests: signature verification, Merkle proofs

### Phase 2: Sovereign Auditor (Week 2)
- [ ] Implement `SoftwareSovereignAuditor` interface
- [ ] Integrate with `maneuver.go` detection pipeline
- [ ] Frame buffering and Merkle root construction
- [ ] Shadow mode operational tests

### Phase 3: Blockchain Bridge (Week 3)
- [ ] Implement ledger client interface
- [ ] Solana Devnet integration (if public acceptable)
- [ ] OR private ledger integration (if defense-grade required)
- [ ] Attestation submission and verification pipeline

### Phase 4: Hardware Preparation (Week 4)
- [ ] Define AUM-1 integration interface
- [ ] PUF key extraction protocol
- [ ] Secure boot attestation
- [ ] Hardware-in-the-loop testing

## 8. References

1. **BLAKE3:** https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf
2. **Ed25519:** https://ed25519.cr.yp.to/ed25519-20110926.pdf (Bernstein et al.)
3. **CBOR:** RFC 8949 - Concise Binary Object Representation
4. **Merkle Trees:** RFC 6962 - Certificate Transparency
5. **HKDF:** RFC 5869 - HMAC-based Extract-and-Expand Key Derivation Function
6. **Solana Programs:** https://docs.solana.com/developing/programming-model/overview
7. **Hyperledger Fabric:** https://hyperledger-fabric.readthedocs.io/

## 9. Go Dependencies

```bash
go get golang.org/x/crypto/blake3
go get golang.org/x/crypto/ed25519
go get github.com/fxamacker/cbor/v2
go get golang.org/x/crypto/hkdf

# Optional - blockchain
# go get github.com/gagliardetto/solana-go  # Solana
# go get github.com/hyperledger/fabric-sdk-go # Fabric
```

---

**Status:** Research complete. Ready for implementation upon priority confirmation.
**Security Note:** Shadow Sovereign mode is for development only. Production requires AUM-1 hardware for kinetic attestation.
