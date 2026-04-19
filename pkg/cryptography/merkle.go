package cryptography

/**
 * Merkle Tree Implementation for Sovereign Audit Bridge
 *
 * Provides cryptographic commitment to raw sensor frames,
 * enabling tamper-evident attestation of SDA events.
 *
 * Design: Incremental tree for streaming radar frame ingestion.
 * Hash: BLAKE3 (fast, secure) or SHA-256 (interoperability)
 */

import (
	"crypto/sha256"
	"fmt"
	"hash"
)

// Tree implements an incremental Merkle tree for frame commitment
type Tree struct {
	leaves [][32]byte
	hasher hash.Hash
}

// NewTree creates a new incremental Merkle tree
func NewTree() *Tree {
	return &Tree{
		hasher: sha256.New(), // TODO: Consider BLAKE3 for speed
	}
}

// AddLeaf adds a raw frame to the tree
func (t *Tree) AddLeaf(data []byte) {
	t.hasher.Reset()
	t.hasher.Write(data)

	var leaf [32]byte
	hash := t.hasher.Sum(nil)
	copy(leaf[:], hash)

	t.leaves = append(t.leaves, leaf)
}

// Root computes the Merkle root hash
// Returns 32-byte root representing commitment to all frames
func (t *Tree) Root() [32]byte {
	if len(t.leaves) == 0 {
		return [32]byte{}
	}
	return computeRoot(t.leaves)
}

// computeRoot recursively computes tree root
func computeRoot(level [][32]byte) [32]byte {
	if len(level) == 1 {
		return level[0]
	}

	var next [][32]byte
	for i := 0; i < len(level); i += 2 {
		if i+1 < len(level) {
			// Combine pair
			combined := append(level[i][:], level[i+1][:]...)
			next = append(next, hashPair(combined))
		} else {
			// Promote odd leaf
			next = append(next, level[i])
		}
	}

	return computeRoot(next)
}

// hashPair hashes two 32-byte values together
func hashPair(data []byte) [32]byte {
	h := sha256.New()
	h.Write(data)
	var result [32]byte
	copy(result[:], h.Sum(nil))
	return result
}

// Proof represents a Merkle inclusion proof
type Proof struct {
	LeafIndex int
	LeafHash  [32]byte
	Siblings  [][32]byte // Path from leaf to root
}

// GenerateProof creates inclusion proof for a specific leaf
// Allows verification without revealing all frames
func (t *Tree) GenerateProof(index int) (Proof, error) {
	if index < 0 || index >= len(t.leaves) {
		return Proof{}, fmt.Errorf("index %d out of range [0, %d)", index, len(t.leaves))
	}

	proof := Proof{
		LeafIndex: index,
		LeafHash:  t.leaves[index],
		Siblings:  make([][32]byte, 0),
	}

	// Build sibling path
	level := make([][32]byte, len(t.leaves))
	copy(level, t.leaves)
	currentIdx := index

	for len(level) > 1 {
		var next [][32]byte
		for i := 0; i < len(level); i += 2 {
			if i+1 < len(level) {
				if i == currentIdx || i+1 == currentIdx {
					// Add sibling to proof
					if i == currentIdx {
						proof.Siblings = append(proof.Siblings, level[i+1])
					} else {
						proof.Siblings = append(proof.Siblings, level[i])
					}
				}
				combined := append(level[i][:], level[i+1][:]...)
				next = append(next, hashPair(combined))
			} else {
				next = append(next, level[i])
			}
		}
		level = next
		currentIdx /= 2
	}

	return proof, nil
}

// VerifyProof checks if leaf exists in tree with given root
func VerifyProof(root [32]byte, proof Proof) bool {
	current := proof.LeafHash
	idx := proof.LeafIndex

	for _, sibling := range proof.Siblings {
		var combined []byte
		if idx%2 == 0 {
			// Current is left, sibling is right
			combined = append(current[:], sibling[:]...)
		} else {
			// Current is right, sibling is left
			combined = append(sibling[:], current[:]...)
		}
		current = hashPair(combined)
		idx /= 2
	}

	return current == root
}

// Reset clears the tree for new batch
func (t *Tree) Reset() {
	t.leaves = t.leaves[:0]
}
