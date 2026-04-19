package cryptography

import (
	"bytes"
	"testing"
)

/**
 * Merkle Tree Test Suite
 */

func TestMerkleEmptyTree(t *testing.T) {
	tree := NewTree()
	root := tree.Root()

	// Empty tree should return zero hash
	var zero [32]byte
	if root != zero {
		t.Error("Empty tree should return zero hash")
	}
}

func TestMerkleSingleLeaf(t *testing.T) {
	tree := NewTree()
	data := []byte("test frame data")
	tree.AddLeaf(data)

	root := tree.Root()

	// Single leaf = root is hash of leaf
	h := tree.hasher
	h.Reset()
	h.Write(data)
	expected := h.Sum(nil)

	if !bytes.Equal(root[:], expected) {
		t.Error("Single leaf root should equal leaf hash")
	}
}

func TestMerkleMultipleLeaves(t *testing.T) {
	tree := NewTree()
	frames := [][]byte{
		[]byte("frame 1"),
		[]byte("frame 2"),
		[]byte("frame 3"),
		[]byte("frame 4"),
	}

	for _, f := range frames {
		tree.AddLeaf(f)
	}

	root := tree.Root()

	// Root should be non-zero and deterministic
	var zero [32]byte
	if root == zero {
		t.Error("Multi-leaf tree should have non-zero root")
	}

	// Same frames should produce same root
	tree2 := NewTree()
	for _, f := range frames {
		tree2.AddLeaf(f)
	}
	root2 := tree2.Root()

	if root != root2 {
		t.Error("Same frames should produce same root")
	}

	// Different frames should produce different root
	tree3 := NewTree()
	tree3.AddLeaf([]byte("different frame"))
	root3 := tree3.Root()

	if root == root3 {
		t.Error("Different frames should produce different root")
	}
}

func TestMerkleOddNumberOfLeaves(t *testing.T) {
	tree := NewTree()
	frames := [][]byte{
		[]byte("frame 1"),
		[]byte("frame 2"),
		[]byte("frame 3"), // Odd - should be promoted
	}

	for _, f := range frames {
		tree.AddLeaf(f)
	}

	root := tree.Root()

	var zero [32]byte
	if root == zero {
		t.Error("Odd-leaf tree should have non-zero root")
	}
}

func TestMerkleReset(t *testing.T) {
	tree := NewTree()
	tree.AddLeaf([]byte("old data"))
	oldRoot := tree.Root()

	tree.Reset()
	tree.AddLeaf([]byte("new data"))
	newRoot := tree.Root()

	if oldRoot == newRoot {
		t.Error("Reset should clear previous leaves")
	}
}

func BenchmarkMerkleAddLeaf(b *testing.B) {
	frame := make([]byte, 1024) // 1KB radar frame
	for i := range frame {
		frame[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree := NewTree()
		for j := 0; j < 100; j++ { // 100 frames per batch
			tree.AddLeaf(frame)
		}
		_ = tree.Root()
	}
}

func BenchmarkMerkleRoot1024(b *testing.B) {
	// Benchmark with 1024 leaves (typical batch size)
	frames := make([][]byte, 1024)
	for i := range frames {
		frames[i] = make([]byte, 1024)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree := NewTree()
		for _, f := range frames {
			tree.AddLeaf(f)
		}
		_ = tree.Root()
	}
}

/**
 * Proof Generation and Verification Tests
 */

func TestMerkleProofGeneration(t *testing.T) {
	tree := NewTree()
	frames := [][]byte{
		[]byte("frame 1"),
		[]byte("frame 2"),
		[]byte("frame 3"),
		[]byte("frame 4"),
	}

	for _, f := range frames {
		tree.AddLeaf(f)
	}

	// Generate proof for leaf 1
	proof, err := tree.GenerateProof(1)
	if err != nil {
		t.Fatalf("Failed to generate proof: %v", err)
	}

	// Proof should have correct leaf hash
	if proof.LeafIndex != 1 {
		t.Error("Proof should have correct leaf index")
	}

	// Proof should have siblings
	if len(proof.Siblings) == 0 {
		t.Error("Proof should have at least one sibling")
	}
}

func TestMerkleProofVerification(t *testing.T) {
	tree := NewTree()
	frames := [][]byte{
		[]byte("frame 1"),
		[]byte("frame 2"),
		[]byte("frame 3"),
		[]byte("frame 4"),
	}

	for _, f := range frames {
		tree.AddLeaf(f)
	}

	root := tree.Root()

	// Test all leaves
	for i := range frames {
		proof, err := tree.GenerateProof(i)
		if err != nil {
			t.Fatalf("Failed to generate proof for leaf %d: %v", i, err)
		}

		if !VerifyProof(root, proof) {
			t.Errorf("Proof verification failed for leaf %d", i)
		}
	}
}

func TestMerkleProofInvalidIndex(t *testing.T) {
	tree := NewTree()
	tree.AddLeaf([]byte("only leaf"))

	_, err := tree.GenerateProof(-1)
	if err == nil {
		t.Error("Should error on negative index")
	}

	_, err = tree.GenerateProof(1)
	if err == nil {
		t.Error("Should error on out-of-bounds index")
	}

	_, err = tree.GenerateProof(10)
	if err == nil {
		t.Error("Should error on out-of-bounds index")
	}
}

func TestMerkleProofTampering(t *testing.T) {
	tree := NewTree()
	frames := [][]byte{
		[]byte("frame 1"),
		[]byte("frame 2"),
		[]byte("frame 3"),
		[]byte("frame 4"),
	}

	for _, f := range frames {
		tree.AddLeaf(f)
	}

	root := tree.Root()
	proof, _ := tree.GenerateProof(0)

	// Tamper with leaf hash
	proof.LeafHash[0] ^= 0xFF
	if VerifyProof(root, proof) {
		t.Error("Verification should fail with tampered leaf")
	}
}

func TestMerkleProofSingleLeaf(t *testing.T) {
	tree := NewTree()
	tree.AddLeaf([]byte("solo"))

	proof, err := tree.GenerateProof(0)
	if err != nil {
		t.Fatalf("Failed to generate proof for single leaf: %v", err)
	}

	// Single leaf has no siblings
	if len(proof.Siblings) != 0 {
		t.Error("Single leaf proof should have no siblings")
	}

	root := tree.Root()
	if !VerifyProof(root, proof) {
		t.Error("Single leaf proof should verify")
	}
}
