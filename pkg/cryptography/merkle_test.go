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
