package assistant

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"ssis/pkg/cryptography"
)

const BundleManifestFile = "bundle_manifest.sig.json"

type bundleFileHash struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

type bundleManifestPayload struct {
	BundleDir    string           `json:"bundle_dir"`
	CreatedUTC   string           `json:"created_utc"`
	ModelPath    string           `json:"model_path"`
	ModelSize    int64            `json:"model_size_bytes"`
	ModelSHA256  string           `json:"model_sha256"`
	RequiredOK   bool             `json:"required_ok"`
	TrackedFiles []bundleFileHash `json:"tracked_files"`
}

type bundleManifest struct {
	Payload      bundleManifestPayload `json:"payload"`
	Signature    string                `json:"signature"`
	PublicKeyHex string                `json:"public_key_hex"`
}

func GenerateBundleManifest(bundleDir string) (string, error) {
	bundleDir = strings.TrimSpace(bundleDir)
	if bundleDir == "" {
		return "", fmt.Errorf("bundle directory is required")
	}
	absDir, err := filepath.Abs(bundleDir)
	if err != nil {
		return "", fmt.Errorf("resolve bundle directory: %w", err)
	}
	if err := validateRequiredBundleFiles(absDir); err != nil {
		return "", err
	}

	modelPath, modelSize, modelHash, err := resolveModelArtifact(absDir)
	if err != nil {
		return "", err
	}

	tracked, err := collectTrackedFiles(absDir)
	if err != nil {
		return "", err
	}
	payload := bundleManifestPayload{
		BundleDir:    absDir,
		CreatedUTC:   time.Now().UTC().Format(time.RFC3339),
		ModelPath:    modelPath,
		ModelSize:    modelSize,
		ModelSHA256:  modelHash,
		RequiredOK:   true,
		TrackedFiles: tracked,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal manifest payload: %w", err)
	}
	signer, err := cryptography.NewSignerFromEnv("SSIS_SIGNING_KEY")
	if err != nil {
		return "", fmt.Errorf("initialize signer: %w", err)
	}

	manifest := bundleManifest{
		Payload:      payload,
		Signature:    signer.Sign(payloadBytes),
		PublicKeyHex: signer.PublicKeyHex(),
	}
	outBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal manifest: %w", err)
	}
	outPath := filepath.Join(absDir, BundleManifestFile)
	if err := os.WriteFile(outPath, outBytes, 0644); err != nil {
		return "", fmt.Errorf("write manifest: %w", err)
	}
	return outPath, nil
}

func VerifyBundleManifest(bundleDir string) error {
	bundleDir = strings.TrimSpace(bundleDir)
	if bundleDir == "" {
		return fmt.Errorf("bundle directory is required")
	}
	absDir, err := filepath.Abs(bundleDir)
	if err != nil {
		return fmt.Errorf("resolve bundle directory: %w", err)
	}
	if err := validateRequiredBundleFiles(absDir); err != nil {
		return err
	}

	manifestPath := filepath.Join(absDir, BundleManifestFile)
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}
	var m bundleManifest
	if err := json.Unmarshal(content, &m); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	payloadBytes, err := json.Marshal(m.Payload)
	if err != nil {
		return fmt.Errorf("marshal manifest payload: %w", err)
	}
	pub, err := hex.DecodeString(strings.TrimSpace(m.PublicKeyHex))
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return fmt.Errorf("manifest public key invalid")
	}
	sig, err := hex.DecodeString(strings.TrimSpace(m.Signature))
	if err != nil {
		return fmt.Errorf("manifest signature invalid encoding")
	}
	if !ed25519.Verify(ed25519.PublicKey(pub), payloadBytes, sig) {
		return fmt.Errorf("manifest signature verification failed")
	}

	if len(m.Payload.TrackedFiles) == 0 {
		return fmt.Errorf("manifest has no tracked files")
	}
	for _, f := range m.Payload.TrackedFiles {
		if strings.TrimSpace(f.Path) == "" {
			return fmt.Errorf("manifest contains empty file path")
		}
		full := filepath.Join(absDir, filepath.FromSlash(f.Path))
		info, err := os.Stat(full)
		if err != nil {
			return fmt.Errorf("manifest file missing: %s", f.Path)
		}
		if info.Size() != f.SizeBytes {
			return fmt.Errorf("manifest size mismatch for %s", f.Path)
		}
		h, err := fileSHA256(full)
		if err != nil {
			return fmt.Errorf("hash %s: %w", f.Path, err)
		}
		if !strings.EqualFold(h, strings.TrimSpace(f.SHA256)) {
			return fmt.Errorf("manifest hash mismatch for %s", f.Path)
		}
	}

	modelPath, modelSize, modelHash, err := resolveModelArtifact(absDir)
	if err != nil {
		return err
	}
	if !strings.EqualFold(modelPath, m.Payload.ModelPath) || modelSize != m.Payload.ModelSize || !strings.EqualFold(modelHash, m.Payload.ModelSHA256) {
		return fmt.Errorf("model artifact differs from signed manifest")
	}
	if !m.Payload.RequiredOK {
		return fmt.Errorf("manifest required-file validation failed")
	}
	return nil
}

func collectTrackedFiles(bundleDir string) ([]bundleFileHash, error) {
	files := make([]bundleFileHash, 0, 8)
	add := func(rel string) error {
		full := filepath.Join(bundleDir, rel)
		info, err := os.Stat(full)
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		h, err := fileSHA256(full)
		if err != nil {
			return err
		}
		files = append(files, bundleFileHash{Path: filepath.ToSlash(rel), SizeBytes: info.Size(), SHA256: h})
		return nil
	}
	for _, rel := range []string{"model.onnx", "decoder_model.onnx", "model.onnx.data", "decoder_model.onnx.data", "tokenizer.json", "tokenizer.model", "vocab.json", "config.json", "generation_config.json", "special_tokens_map.json"} {
		if err := add(rel); err != nil {
			return nil, err
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func validateRequiredBundleFiles(bundleDir string) error {
	if _, _, _, err := resolveModelArtifact(bundleDir); err != nil {
		return err
	}
	for _, rel := range []string{"tokenizer.json", "config.json"} {
		if _, err := os.Stat(filepath.Join(bundleDir, rel)); err != nil {
			return fmt.Errorf("required bundle file missing: %s", rel)
		}
	}
	hasTokenizerModel := fileExists(filepath.Join(bundleDir, "tokenizer.model"))
	hasVocab := fileExists(filepath.Join(bundleDir, "vocab.json"))
	if !hasTokenizerModel && !hasVocab {
		return fmt.Errorf("required bundle file missing: tokenizer.model or vocab.json")
	}
	hasGeneration := fileExists(filepath.Join(bundleDir, "generation_config.json"))
	hasSpecial := fileExists(filepath.Join(bundleDir, "special_tokens_map.json"))
	if !hasGeneration && !hasSpecial {
		return fmt.Errorf("required bundle file missing: generation_config.json or special_tokens_map.json")
	}
	return nil
}

func resolveModelArtifact(bundleDir string) (string, int64, string, error) {
	candidates := []string{"model.onnx", "decoder_model.onnx"}
	for _, rel := range candidates {
		full := filepath.Join(bundleDir, rel)
		info, err := os.Stat(full)
		if err == nil {
			h, herr := fileSHA256(full)
			if herr != nil {
				return "", 0, "", fmt.Errorf("hash model artifact: %w", herr)
			}
			return filepath.ToSlash(rel), info.Size(), h, nil
		}
	}
	return "", 0, "", fmt.Errorf("required bundle file missing: model.onnx or decoder_model.onnx")
}

func fileSHA256(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
