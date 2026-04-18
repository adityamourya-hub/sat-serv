package orbital

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

/**
 * DistillSovereignLaw invokes the Axiom-Zero Symbolic Oracle to extract 
 * human-verifiable physical laws from the learned rKAN manifold.
 * 
 * This enables the 'Discovery-Operational Bridge' by bridging JAX-native 
 * law distillation into the Go mission substrate.
 *
 * @param weightsPath: Path to the axiom_v3_weights.json file.
 * @return: LaTeX representation of the discovered Hamiltonian.
 */
func DistillSovereignLaw(weightsPath string) (string, error) {
	fmt.Printf("DEBUG: Triggering Symbolic Oracle distillation for %s\n", weightsPath)
	
	// Ensure the weights file exists
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		return "", fmt.Errorf("weights file not found: %s", weightsPath)
	}

	// Construct the command
	// We use python3 and ensure the PYTHONPATH includes the ai/v3 directory
	cmd := exec.Command("python3", "ai/v3/hamiltonian_discovery.py", "--weights", weightsPath)
	
	// Set execution directory to repo root
	if _, err := os.Stat("ai/v3/hamiltonian_discovery.py"); os.IsNotExist(err) {
		// Attempt to fallback to parent-relative path if run from a package subdir
		cmd.Args[1] = "../../ai/v3/hamiltonian_discovery.py"
	}
	
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Execute
	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("oracle execution failed: %v (stderr: %s)", err, stderr.String())
	}

	// Process output
	latexH := strings.TrimSpace(stdout.String())
	if latexH == "" {
		return "", fmt.Errorf("oracle returned empty law")
	}

	return latexH, nil
}

/**
 * GenerateDiscoveryReport generates a full characterization report including 
 * LaTeX Hamiltonians and system residuals.
 */
func GenerateDiscoveryReport(weightsPath, outputPath string) error {
	law, err := DistillSovereignLaw(weightsPath)
	if err != nil {
		return err
	}

	report := fmt.Sprintf("# SSIS Discovery Report\n\n## Discovered Hamiltonian\n\n```latex\n%s\n```\n", law)
	
	return os.WriteFile(outputPath, []byte(report), 0644)
}
