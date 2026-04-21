// cmd/aditi/root.go - Cobra Root Command for ADITI 4.0
//
// This file initializes the root Cobra command and sets up Viper configuration.
// The root command is the entry point for all ADITI 4.0 CLI operations.
//
// Air-Gap Config Mandate:
// Viper allows locking down a config.yaml file on ruggedized servers,
// merging with environment variables, and binding directly to Cobra flags.
// This guarantees Configuration Determinism for air-gapped deployments.

package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"ssis/pkg/config"
)

var cfgFile string

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "aditi",
	Short: "Axiom-Zero Hamiltonian-Gated Sovereign SDA Engine",
	Long: `ADITI 4.0 - IAF Axiom-Zero Hamiltonian-Gated Sovereign Space Domain Awareness Engine

The Command Center for sovereign SDA operations, providing direct control over:
  - Hybrid orbital propagation (Forest-Ruth + RKF45)
  - SINDy discovery filter for zero-day maneuver detection
  - Adversarial telescope control with <50ms settle-time
  - Catalog management with Hamiltonian-Gated states
  - PAX thermal governor and execution tier management
  - Hamiltonian integrity verification and audit

Configuration:
  Configuration is loaded from config.yaml (or specified via --config).
  Environment variables (ADITI_*) override config file values.
  Command-line flags override both config and environment variables.

Examples:
  aditi catalog import celestrak.tle
  aditi propagate --target 25544 --regime leo
  aditi track --target HGV-001 --mode discovery
  aditi pax status`,
	Version: "4.0.0",
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	// Initialize Viper before running commands
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./config.yaml, /etc/aditi/config.yaml, or $HOME/.aditi/config.yaml)")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "verbose output")
	rootCmd.PersistentFlags().BoolP("quiet", "q", false, "suppress non-error output")
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if err := config.LoadConfig(cfgFile); err != nil {
		fmt.Fprintf(os.Stderr, "[FATAL] Failed to load configuration: %v\n", err)
		os.Exit(1)
	}
}
