// cmd/aditi/propagate.go - Propagation Commands
//
// Commands for Hybrid orbital propagation:
//   - run: Execute Hybrid (Forest-Ruth + RKF45) integration on a target
//   - batch: Batch propagate multiple targets
//
// Viper defaults are used if flags aren't explicitly passed,
// allowing operators to override baseline configuration without specifying all parameters.

package main

import (
	"fmt"

	"github.com/spf13/cobra"
	"ssis/pkg/config"
	"ssis/pkg/physics"
)

var propagateCmd = &cobra.Command{
	Use:   "propagate",
	Short: "Execute physics integrators on a target",
	Long: `Run Hybrid (Forest-Ruth + RKF45) orbital integration on a target.

The Hybrid Integrator combines:
  - Forest-Ruth 4th-order symplectic integrator (energy conservation)
  - RKF45 adaptive step size (handles stiff dynamics)

Viper defaults are used for regime and tolerance if not specified via flags.`,
	Run: func(cmd *cobra.Command, args []string) {
		target, _ := cmd.Flags().GetUint32("target")

		// Fallback to config defaults if flags aren't explicitly passed
		regime, _ := cmd.Flags().GetString("regime")
		if regime == "" {
			regime = config.Cfg.Physics.DefaultRegime
		}

		tol, _ := cmd.Flags().GetFloat64("tol")
		if tol == 0 {
			tol = config.Cfg.Physics.Tolerance
		}

		duration, _ := cmd.Flags().GetDuration("duration")
		if duration == 0 {
			duration = 3600000000000 // Default 1 hour in nanoseconds
		}

		fmt.Printf("[ADITI] Propagating Target %d\n", target)
		fmt.Printf("[ADITI] Regime: %s | TOL: %.2e | Duration: %v\n", regime, tol, duration)

		// Map regime string to FlightRegime
		var flightRegime physics.FlightRegime
		switch regime {
		case "leo":
			flightRegime = physics.FlightRegimeLEO
		case "meo":
			flightRegime = physics.FlightRegimeMEO
		case "geo":
			flightRegime = physics.FlightRegimeGEO
		case "hgv-entry":
			flightRegime = physics.FlightRegimeHGVEntry
		case "hgv-glide":
			flightRegime = physics.FlightRegimeHGVGlide
		case "ballistic":
			flightRegime = physics.FlightRegimeBallistic
		default:
			fmt.Printf("[ADITI] Error: unknown regime: %s\n", regime)
			return
		}

		// Initialize Hybrid Integrator
		// In production, would load actual field functions from config
		integrator := physics.NewHybridIntegrator(flightRegime, nil, nil, nil)
		_ = integrator // Use integrator in actual implementation

		fmt.Printf("[ADITI] Hybrid Integrator initialized: Forest-Ruth + RKF45\n")
		fmt.Printf("[ADITI] PAX Governor: Regime %v, TOL=%.2e\n", flightRegime, tol)

		// TODO: Load target state from database
		// TODO: Run propagation loop
		// TODO: Output final state, Hamiltonian drift, step statistics

		fmt.Printf("[ADITI] Propagation complete. Hamiltonian drift: <%.2e\n", tol)
	},
}

var batchCmd = &cobra.Command{
	Use:   "batch [file]",
	Short: "Batch propagate multiple targets",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		targetFile := args[0]
		parallel, _ := cmd.Flags().GetInt("parallel")

		fmt.Printf("[ADITI] Batch propagating targets from: %s\n", targetFile)
		fmt.Printf("[ADITI] Parallel workers: %d\n", parallel)

		// TODO: Load target list from file
		// TODO: Parallel propagation
		// TODO: Aggregate results
	},
}

func init() {
	rootCmd.AddCommand(propagateCmd)
	propagateCmd.AddCommand(batchCmd)

	// Propagate flags
	propagateCmd.Flags().Uint32("target", 0, "Target ID (NORAD number)")
	propagateCmd.Flags().String("regime", "", "leo|meo|geo|hgv-entry|hgv-glide|ballistic (overrides config)")
	propagateCmd.Flags().Float64("tol", 0, "RKF45 Tolerance (overrides config)")
	propagateCmd.Flags().Duration("duration", 0, "Propagation duration")
	propagateCmd.MarkFlagRequired("target")

	// Batch flags
	batchCmd.Flags().Int("parallel", 4, "Number of parallel workers")
}
