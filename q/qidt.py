#!/usr/bin/env python3
"""
Quantum Immune Digital Twin (QIDT) - CLI Entry Point

Active QP Evasion for Rad-Hard Majorana-Photonic Compute Substrate
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pathlib import Path
import sys

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from core.bdg_hamiltonian import DigitalTwinEngine, CosmicRayParameters, NanowireGeometry
from dashboard.metrics_visualizer import MetricsDashboard, SimulationConfig

app = typer.Typer(
    name="qidt",
    help="Quantum Immune Digital Twin - Active QP Evasion Simulator",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def simulate(
    duration_ms: float = typer.Option(
        10.0,
        "--duration",
        "-d",
        help="Simulation duration in milliseconds"
    ),
    dose_rate: float = typer.Option(
        100.0,
        "--dose",
        help="Radiation dose rate in mSv/hr (deep space: ~100)"
    ),
    magnetic_field: float = typer.Option(
        1.0,
        "--bfield",
        "-B",
        help="Magnetic field in Tesla"
    ),
    output: Path = typer.Option(
        Path("./outputs"),
        "--output",
        "-o",
        help="Output directory for results"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
) -> None:
    """Run single simulation and report survival metrics."""
    
    # Header
    title = Text("Quantum Immune Digital Twin", style="bold cyan")
    subtitle = Text("Active QP Evasion Engine v0.1.0", style="dim")
    console.print(Panel.fit(f"{title}\n{subtitle}"))
    
    # Configuration display
    console.print("\n[bold]Simulation Configuration:[/bold]")
    console.print(f"  • Duration: {duration_ms} ms")
    console.print(f"  • Dose Rate: {dose_rate} mSv/hr")
    console.print(f"  • Magnetic Field: {magnetic_field} T")
    console.print(f"  • Random Seed: {seed}")
    
    # Initialize engine
    console.print("\n[dim]Initializing Digital Twin Engine...[/dim]")
    
    cosmic = CosmicRayParameters(
        dose_rate_msv_hr=dose_rate,
        qp_lifetime_us=100.0
    )
    
    geo = NanowireGeometry()
    
    twin = DigitalTwinEngine(
        geometry=geo,
        cosmic_params=cosmic,
        magnetic_field_t=magnetic_field,
        random_seed=seed
    )
    
    # Run simulation
    console.print(f"[dim]Running simulation ({int(duration_ms * 100)} timesteps)...[/dim]")
    
    with console.status("[bold green]Simulating QP dynamics...") as status:
        results = twin.run_simulation(duration_ms * 1000, 10.0)  # dt = 10 μs
        
        # Quick stats
        num_bursts = len(twin.burst_sim.burst_history)
        status.update(f"[bold green]Processed {num_bursts} QP burst events")
    
    # Calculate metrics
    survival_mult = twin.get_survival_multiplier()
    
    final_gap = twin.topological_gap_history[-1][1] if twin.topological_gap_history else 0
    mean_gap = sum(g for _, g in twin.topological_gap_history) / len(twin.topological_gap_history) if twin.topological_gap_history else 0
    
    lifetimes = [l for _, l in twin.parity_lifetime_history]
    mean_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0
    min_lifetime = min(lifetimes) if lifetimes else 0
    
    # Results
    console.print("\n[bold]Simulation Results:[/bold]")
    console.print(f"  • Burst Events: {num_bursts}")
    console.print(f"  • Survival Multiplier: [bold {'green' if survival_mult >= 10 else 'red'}]{survival_mult:.1f}x[/]")
    console.print(f"  • Mean Parity Lifetime: {mean_lifetime:.1f} μs")
    console.print(f"  • Min Parity Lifetime: {min_lifetime:.1f} μs")
    console.print(f"  • Mean Topological Gap: {mean_gap:.3f} meV")
    console.print(f"  • Final Gap: {final_gap:.3f} meV")
    
    # NASA SBIR assessment
    console.print("\n[bold]NASA SBIR Feasibility Assessment:[/bold]")
    if survival_mult >= 100:
        console.print("  [bold green]✓ EXCEEDS TARGET[/bold green]: 100× survival multiplier achieved")
    elif survival_mult >= 10:
        console.print("  [bold green]✓ TARGET ACHIEVED[/bold green]: 10-100× survival multiplier")
    else:
        console.print("  [bold red]✗ BELOW TARGET[/bold red]: <10× survival multiplier")
    
    if mean_gap >= 0.1:
        console.print("  [bold green]✓ GAP PROTECTION[/bold green]: Topological gap maintained")
    else:
        console.print("  [bold red]✗ GAP COLLAPSE[/bold red]: Insufficient topological protection")
    
    # Save output if requested
    output.mkdir(exist_ok=True)
    summary_file = output / "simulation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Quantum Immune Digital Twin - Simulation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Duration: {duration_ms} ms\n")
        f.write(f"  Dose Rate: {dose_rate} mSv/hr\n")
        f.write(f"  Magnetic Field: {magnetic_field} T\n")
        f.write(f"  Random Seed: {seed}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Burst Events: {num_bursts}\n")
        f.write(f"  Survival Multiplier: {survival_mult:.2f}x\n")
        f.write(f"  Mean Parity Lifetime: {mean_lifetime:.2f} μs\n")
        f.write(f"  Mean Topological Gap: {mean_gap:.4f} meV\n")
    
    console.print(f"\n[dim]Results saved to: {summary_file}[/dim]")


@app.command()
def dashboard(
    output: Path = typer.Option(
        Path("./outputs"),
        "--output",
        "-o",
        help="Output directory for dashboard files"
    ),
    duration_ms: float = typer.Option(
        10.0,
        "--duration",
        "-d",
        help="Simulation duration for time-series plots (ms)"
    ),
) -> None:
    """Generate full validation dashboard with interactive plots."""
    
    title = Text("Quantum Immune Digital Twin", style="bold cyan")
    subtitle = Text("Metrics Validation Dashboard", style="dim")
    console.print(Panel.fit(f"{title}\n{subtitle}"))
    
    console.print(f"\n[dim]Output directory: {output.absolute()}[/dim]")
    console.print(f"[dim]Duration: {duration_ms} ms[/dim]\n")
    
    dash = MetricsDashboard(output_dir=str(output))
    
    try:
        outputs = dash.generate_full_report()
        
        console.print("\n[bold green]Dashboard generated successfully![/bold green]")
        console.print("\n[bold]Output files:[/bold]")
        for f in outputs:
            console.print(f"  • {f.name}")
            
    except Exception as e:
        console.print(f"\n[bold red]Error generating dashboard: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def validate(
    num_runs: int = typer.Option(
        5,
        "--runs",
        "-n",
        help="Number of ensemble runs per configuration"
    ),
) -> None:
    """Run validation suite against NASA SBIR targets."""
    
    title = Text("NASA SBIR Phase I Validation Suite", style="bold cyan")
    console.print(Panel.fit(title))
    
    # Test configurations
    test_cases = [
        ("Low Earth Orbit", 10.0, 1.0),
        ("Deep Space (nominal)", 100.0, 1.0),
        ("Deep Space (solar max)", 200.0, 1.0),
        ("High field regime", 100.0, 2.0),
    ]
    
    console.print(f"\n[bold]Running {num_runs} ensemble simulations per configuration...[/bold]\n")
    
    all_passed = True
    
    for name, dose, field in test_cases:
        console.print(f"[dim]Testing: {name} ({dose} mSv/hr, {field} T)...[/dim]")
        
        multipliers = []
        for run in range(num_runs):
            cosmic = CosmicRayParameters(dose_rate_msv_hr=dose, qp_lifetime_us=100.0)
            twin = DigitalTwinEngine(
                cosmic_params=cosmic,
                magnetic_field_t=field,
                random_seed=42 + run
            )
            twin.run_simulation(5000.0, 10.0)  # 5 ms
            multipliers.append(twin.get_survival_multiplier())
        
        mean_mult = sum(multipliers) / len(multipliers)
        std_mult = (sum((m - mean_mult)**2 for m in multipliers) / len(multipliers)) ** 0.5
        
        passed = mean_mult >= 10.0
        all_passed = all_passed and passed
        
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {status} Survival: {mean_mult:.1f}±{std_mult:.1f}x")
    
    console.print("\n" + "=" * 50)
    if all_passed:
        console.print("[bold green]All validation tests PASSED[/bold green]")
        console.print("[dim]System meets NASA SBIR Phase I feasibility criteria[/dim]")
    else:
        console.print("[bold red]Some validation tests FAILED[/bold red]")
        console.print("[dim]Additional optimization required for SBIR targets[/dim]")


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version information"
    ),
) -> None:
    """Quantum Immune Digital Twin - Active QP Evasion Simulator"""
    if version:
        console.print("[bold]QIDT[/bold] v0.1.0 - Phase 0 Release")
        console.print("Active QP Evasion for Rad-Hard Majorana-Photonic Compute")
        raise typer.Exit()


if __name__ == "__main__":
    app()
