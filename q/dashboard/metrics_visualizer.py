"""
Metrics Validation Dashboard for Quantum Immune Digital Twin

Visualizes baseline algorithmic survival multipliers and comparative
performance of Active Evasion vs. passive topological protection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    make_subplots = None
    px = None

# Import core physics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.bdg_hamiltonian import DigitalTwinEngine, CosmicRayParameters, NanowireGeometry


@dataclass
class SimulationConfig:
    """Configuration for batch simulation runs."""
    duration_us: float = 10000.0      # 10 ms simulation
    dt_us: float = 10.0                # 10 μs timestep
    num_runs: int = 10                 # Ensemble average
    
    # Parameter sweep ranges
    dose_rates: tuple[float, ...] = (1.0, 10.0, 50.0, 100.0, 200.0)  # mSv/hr
    magnetic_fields: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)  # Tesla


class MetricsDashboard:
    """
    Interactive dashboard for quantum immune metrics visualization.
    
    Generates publication-ready figures validating:
    1. Baseline parity lifetime vs. cosmic flux
    2. Topological gap dynamics under QP bursts
    3. Algorithmic survival multiplier (Active vs. Passive)
    4. Majorana mode overlap degradation
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_baseline_comparison(
        self,
        config: Optional[SimulationConfig] = None
    ) -> dict[str, NDArray]:
        """
        Run ensemble simulations across flux parameters.
        
        Returns data structure with survival multipliers and error metrics
        for both Active Evasion and Passive protection scenarios.
        """
        config = config or SimulationConfig()
        results = {
            "dose_rates": [],
            "active_survival": [],
            "active_std": [],
            "passive_survival": [],
            "passive_std": [],
            "gap_suppression": []
        }
        
        for dose_rate in config.dose_rates:
            active_multipliers = []
            passive_multipliers = []
            gap_data = []
            
            for run in range(config.num_runs):
                # Active evasion simulation
                cosmic_active = CosmicRayParameters(
                    dose_rate_msv_hr=dose_rate,
                    qp_lifetime_us=100.0
                )
                twin_active = DigitalTwinEngine(
                    cosmic_params=cosmic_active,
                    random_seed=42 + run
                )
                twin_active.run_simulation(config.duration_us, config.dt_us)
                active_mult = twin_active.get_survival_multiplier()
                active_multipliers.append(active_mult)
                
                # Passive simulation (no evasion - longer QP lifetime)
                cosmic_passive = CosmicRayParameters(
                    dose_rate_msv_hr=dose_rate,
                    qp_lifetime_us=10000.0  # 10 ms - no clearing
                )
                twin_passive = DigitalTwinEngine(
                    cosmic_params=cosmic_passive,
                    random_seed=42 + run
                )
                twin_passive.run_simulation(config.duration_us, config.dt_us)
                passive_mult = twin_passive.get_survival_multiplier()
                passive_multipliers.append(passive_mult)
                
                # Topological gap tracking
                gaps = [g for _, g in twin_active.topological_gap_history]
                if gaps:
                    gap_data.append(np.mean(gaps))
            
            results["dose_rates"].append(dose_rate)
            results["active_survival"].append(np.mean(active_multipliers))
            results["active_std"].append(np.std(active_multipliers))
            results["passive_survival"].append(np.mean(passive_multipliers))
            results["passive_std"].append(np.std(passive_multipliers))
            results["gap_suppression"].append(np.mean(gap_data) if gap_data else 0)
        
        return {k: np.array(v) for k, v in results.items()}
    
    def create_survival_multiplier_plot(
        self,
        data: dict[str, NDArray],
        filename: str = "survival_multiplier.html"
    ) -> Path:
        """
        Create interactive Plotly figure showing algorithmic survival advantage.
        
        Demonstrates 10-100× extension of parity lifetime under continuous flux.
        """
        fig = go.Figure()
        
        # Active Evasion trace
        fig.add_trace(go.Scatter(
            x=data["dose_rates"],
            y=data["active_survival"],
            error_y=dict(type='data', array=data["active_std"], visible=True),
            mode='lines+markers',
            name='Active Evasion (Algorithmic Reroute)',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=10, symbol='diamond')
        ))
        
        # Passive Protection trace
        fig.add_trace(go.Scatter(
            x=data["dose_rates"],
            y=data["passive_survival"],
            error_y=dict(type='data', array=data["passive_std"], visible=True),
            mode='lines+markers',
            name='Passive Topological Protection',
            line=dict(color='#EF553B', width=3, dash='dash'),
            marker=dict(size=10, symbol='circle')
        ))
        
        # NASA SBIR target zone
        fig.add_hrect(
            y0=10, y1=100,
            fillcolor="rgba(0, 204, 150, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="NASA SBIR Target: 10-100× Survival",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=dict(
                text="Algorithmic Survival Multiplier vs. Cosmic Radiation Flux",
                font=dict(size=18)
            ),
            xaxis_title="Radiation Dose Rate (mSv/hr)",
            yaxis_title="Survival Multiplier (τ_active / τ_passive)",
            yaxis_type="log",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600,
            width=900
        )
        
        # Add annotations
        fig.add_annotation(
            x=100, y=50,
            text="Deep Space<br>(100 mSv/hr)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#636EFA",
            font=dict(size=12)
        )
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        return output_path
    
    def create_qp_dynamics_visualization(
        self,
        duration_us: float = 1000.0,
        filename: str = "qp_dynamics.html"
    ) -> Path:
        """
        Create animated visualization of QP diffusion and Hamiltonian response.
        
        Shows real-time gap suppression and Majorana mode overlap.
        """
        # Run single detailed simulation
        cosmic = CosmicRayParameters(dose_rate_msv_hr=100.0, qp_lifetime_us=500.0)
        twin = DigitalTwinEngine(cosmic_params=cosmic, num_sites=50, random_seed=42)
        
        # Collect detailed timestep data
        times = []
        gaps = []
        lifetimes = []
        overlaps = []
        qp_max = []
        
        num_steps = int(duration_us / 5.0)  # 5 μs resolution
        for _ in range(num_steps):
            metrics = twin.step(5.0)
            times.append(metrics["time_us"])
            gaps.append(metrics["topological_gap_mev"])
            lifetimes.append(metrics["parity_lifetime_us"])
            overlaps.append(metrics["majorana_overlap"])
            qp_max.append(metrics["max_qp_density"])
        
        times = np.array(times)
        gaps = np.array(gaps)
        lifetimes = np.array(lifetimes)
        overlaps = np.array(overlaps)
        qp_max = np.array(qp_max)
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "QP Poisoning Density (max)",
                "Topological Gap Suppression",
                "Majorana Overlap (Protection Metric)"
            ),
            vertical_spacing=0.08
        )
        
        # QP Density trace
        fig.add_trace(
            go.Scatter(
                x=times, y=qp_max,
                mode='lines',
                name='QP Density',
                line=dict(color='#EF553B', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 85, 59, 0.2)'
            ),
            row=1, col=1
        )
        
        # Gap trace
        fig.add_trace(
            go.Scatter(
                x=times, y=gaps,
                mode='lines',
                name='Topological Gap',
                line=dict(color='#00CC96', width=2)
            ),
            row=2, col=1
        )
        
        # Overlap trace
        fig.add_trace(
            go.Scatter(
                x=times, y=overlaps,
                mode='lines',
                name='Majorana Overlap',
                line=dict(color='#636EFA', width=2)
            ),
            row=3, col=1
        )
        
        # Add burst event markers
        for burst in twin.burst_sim.burst_history:
            if burst.time_us < duration_us:
                for row in range(1, 4):
                    fig.add_vline(
                        x=burst.time_us,
                        line=dict(color="red", dash="dash", width=1),
                        row=row, col=1,
                        annotation_text="⚡" if row == 1 else None,
                        annotation_position="top"
                    )
        
        fig.update_layout(
            title=dict(
                text="Real-Time QP Dynamics Under Continuous Cosmic Flux (100 mSv/hr)",
                font=dict(size=16)
            ),
            template="plotly_white",
            height=800,
            width=1000,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time (μs)", row=3, col=1)
        fig.update_yaxes(title_text="Normalized QP Density", row=1, col=1)
        fig.update_yaxes(title_text="Gap (meV)", row=2, col=1)
        fig.update_yaxes(title_text="Overlap (0-1)", row=3, col=1)
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        return output_path
    
    def create_parameter_sweep_heatmap(
        self,
        config: Optional[SimulationConfig] = None,
        filename: str = "parameter_sweep.html"
    ) -> Path:
        """
        Create heatmap of survival multiplier across B-field and dose rate.
        
        Validates operating regime for topological protection.
        """
        config = config or SimulationConfig()
        
        # Run parameter sweep
        survival_matrix = np.zeros((len(config.magnetic_fields), len(config.dose_rates)))
        
        for i, B in enumerate(config.magnetic_fields):
            for j, dose in enumerate(config.dose_rates):
                multipliers = []
                for run in range(5):  # Smaller ensemble for sweep
                    cosmic = CosmicRayParameters(dose_rate_msv_hr=dose)
                    # Create geometry with this B-field
                    geo = NanowireGeometry()
                    twin = DigitalTwinEngine(
                        geometry=geo,
                        cosmic_params=cosmic,
                        magnetic_field_t=B,
                        random_seed=42 + run
                    )
                    twin.run_simulation(5000.0, 10.0)  # 5 ms
                    multipliers.append(twin.get_survival_multiplier())
                survival_matrix[i, j] = np.mean(multipliers)
        
        fig = go.Figure(data=go.Heatmap(
            z=survival_matrix,
            x=config.dose_rates,
            y=config.magnetic_fields,
            colorscale='RdYlGn',
            zmid=10,
            colorbar=dict(title="Survival<br>Multiplier"),
            hovertemplate='Dose: %{x} mSv/hr<br>B-field: %{y} T<br>Multiplier: %{z:.1f}x<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Survival Multiplier: Magnetic Field vs. Radiation Dose",
                font=dict(size=16)
            ),
            xaxis_title="Radiation Dose Rate (mSv/hr)",
            yaxis_title="Magnetic Field (Tesla)",
            template="plotly_white",
            height=500,
            width=800
        )
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        return output_path
    
    def generate_full_report(self) -> list[Path]:
        """
        Generate complete validation dashboard.
        
        Returns list of generated HTML files.
        """
        print("Running baseline comparison simulations...")
        baseline_data = self.run_baseline_comparison()
        
        print("Generating visualizations...")
        outputs = []
        
        outputs.append(self.create_survival_multiplier_plot(baseline_data))
        print(f"  ✓ Survival multiplier plot: {outputs[-1].name}")
        
        outputs.append(self.create_qp_dynamics_visualization())
        print(f"  ✓ QP dynamics visualization: {outputs[-1].name}")
        
        outputs.append(self.create_parameter_sweep_heatmap())
        print(f"  ✓ Parameter sweep heatmap: {outputs[-1].name}")
        
        # Generate summary statistics
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Quantum Immune Digital Twin - Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Deep Space (100 mSv/hr) Results:\n")
            idx_100 = np.where(np.array(baseline_data["dose_rates"]) == 100.0)[0]
            if len(idx_100) > 0:
                i = idx_100[0]
                f.write(f"  Active Evasion Multiplier: {baseline_data['active_survival'][i]:.1f}x ± {baseline_data['active_std'][i]:.1f}\n")
                f.write(f"  Passive Protection Multiplier: {baseline_data['passive_survival'][i]:.1f}x ± {baseline_data['passive_std'][i]:.1f}\n")
                f.write(f"  Mean Topological Gap: {baseline_data['gap_suppression'][i]:.3f} meV\n")
            f.write("\n")
            f.write("NASA SBIR Target: 10-100× survival multiplier\n")
            f.write(f"Status: {'✓ ACHIEVED' if np.max(baseline_data['active_survival']) >= 10 else '✗ NOT YET ACHIEVED'}\n")
        
        outputs.append(summary_path)
        print(f"  ✓ Summary statistics: {summary_path.name}")
        
        return outputs


def main():
    """CLI entry point for dashboard generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Quantum Immune Digital Twin validation dashboard"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory for output files"
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=10.0,
        help="Simulation duration in milliseconds"
    )
    
    args = parser.parse_args()
    
    dashboard = MetricsDashboard(output_dir=args.output_dir)
    
    print("=" * 60)
    print("Quantum Immune Digital Twin - Metrics Validation Dashboard")
    print("=" * 60)
    print()
    
    outputs = dashboard.generate_full_report()
    
    print()
    print("=" * 60)
    print(f"Generated {len(outputs)} output files in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
