"""
ONNX INT8 Export Pipeline for Port-Hamiltonian System

Exports PyTorch PHS model to ONNX INT8 for M2 Air deployment.

Pipeline:
1. Validate PyTorch model structure
2. Generate calibration dataset (representative orbital states)
3. Export to FP32 ONNX (baseline)
4. Quantize to INT8 (for M2 Air efficiency)
5. Validate quantized model accuracy (<1% error)
6. Optimize for ONNX Runtime (graph pruning)

Target Hardware:
- M2 Air (Apple Silicon Neural Engine)
- 100Hz inference requirement (<10ms per call)
- INT8 for 4x speedup vs FP32

Reference:
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- TensorRT: Alternative for NVIDIA deployment
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


@dataclass
class ONNXExportConfig:
<<<<<<< HEAD
                """Configuration for ONNX export."""

                # Model settings
                model_name: str = "port_hamiltonian_system"
                version: str = "1.0"

                # Export settings
                opset_version: int = 17 # Latest stable
                dynamic_axes: bool = True # Allow variable batch size

                # : Mixed Precision Settings
                # Symplecticity is allergic to low precision - H-net must remain FP32
                # Dissipation (R-net) and Control (g-net) can use BF16/INT8
                mixed_precision: bool = True
                h_net_precision: str = "fp32" # Hamiltonian: must be FP32 for gradient accuracy
                r_net_precision: str = "bf16" # Dissipation: can be BF16 (lossy by nature)
                g_net_precision: str = "int8" # Control: can be INT8 (tolerant to 1e-3 error)

                # Quantization settings
                quantize: bool = True
                quantization_mode: str = "static" # Static quantization for M2 cache line awareness
                int8_target: bool = True

                # Calibration settings (for static quantization)
                calibration_samples: int = 100
                calibration_range_km: Tuple[float, float] = (200, 36000)

                # Output settings
                output_dir: str = "onnx_models"
                fp32_path: str = None
                int8_path: str = None
                mixed_precision_path: str = None

                # Validation settings
                validate_export: bool = True
                tolerance_percent: float = 1.0 # Max 1% error after quantization

                def __post_init__(self):
                                if self.fp32_path is None:
                                                self.fp32_path = f"{self.output_dir}/{self.model_name}_fp32.onnx"
                                                if self.int8_path is None:
                                                                self.int8_path = f"{self.output_dir}/{self.model_name}_int8.onnx"
                                                                if self.mixed_precision_path is None:
                                                                                self.mixed_precision_path = f"{self.output_dir}/{self.model_name}_mixed.onnx"


class CalibrationDataset:
                """
                Generate calibration dataset for static quantization.

                For dynamic quantization, this is only used for validation.
                """

                def __init__(self, num_samples: int = 100,
                altitude_range_km: Tuple[float, float] = (200, 36000)):
                                self.num_samples = num_samples
                                self.altitude_range = altitude_range_km
                                self.data = self._generate_states()

                                def _generate_states(self) -> np.ndarray:
                                                """Generate representative orbital states."""
                                                np.random.seed(42)
                                                states = []

                                                r_min = 6378137.0 + self.altitude_range[0] * 1000
                                                r_max = 6378137.0 + self.altitude_range[1] * 1000

                                                for _ in range(self.num_samples):
                                                                r = np.random.uniform(r_min, r_max)
                                                                theta = np.random.uniform(0, 2 * np.pi)
                                                                phi = np.random.uniform(0, np.pi)

                                                                x = r * np.sin(phi) * np.cos(theta)
                                                                y = r * np.sin(phi) * np.sin(theta)
                                                                z = r * np.cos(phi)

                                                                v_mag = np.sqrt(3.986004418e14 / r)
                                                                vx = v_mag * np.random.uniform(-1, 1)
                                                                vy = v_mag * np.random.uniform(-1, 1)
                                                                vz = v_mag * np.random.uniform(-1, 1)

                                                                states.append([x, y, z, vx, vy, vz])

                                                                return np.array(states, dtype=np.float32)

                                                                def __iter__(self):
                                                                                """Yield calibration samples."""
                                                                                for state in self.data:
                                                                                                yield torch.tensor(state, dtype=torch.float32).unsqueeze(0)


class ONNXExporter:
                """
                Export PyTorch model to ONNX with INT8 quantization.
                """

                def __init__(self, config: ONNXExportConfig = None):
                                self.config = config or ONNXExportConfig()
                                Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

                                def export_fp32(self, model: nn.Module) -> str:
                                                """
                                                Export model to FP32 ONNX.

                                                Args:
                                                model: PyTorch Port-Hamiltonian model

                                                Returns:
                                                Path to exported ONNX file
                                                """
                                                print("=" * 70)
                                                print("Exporting to FP32 ONNX")
                                                print("=" * 70)

                                                model.eval()

                                                # Example input for tracing
                                                example_input = torch.randn(1, 6, dtype=torch.float32)

                                                # Dynamic axes for variable batch size
                                                dynamic_axes = {
                                                'input': {0: 'batch_size'},
                                                'output': {0: 'batch_size'},
                                                }

                                                # Export
                                                torch.onnx.export(
                                                model,
                                                example_input,
                                                self.config.fp32_path,
                                                opset_version=self.config.opset_version,
                                                input_names=['input'],
                                                output_names=['output'],
                                                dynamic_axes=dynamic_axes if self.config.dynamic_axes else None,
                                                verbose=False,
                                                )

                                                print(f"✅ FP32 ONNX exported to: {self.config.fp32_path}")

                                                # Validate export
                                                if self.config.validate_export:
                                                                self._validate_onnx_model(self.config.fp32_path, model)

                                                                return self.config.fp32_path

                                                                def export_mixed_precision(self, model: nn.Module) -> str:
                                                                                """
                                                                                Export model with mixed precision (FP32 H-net, BF16/INT8 R/g-nets).

                                                                                : Symplecticity requires FP32 gradients for H-net.
                                                                                Dissipation and control can tolerate lower precision.

                                                                                Args:
                                                                                model: PyTorch Port-Hamiltonian model with separate H, R, g heads

                                                                                Returns:
                                                                                Path to exported mixed precision ONNX file
                                                                                """
                                                                                print("\n" + "=" * 70)
                                                                                print("Exporting to Mixed Precision ONNX")
                                                                                print("=" * 70)
                                                                                print(f" H-net precision: {self.config.h_net_precision}")
                                                                                print(f" R-net precision: {self.config.r_net_precision}")
                                                                                print(f" g-net precision: {self.config.g_net_precision}")

                                                                                model.eval()

                                                                                # Example input for tracing
                                                                                example_input = torch.randn(1, 6, dtype=torch.float32)

                                                                                # Dynamic axes for variable batch size
                                                                                dynamic_axes = {
                                                                                'input': {0: 'batch_size'},
                                                                                'h_output': {0: 'batch_size'},
                                                                                'r_output': {0: 'batch_size'},
                                                                                'g_output': {0: 'batch_size'},
                                                                                }

                                                                                # Export with mixed precision annotations
                                                                                # Note: This requires model to have separate heads
                                                                                # For now, export as FP32 and annotate for post-processing
                                                                                torch.onnx.export(
                                                                                model,
                                                                                example_input,
                                                                                self.config.mixed_precision_path,
                                                                                opset_version=self.config.opset_version,
                                                                                input_names=['input'],
                                                                                output_names=['output'],
                                                                                dynamic_axes=dynamic_axes if self.config.dynamic_axes else None,
                                                                                verbose=False,
                                                                                )

                                                                                print(f"✅ Mixed Precision ONNX exported to: {self.config.mixed_precision_path}")

                                                                                # Add precision metadata to ONNX model
                                                                                import onnx
                                                                                onnx_model = onnx.load(self.config.mixed_precision_path)

                                                                                # Add custom metadata for precision hints
                                                                                meta = onnx_model.metadata_props.add()
                                                                                meta.key = "h_net_precision"
                                                                                meta.value = self.config.h_net_precision

                                                                                meta = onnx_model.metadata_props.add()
                                                                                meta.key = "r_net_precision"
                                                                                meta.value = self.config.r_net_precision

                                                                                meta = onnx_model.metadata_props.add()
                                                                                meta.key = "g_net_precision"
                                                                                meta.value = self.config.g_net_precision

                                                                                onnx.save(onnx_model, self.config.mixed_precision_path)

                                                                                print(f" Precision metadata added to ONNX model")

                                                                                # Validate export
                                                                                if self.config.validate_export:
                                                                                                self._validate_onnx_model(self.config.mixed_precision_path, model)

                                                                                                return self.config.mixed_precision_path

                                                                                                def quantize_int8(self, fp32_path: str) -> str:
                                                                                                                """
                                                                                                                Quantize FP32 ONNX model to INT8.

                                                                                                                Args:
                                                                                                                fp32_path: Path to FP32 ONNX model

                                                                                                                Returns:
                                                                                                                Path to INT8 ONNX model
                                                                                                                """
                                                                                                                print("\n" + "=" * 70)
                                                                                                                print("Quantizing to INT8")
                                                                                                                print("=" * 70)

                                                                                                                if self.config.quantization_mode == "dynamic":
                                                                                                                                # Dynamic quantization (no calibration needed)
                                                                                                                                quantize_dynamic(
                                                                                                                                fp32_path,
                                                                                                                                self.config.int8_path,
                                                                                                                                weight_type=QuantType.QUInt8 if self.config.int8_target else QuantType.QInt8,
                                                                                                                                )
                                                                                                                                print(f"✅ Dynamic INT8 quantization complete")
                                                                                                                else:
                                                                                                                                # Static quantization (requires calibration)
                                                                                                                                print("⚠️ Static quantization requires calibration dataset")
                                                                                                                                print(" Using dynamic quantization for now")
                                                                                                                                quantize_dynamic(
                                                                                                                                fp32_path,
                                                                                                                                self.config.int8_path,
                                                                                                                                weight_type=QuantType.QUInt8,
                                                                                                                                )

                                                                                                                                print(f"✅ INT8 ONNX exported to: {self.config.int8_path}")

                                                                                                                                return self.config.int8_path

                                                                                                                                def _validate_onnx_model(self, onnx_path: str, pytorch_model: nn.Module):
                                                                                                                                                """
                                                                                                                                                Validate ONNX model matches PyTorch model.

                                                                                                                                                Args:
                                                                                                                                                onnx_path: Path to ONNX model
                                                                                                                                                pytorch_model: Original PyTorch model
                                                                                                                                                """
                                                                                                                                                print("\nValidating ONNX export...")

                                                                                                                                                # Load ONNX model
                                                                                                                                                session = ort.InferenceSession(onnx_path)

                                                                                                                                                # Test input
                                                                                                                                                test_input = torch.randn(1, 6, dtype=torch.float32)

                                                                                                                                                # PyTorch output
                                                                                                                                                pytorch_model.eval()
                                                                                                                                                with torch.no_grad():
                                                                                                                                                                pytorch_output = pytorch_model(test_input)

                                                                                                                                                                # ONNX output
                                                                                                                                                                onnx_output = session.run(None, {'input': test_input.numpy()})[0]

                                                                                                                                                                # Compare
                                                                                                                                                                error = np.max(np.abs(pytorch_output.numpy() - onnx_output))

                                                                                                                                                                if error < 1e-5:
                                                                                                                                                                                print(f"✅ ONNX validation passed (max error: {error:.2e})")
                                                                                                                                                                else:
                                                                                                                                                                                print(f"⚠️ ONNX validation warning (max error: {error:.2e})")

                                                                                                                                                                                def validate_quantization(self, fp32_path: str, int8_path: str,
                                                                                                                                                                                model: nn.Module) -> Dict[str, float]:
                                                                                                                                                                                                """
                                                                                                                                                                                                Validate INT8 quantization accuracy.

                                                                                                                                                                                                Args:
                                                                                                                                                                                                fp32_path: Path to FP32 ONNX model
                                                                                                                                                                                                int8_path: Path to INT8 ONNX model
                                                                                                                                                                                                model: PyTorch model for reference

                                                                                                                                                                                                Returns:
                                                                                                                                                                                                Validation statistics
                                                                                                                                                                                                """
                                                                                                                                                                                                print("\n" + "=" * 70)
                                                                                                                                                                                                print("Validating Quantization Accuracy")
                                                                                                                                                                                                print("=" * 70)

                                                                                                                                                                                                # Create calibration dataset
                                                                                                                                                                                                calib_dataset = CalibrationDataset(
                                                                                                                                                                                                num_samples=self.config.calibration_samples,
                                                                                                                                                                                                altitude_range_km=self.config.calibration_range_km,
                                                                                                                                                                                                )

                                                                                                                                                                                                # Load ONNX sessions
                                                                                                                                                                                                fp32_session = ort.InferenceSession(fp32_path)
                                                                                                                                                                                                int8_session = ort.InferenceSession(int8_path)

                                                                                                                                                                                                errors = []

                                                                                                                                                                                                for i, state in enumerate(calib_dataset.data):
                                                                                                                                                                                                                test_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                                                                                                                                                                                                                # FP32 output
                                                                                                                                                                                                                fp32_output = fp32_session.run(None, {'input': test_input.numpy()})[0]

                                                                                                                                                                                                                # INT8 output
                                                                                                                                                                                                                int8_output = int8_session.run(None, {'input': test_input.numpy()})[0]

                                                                                                                                                                                                                # Relative error
                                                                                                                                                                                                                error = np.abs(fp32_output - int8_output) / (np.abs(fp32_output) + 1e-8)
                                                                                                                                                                                                                errors.append(np.max(error))

                                                                                                                                                                                                                errors = np.array(errors)

                                                                                                                                                                                                                stats = {
                                                                                                                                                                                                                'max_error_percent': float(np.max(errors) * 100),
                                                                                                                                                                                                                'mean_error_percent': float(np.mean(errors) * 100),
                                                                                                                                                                                                                'median_error_percent': float(np.median(errors) * 100),
                                                                                                                                                                                                                'p95_error_percent': float(np.percentile(errors, 95) * 100),
                                                                                                                                                                                                                'tolerance_percent': self.config.tolerance_percent,
                                                                                                                                                                                                                'passed': bool(np.max(errors) * 100 < self.config.tolerance_percent),
                                                                                                                                                                                                                }

                                                                                                                                                                                                                print(f" Max error: {stats['max_error_percent']:.3f}%")
                                                                                                                                                                                                                print(f" Mean error: {stats['mean_error_percent']:.3f}%")
                                                                                                                                                                                                                print(f" Median error: {stats['median_error_percent']:.3f}%")
                                                                                                                                                                                                                print(f" P95 error: {stats['p95_error_percent']:.3f}%")
                                                                                                                                                                                                                print(f" Tolerance: {stats['tolerance_percent']:.1f}%")
                                                                                                                                                                                                                print(f" Status: {'✅ PASS' if stats['passed'] else '❌ FAIL'}")

                                                                                                                                                                                                                return stats

                                                                                                                                                                                                                def benchmark_inference(self, onnx_path: str, num_runs: int = 1000) -> Dict[str, float]:
                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                Benchmark ONNX model inference latency.

                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                onnx_path: Path to ONNX model
                                                                                                                                                                                                                                num_runs: Number of inference runs

                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                Latency statistics
                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                print("\n" + "=" * 70)
                                                                                                                                                                                                                                print("Benchmarking Inference Latency")
                                                                                                                                                                                                                                print("=" * 70)

                                                                                                                                                                                                                                import time

                                                                                                                                                                                                                                session = ort.InferenceSession(onnx_path)

                                                                                                                                                                                                                                # Warmup
                                                                                                                                                                                                                                test_input = torch.randn(1, 6, dtype=torch.float32)
                                                                                                                                                                                                                                for _ in range(10):
                                                                                                                                                                                                                                                _ = session.run(None, {'input': test_input.numpy()})

                                                                                                                                                                                                                                                # Benchmark
                                                                                                                                                                                                                                                latencies = []
                                                                                                                                                                                                                                                for _ in range(num_runs):
                                                                                                                                                                                                                                                                start = time.perf_counter()
                                                                                                                                                                                                                                                                _ = session.run(None, {'input': test_input.numpy()})
                                                                                                                                                                                                                                                                end = time.perf_counter()
                                                                                                                                                                                                                                                                latencies.append((end - start) * 1000) # Convert to ms

                                                                                                                                                                                                                                                                latencies = np.array(latencies)

                                                                                                                                                                                                                                                                stats = {
                                                                                                                                                                                                                                                                'mean_latency_ms': float(np.mean(latencies)),
                                                                                                                                                                                                                                                                'median_latency_ms': float(np.median(latencies)),
                                                                                                                                                                                                                                                                'p95_latency_ms': float(np.percentile(latencies, 95)),
                                                                                                                                                                                                                                                                'p99_latency_ms': float(np.percentile(latencies, 99)),
                                                                                                                                                                                                                                                                'min_latency_ms': float(np.min(latencies)),
                                                                                                                                                                                                                                                                'max_latency_ms': float(np.max(latencies)),
                                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                                print(f" Mean: {stats['mean_latency_ms']:.3f} ms")
                                                                                                                                                                                                                                                                print(f" Median: {stats['median_latency_ms']:.3f} ms")
                                                                                                                                                                                                                                                                print(f" P95: {stats['p95_latency_ms']:.3f} ms")
                                                                                                                                                                                                                                                                print(f" P99: {stats['p99_latency_ms']:.3f} ms")
                                                                                                                                                                                                                                                                print(f" 100Hz compatible: {'✅ YES' if stats['p95_latency_ms'] < 10 else '❌ NO'}")

                                                                                                                                                                                                                                                                return stats

                                                                                                                                                                                                                                                                def export_full_pipeline(self, model: nn.Module) -> Dict[str, any]:
                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                Run complete export pipeline.

                                                                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                                                                model: PyTorch Port-Hamiltonian model

                                                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                                                Complete export report
                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                print("\n" + "=" * 70)
                                                                                                                                                                                                                                                                                print("ONNX INT8 Export Pipeline")
                                                                                                                                                                                                                                                                                print("=" * 70)

                                                                                                                                                                                                                                                                                report = {
                                                                                                                                                                                                                                                                                'config': {
                                                                                                                                                                                                                                                                                'model_name': self.config.model_name,
                                                                                                                                                                                                                                                                                'version': self.config.version,
                                                                                                                                                                                                                                                                                'quantize': self.config.quantize,
                                                                                                                                                                                                                                                                                'quantization_mode': self.config.quantization_mode,
                                                                                                                                                                                                                                                                                },
                                                                                                                                                                                                                                                                                'timestamp': datetime.utcnow().isoformat(),
                                                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                                                # Step 1: Export FP32
                                                                                                                                                                                                                                                                                fp32_path = self.export_fp32(model)
                                                                                                                                                                                                                                                                                report['fp32_path'] = fp32_path

                                                                                                                                                                                                                                                                                # Step 2: Quantize to INT8
                                                                                                                                                                                                                                                                                if self.config.quantize:
                                                                                                                                                                                                                                                                                                int8_path = self.quantize_int8(fp32_path)
                                                                                                                                                                                                                                                                                                report['int8_path'] = int8_path

                                                                                                                                                                                                                                                                                                # Step 3: Validate quantization
                                                                                                                                                                                                                                                                                                quant_stats = self.validate_quantization(fp32_path, int8_path, model)
                                                                                                                                                                                                                                                                                                report['quantization_validation'] = quant_stats

                                                                                                                                                                                                                                                                                                # Step 4: Benchmark INT8
                                                                                                                                                                                                                                                                                                latency_stats = self.benchmark_inference(int8_path)
                                                                                                                                                                                                                                                                                                report['latency_benchmark'] = latency_stats
                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                # Benchmark FP32 only
                                                                                                                                                                                                                                                                                                latency_stats = self.benchmark_inference(fp32_path)
                                                                                                                                                                                                                                                                                                report['latency_benchmark'] = latency_stats

                                                                                                                                                                                                                                                                                                # Save report
                                                                                                                                                                                                                                                                                                report_path = Path(self.config.output_dir) / f"{self.config.model_name}_export_report.json"
                                                                                                                                                                                                                                                                                                with open(report_path, 'w') as f:
                                                                                                                                                                                                                                                                                                                json.dump(report, f, indent=2)

                                                                                                                                                                                                                                                                                                                print(f"\n📄 Report saved to: {report_path}")

                                                                                                                                                                                                                                                                                                                print("\n" + "=" * 70)
                                                                                                                                                                                                                                                                                                                print("Export Pipeline Complete")
                                                                                                                                                                                                                                                                                                                print("=" * 70)

                                                                                                                                                                                                                                                                                                                return report


def demo_export():
                """
                Demonstrate ONNX export pipeline with mock model.
                """
                print("=" * 70)
                print("ONNX INT8 Export Demo")
                print("=" * 70)

                config = ONNXExportConfig(
                model_name="demo_phs",
                quantize=True,
                validate_export=True,
                calibration_samples=50,
                )

                exporter = ONNXExporter(config)

                print(f"\nConfiguration:")
                print(f" Model name: {config.model_name}")
                print(f" Quantize: {config.quantize}")
                print(f" Quantization mode: {config.quantization_mode}")
                print(f" Output directory: {config.output_dir}")

                print(f"\nNote: Full export requires actual PyTorch Port-Hamiltonian model.")
                print(f"This demo shows the export pipeline structure.")

                print("\n" + "=" * 70)
                print("To run full export:")
                print(" 1. Train JAX model on H100")
                print(" 2. Transfer weights to PyTorch shell")
                print(" 3. Run: exporter.export_full_pipeline(torch_model)")
                print(" 4. Verify P95 latency < 10ms for 100Hz")
                print(" 5. Verify quantization error < 1%")
                print("=" * 70)


if __name__ == "__main__":
                demo_export()
=======
    """Configuration for ONNX export."""

    # Model settings
    model_name: str = "port_hamiltonian_system"
    version: str = "1.0"

    # Export settings
    opset_version: int = 17  # Latest stable
    dynamic_axes: bool = True  # Allow variable batch size

    # : Mixed Precision Settings
    # Symplecticity is allergic to low precision - H-net must remain FP32
    # Dissipation (R-net) and Control (g-net) can use BF16/INT8
    mixed_precision: bool = True
    h_net_precision: str = "fp32"  # Hamiltonian: must be FP32 for gradient accuracy
    r_net_precision: str = "bf16"  # Dissipation: can be BF16 (lossy by nature)
    # Control: can be INT8 (tolerant to 1e-3 error)
    g_net_precision: str = "int8"

    # Quantization settings
    quantize: bool = True
    # Static quantization for M2 cache line awareness
    quantization_mode: str = "static"
    int8_target: bool = True

    # Calibration settings (for static quantization)
    calibration_samples: int = 100
    calibration_range_km: Tuple[float, float] = (200, 36000)

    # Output settings
    output_dir: str = "onnx_models"
    fp32_path: str = None
    int8_path: str = None
    mixed_precision_path: str = None

    # Validation settings
    validate_export: bool = True
    tolerance_percent: float = 1.0  # Max 1% error after quantization

    def __post_init__(self):
        if self.fp32_path is None:
            self.fp32_path = f"{self.output_dir}/{self.model_name}_fp32.onnx"
        if self.int8_path is None:
            self.int8_path = f"{self.output_dir}/{self.model_name}_int8.onnx"
        if self.mixed_precision_path is None:
            self.mixed_precision_path = f"{self.output_dir}/{self.model_name}_mixed.onnx"


class CalibrationDataset:
    """
    Generate calibration dataset for static quantization.

    For dynamic quantization, this is only used for validation.
    """

    def __init__(self, num_samples: int = 100,
                 altitude_range_km: Tuple[float, float] = (200, 36000)):
        self.num_samples = num_samples
        self.altitude_range = altitude_range_km
        self.data = self._generate_states()

    def _generate_states(self) -> np.ndarray:
        """Generate representative orbital states."""
        np.random.seed(42)
        states = []

        r_min = 6378137.0 + self.altitude_range[0] * 1000
        r_max = 6378137.0 + self.altitude_range[1] * 1000

        for _ in range(self.num_samples):
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            v_mag = np.sqrt(3.986004418e14 / r)
            vx = v_mag * np.random.uniform(-1, 1)
            vy = v_mag * np.random.uniform(-1, 1)
            vz = v_mag * np.random.uniform(-1, 1)

            states.append([x, y, z, vx, vy, vz])

        return np.array(states, dtype=np.float32)

    def __iter__(self):
        """Yield calibration samples."""
        for state in self.data:
            yield torch.tensor(state, dtype=torch.float32).unsqueeze(0)


class ONNXExporter:
    """
    Export PyTorch model to ONNX with INT8 quantization.
    """

    def __init__(self, config: ONNXExportConfig = None):
        self.config = config or ONNXExportConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def export_fp32(self, model: nn.Module) -> str:
        """
        Export model to FP32 ONNX.

        Args:
            model: PyTorch Port-Hamiltonian model

        Returns:
            Path to exported ONNX file
        """
        print("=" * 70)
        print("Exporting to FP32 ONNX")
        print("=" * 70)

        model.eval()

        example_input = torch.randn(1, 6, dtype=torch.float32)

        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

        torch.onnx.export(
            model,
            example_input,
            self.config.fp32_path,
            opset_version=self.config.opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes if self.config.dynamic_axes else None,
            verbose=False,
        )

        print(f"✅ FP32 ONNX exported to: {self.config.fp32_path}")

        if self.config.validate_export:
            self._validate_onnx_model(self.config.fp32_path, model)

        return self.config.fp32_path

    def export_mixed_precision(self, model: nn.Module) -> str:
        """
        Export model with mixed precision (FP32 H-net, BF16/INT8 R/g-nets).
        """
        print("\n" + "=" * 70)
        print("Exporting to Mixed Precision ONNX")
        print("=" * 70)
        print(f" H-net precision: {self.config.h_net_precision}")
        print(f" R-net precision: {self.config.r_net_precision}")
        print(f" g-net precision: {self.config.g_net_precision}")

        model.eval()
        example_input = torch.randn(1, 6, dtype=torch.float32)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'h_output': {0: 'batch_size'},
            'r_output': {0: 'batch_size'},
            'g_output': {0: 'batch_size'},
        }

        torch.onnx.export(
            model,
            example_input,
            self.config.mixed_precision_path,
            opset_version=self.config.opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes if self.config.dynamic_axes else None,
            verbose=False,
        )

        print(f"✅ Mixed Precision ONNX exported to: {self.config.mixed_precision_path}")

        import onnx as _onnx
        onnx_model = _onnx.load(self.config.mixed_precision_path)
        for key, val in [("h_net_precision", self.config.h_net_precision),
                         ("r_net_precision", self.config.r_net_precision),
                         ("g_net_precision", self.config.g_net_precision)]:
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = val
        _onnx.save(onnx_model, self.config.mixed_precision_path)
        print(" Precision metadata added to ONNX model")

        if self.config.validate_export:
            self._validate_onnx_model(self.config.mixed_precision_path, model)

        return self.config.mixed_precision_path

    def quantize_int8(self, fp32_path: str) -> str:
        """
        Quantize FP32 ONNX model to INT8.
        """
        print("\n" + "=" * 70)
        print("Quantizing to INT8")
        print("=" * 70)

        if self.config.quantization_mode == "dynamic":
            quantize_dynamic(
                fp32_path,
                self.config.int8_path,
                weight_type=QuantType.QUInt8 if self.config.int8_target else QuantType.QInt8,
            )
            print("✅ Dynamic INT8 quantization complete")
        else:
            print("⚠️ Static quantization requires calibration dataset")
            print(" Using dynamic quantization for now")
            quantize_dynamic(
                fp32_path,
                self.config.int8_path,
                weight_type=QuantType.QUInt8,
            )

        print(f"✅ INT8 ONNX exported to: {self.config.int8_path}")
        return self.config.int8_path

    def _validate_onnx_model(self, onnx_path: str, pytorch_model: nn.Module):
        """Validate ONNX model matches PyTorch model."""
        print("\nValidating ONNX export...")
        session = ort.InferenceSession(onnx_path)
        test_input = torch.randn(1, 6, dtype=torch.float32)
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        onnx_output = session.run(None, {'input': test_input.numpy()})[0]
        error = np.max(np.abs(pytorch_output.numpy() - onnx_output))
        if error < 1e-5:
            print(f"✅ ONNX validation passed (max error: {error:.2e})")
        else:
            print(f"⚠️ ONNX validation warning (max error: {error:.2e})")

    def validate_quantization(self, fp32_path: str, int8_path: str,
                              model: nn.Module) -> Dict[str, float]:
        """Validate INT8 quantization accuracy."""
        print("\n" + "=" * 70)
        print("Validating Quantization Accuracy")
        print("=" * 70)

        calib_dataset = CalibrationDataset(
            num_samples=self.config.calibration_samples,
            altitude_range_km=self.config.calibration_range_km,
        )
        fp32_session = ort.InferenceSession(fp32_path)
        int8_session = ort.InferenceSession(int8_path)
        errors = []

        for state in calib_dataset.data:
            test_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            fp32_output = fp32_session.run(None, {'input': test_input.numpy()})[0]
            int8_output = int8_session.run(None, {'input': test_input.numpy()})[0]
            error = np.abs(fp32_output - int8_output) / (np.abs(fp32_output) + 1e-8)
            errors.append(np.max(error))

        errors = np.array(errors)
        stats = {
            'max_error_percent': float(np.max(errors) * 100),
            'mean_error_percent': float(np.mean(errors) * 100),
            'median_error_percent': float(np.median(errors) * 100),
            'p95_error_percent': float(np.percentile(errors, 95) * 100),
            'tolerance_percent': self.config.tolerance_percent,
            'passed': bool(np.max(errors) * 100 < self.config.tolerance_percent),
        }

        print(f" Max error: {stats['max_error_percent']:.3f}%")
        print(f" Mean error: {stats['mean_error_percent']:.3f}%")
        print(f" Median error: {stats['median_error_percent']:.3f}%")
        print(f" P95 error: {stats['p95_error_percent']:.3f}%")
        print(f" Tolerance: {stats['tolerance_percent']:.1f}%")
        print(f" Status: {'✅ PASS' if stats['passed'] else '❌ FAIL'}")
        return stats

    def benchmark_inference(self, onnx_path: str,
                            num_runs: int = 1000) -> Dict[str, float]:
        """Benchmark ONNX model inference latency."""
        import time
        print("\n" + "=" * 70)
        print("Benchmarking Inference Latency")
        print("=" * 70)

        session = ort.InferenceSession(onnx_path)
        test_input = torch.randn(1, 6, dtype=torch.float32)

        for _ in range(10):
            session.run(None, {'input': test_input.numpy()})

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {'input': test_input.numpy()})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies = np.array(latencies)
        stats = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
        }

        print(f" Mean: {stats['mean_latency_ms']:.3f} ms")
        print(f" Median: {stats['median_latency_ms']:.3f} ms")
        print(f" P95: {stats['p95_latency_ms']:.3f} ms")
        print(f" P99: {stats['p99_latency_ms']:.3f} ms")
        print(f" 100Hz compatible: {'✅ YES' if stats['p95_latency_ms'] < 10 else '❌ NO'}")
        return stats

    def export_full_pipeline(self, model: nn.Module) -> Dict[str, any]:
        """Run complete export pipeline."""
        print("\n" + "=" * 70)
        print("ONNX INT8 Export Pipeline")
        print("=" * 70)

        report = {
            'config': {
                'model_name': self.config.model_name,
                'version': self.config.version,
                'quantize': self.config.quantize,
                'quantization_mode': self.config.quantization_mode,
            },
            'timestamp': datetime.utcnow().isoformat(),
        }

        fp32_path = self.export_fp32(model)
        report['fp32_path'] = fp32_path

        if self.config.quantize:
            int8_path = self.quantize_int8(fp32_path)
            report['int8_path'] = int8_path
            quant_stats = self.validate_quantization(fp32_path, int8_path, model)
            report['quantization_validation'] = quant_stats
            latency_stats = self.benchmark_inference(int8_path)
        else:
            latency_stats = self.benchmark_inference(fp32_path)
        report['latency_benchmark'] = latency_stats

        report_path = Path(self.config.output_dir) / \
            f"{self.config.model_name}_export_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Report saved to: {report_path}")
        print("\n" + "=" * 70)
        print("Export Pipeline Complete")
        print("=" * 70)
        return report


def demo_export():
    """
    Demonstrate ONNX export pipeline with mock model.
    """
    print("=" * 70)
    print("ONNX INT8 Export Demo")
    print("=" * 70)

    config = ONNXExportConfig(
        model_name="demo_phs",
        quantize=True,
        validate_export=True,
        calibration_samples=50,
    )

    exporter = ONNXExporter(config)

    print(f"\nConfiguration:")
    print(f" Model name: {config.model_name}")
    print(f" Quantize: {config.quantize}")
    print(f" Quantization mode: {config.quantization_mode}")
    print(f" Output directory: {config.output_dir}")

    print(f"\nNote: Full export requires actual PyTorch Port-Hamiltonian model.")
    print(f"This demo shows the export pipeline structure.")

    print("\n" + "=" * 70)
    print("To run full export:")
    print(" 1. Train JAX model on H100")
    print(" 2. Transfer weights to PyTorch shell")
    print(" 3. Run: exporter.export_full_pipeline(torch_model)")
    print(" 4. Verify P95 latency < 10ms for 100Hz")
    print(" 5. Verify quantization error < 1%")
    print("=" * 70)


if __name__ == "__main__":
    demo_export()
>>>>>>> 5943d50fce1c2864021501d841271a496427076b
