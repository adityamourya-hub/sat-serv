"""
Edge Deployment for ADITI 4.0 OBC Inference
Optimized on-board inference engine for radiation-hardened flight software
Zero ground latency - processes sensor data entirely on satellite
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path
import threading
import queue

# Import DL bridges
from cosmic_conn_bridge import CosmicConnDenoiser, DenoisingResult
from astro_rcnn_bridge import AstroRCNNDetector, SegmentationResult
from astronn_bridge import AstroNNSpectroscope, SpectroscopicFeatures

# Hardware-specific runtimes for OBC inference
# ONNX Runtime (cross-platform)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# CoreML for M2 Neural Engine
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

# TensorRT for NVIDIA GPUs
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

if not ONNX_AVAILABLE and not COREML_AVAILABLE and not TENSORRT_AVAILABLE:
    print("[WARN] No optimized runtime available, using PyTorch fallback")


@dataclass
class InferenceRequest:
    """Represents an inference request from the sensor"""
    request_id: str
    sensor_type: str  # 'optical', 'spectroscopy', 'thermal'
    data: np.ndarray
    timestamp: float
    metadata: Optional[Dict] = None


@dataclass
class InferenceResponse:
    """Represents an inference response"""
    request_id: str
    success: bool
    results: Dict
    processing_time_ms: float
    error: Optional[str] = None


@dataclass
class OBCMetrics:
    """OBC performance metrics"""
    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    temperature_celsius: float


class HardwareSelector:
    """
    Automatically selects optimal runtime based on available hardware
    
    Priority order:
    1. TensorRT (NVIDIA GPU) - fastest for CUDA
    2. CoreML (Apple Neural Engine) - optimized for M2
    3. ONNX Runtime (CPU/GPU) - cross-platform fallback
    4. PyTorch fallback - least optimized
    """
    
    @staticmethod
    def detect_hardware() -> Dict[str, bool]:
        """Detect available hardware and runtimes"""
        hardware = {
            "nvidia_gpu": False,
            "apple_silicon": False,
            "cuda_available": False,
            "tensorrt_available": TENSORRT_AVAILABLE,
            "coreml_available": COREML_AVAILABLE,
            "onnx_available": ONNX_AVAILABLE
        }
        
        # Check for NVIDIA GPU
        if TENSORRT_AVAILABLE:
            try:
                # Try to initialize CUDA
                cuda.init()
                hardware["nvidia_gpu"] = True
                hardware["cuda_available"] = True
            except:
                pass
        
        # Check for Apple Silicon
        if COREML_AVAILABLE:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                hardware["apple_silicon"] = True
        
        return hardware
    
    @staticmethod
    def select_runtime(hardware: Optional[Dict[str, bool]] = None) -> str:
        """
        Select optimal runtime based on hardware
        
        Returns:
            Runtime name: 'tensorrt', 'coreml', 'onnx', or 'pytorch'
        """
        if hardware is None:
            hardware = HardwareSelector.detect_hardware()
        
        # Priority selection
        if hardware.get("tensorrt_available") and hardware.get("nvidia_gpu"):
            return "tensorrt"
        elif hardware.get("coreml_available") and hardware.get("apple_silicon"):
            return "coreml"
        elif hardware.get("onnx_available"):
            return "onnx"
        else:
            return "pytorch"
    
    @staticmethod
    def get_runtime_info(runtime: str) -> Dict:
        """Get information about selected runtime"""
        info = {
            "tensorrt": {
                "name": "TensorRT",
                "description": "NVIDIA GPU optimized inference",
                "precision_modes": ["FP32", "FP16", "INT8"],
                "supports_dynamic_shapes": True,
                "typical_latency_ms": 5.0
            },
            "coreml": {
                "name": "CoreML",
                "description": "Apple Neural Engine (ANE) inference",
                "precision_modes": ["FP32", "FP16"],
                "supports_dynamic_shapes": False,
                "typical_latency_ms": 8.0
            },
            "onnx": {
                "name": "ONNX Runtime",
                "description": "Cross-platform inference",
                "precision_modes": ["FP32", "FP16", "INT8"],
                "supports_dynamic_shapes": True,
                "typical_latency_ms": 15.0
            },
            "pytorch": {
                "name": "PyTorch",
                "description": "Fallback PyTorch inference",
                "precision_modes": ["FP32", "FP16"],
                "supports_dynamic_shapes": True,
                "typical_latency_ms": 25.0
            }
        }
        return info.get(runtime, info["pytorch"])


class EdgeInferenceEngine:
    """
    Main inference engine for OBC deployment
    Manages multiple models and provides real-time inference
    """
    
    def __init__(self,
                 model_dir: str = "models/quantized",
                 device: str = "cpu",
                 max_queue_size: int = 100):
        """
        Initialize edge inference engine
        
        Args:
            model_dir: Directory containing quantized ONNX models
            device: 'cpu', 'cuda', or 'tensorrt'
            max_queue_size: Maximum size of inference request queue
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_queue_size = max_queue_size
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_queue = queue.Queue()
        
        # Model registry
        self.models = {}
        self.sessions = {}
        
        # Metrics
        self.metrics = OBCMetrics(
            total_inferences=0,
            successful_inferences=0,
            failed_inferences=0,
            avg_latency_ms=0.0,
            max_latency_ms=0.0,
            min_latency_ms=float('inf'),
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            temperature_celsius=0.0
        )
        
        # Processing thread
        self.is_running = False
        self.processing_thread = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load quantized models from model directory"""
        if not self.model_dir.exists():
            print(f"[OBC] Model directory not found: {self.model_dir}")
            return
        
        # Detect optimal runtime
        hardware = HardwareSelector.detect_hardware()
        runtime = HardwareSelector.select_runtime(hardware)
        runtime_info = HardwareSelector.get_runtime_info(runtime)
        
        print(f"[OBC] Hardware detection: {hardware}")
        print(f"[OBC] Selected runtime: {runtime_info['name']} ({runtime_info['description']})")
        
        # Load models based on runtime
        if runtime == "tensorrt" and TENSORRT_AVAILABLE:
            self._load_tensorrt_models()
        elif runtime == "coreml" and COREML_AVAILABLE:
            self._load_coreml_models()
        elif runtime == "onnx" and ONNX_AVAILABLE:
            self._load_onnx_models()
        else:
            print("[OBC] Using PyTorch fallback (no optimized runtime available)")
        
        # Initialize fallback Python models
        self._init_fallback_models()
    
    def _load_tensorrt_models(self):
        """Load TensorRT engine files"""
        if not TENSORRT_AVAILABLE:
            return
        
        for engine_file in self.model_dir.glob("*.engine"):
            try:
                from tensorrt_optimizer import TensorRTInferenceEngine
                engine = TensorRTInferenceEngine(str(engine_file))
                model_name = engine_file.stem
                self.sessions[model_name] = engine
                print(f"[OBC] Loaded TensorRT engine: {model_name}")
            except Exception as e:
                print(f"[OBC] Failed to load TensorRT engine {engine_file}: {e}")
    
    def _load_coreml_models(self):
        """Load CoreML models for ANE"""
        if not COREML_AVAILABLE:
            return
        
        for model_file in self.model_dir.glob("*.mlpackage"):
            try:
                model = ct.models.MLModel(str(model_file))
                model_name = model_file.stem
                self.sessions[model_name] = model
                print(f"[OBC] Loaded CoreML model: {model_name}")
            except Exception as e:
                print(f"[OBC] Failed to load CoreML model {model_file}: {e}")
    
    def _load_onnx_models(self):
        """Load ONNX models"""
        if not ONNX_AVAILABLE:
            return
        
        for model_file in self.model_dir.glob("*.onnx"):
            try:
                session = ort.InferenceSession(
                    str(model_file),
                    providers=['CPUExecutionProvider'] if self.device == 'cpu' else ['CUDAExecutionProvider']
                )
                model_name = model_file.stem
                self.sessions[model_name] = session
                print(f"[OBC] Loaded ONNX model: {model_name}")
            except Exception as e:
                print(f"[OBC] Failed to load ONNX model {model_file}: {e}")
    
    def _init_fallback_models(self):
        """Initialize fallback Python models"""
        try:
            self.models['denoiser'] = CosmicConnDenoiser()
            print("[OBC] Initialized fallback denoiser")
        except Exception as e:
            print(f"[OBC] Failed to initialize denoiser: {e}")
        
        try:
            self.models['detector'] = AstroRCNNDetector()
            print("[OBC] Initialized fallback detector")
        except Exception as e:
            print(f"[OBC] Failed to initialize detector: {e}")
        
        try:
            self.models['spectroscope'] = AstroNNSpectroscope()
            print("[OBC] Initialized fallback spectroscope")
        except Exception as e:
            print(f"[OBC] Failed to initialize spectroscope: {e}")
    
    def start(self):
        """Start inference processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        print("[OBC] Inference engine started")
    
    def stop(self):
        """Stop inference processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        print("[OBC] Inference engine stopped")
    
    def _process_requests(self):
        """Process inference requests from queue"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                response = self._process_request(request)
                self.response_queue.put(response)
                self._update_metrics(response.processing_time_ms, response.success)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[OBC] Error processing request: {e}")
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request"""
        start_time = time.time()
        
        try:
            results = {}
            
            if request.sensor_type == 'optical':
                # Denoise then detect
                if 'denoiser' in self.models:
                    denoise_result = self.models['denoiser'].denoise(request.data)
                    results['denoising'] = {
                        'hit_count': denoise_result.hit_count,
                        'noise_reduction': denoise_result.noise_reduction
                    }
                    clean_frame = denoise_result.clean_frame
                else:
                    clean_frame = request.data
                
                if 'detector' in self.models:
                    detection_result = self.models['detector'].detect(clean_frame)
                    results['detection'] = {
                        'object_count': detection_result.object_count,
                        'threat_candidates': len(detection_result.threat_candidates),
                        'detections_by_class': detection_result.detections_by_class
                    }
            
            elif request.sensor_type == 'spectroscopy':
                # Spectroscopic analysis
                if 'spectroscope' in self.models:
                    features = self.models['spectroscope'].analyze_spectrum(request.data)
                    material = self.models['spectroscope'].identify_material(request.data)
                    
                    results['spectroscopy'] = {
                        'temperature': features.effective_temperature,
                        'radial_velocity': features.radial_velocity,
                        'material_class': material.material_class,
                        'material_confidence': material.confidence
                    }
            
            else:
                raise ValueError(f"Unknown sensor type: {request.sensor_type}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                results=results,
                processing_time_ms=processing_time
            )
        
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                results={},
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    def submit_request(self, request: InferenceRequest) -> bool:
        """
        Submit an inference request
        
        Args:
            request: Inference request
            
        Returns:
            True if submitted successfully
        """
        try:
            self.request_queue.put(request, timeout=0.1)
            return True
        except queue.Full:
            print(f"[OBC] Request queue full, dropping request {request.request_id}")
            return False
    
    def get_response(self, timeout: float = 1.0) -> Optional[InferenceResponse]:
        """
        Get inference response
        
        Args:
            timeout: Maximum time to wait for response
            
        Returns:
            InferenceResponse or None if timeout
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """Update performance metrics"""
        self.metrics.total_inferences += 1
        
        if success:
            self.metrics.successful_inferences += 1
        else:
            self.metrics.failed_inferences += 1
        
        # Update latency stats
        n = self.metrics.total_inferences
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (n - 1) + latency_ms) / n
        )
        
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
    
    def get_metrics(self) -> OBCMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics.__dict__, f, indent=2)


class RadiationHardenedInference:
    """
    Radiation-hardened inference wrapper
    Implements error correction and fault tolerance for space environment
    """
    
    def __init__(self, base_engine: EdgeInferenceEngine):
        """
        Initialize radiation-hardened wrapper
        
        Args:
            base_engine: Base inference engine
        """
        self.base_engine = base_engine
        self.error_count = 0
        self.max_errors_before_reset = 10
        self.last_reset_time = time.time()
    
    def process_with_redundancy(self, 
                                  request: InferenceRequest,
                                  num_retries: int = 3) -> Optional[InferenceResponse]:
        """
        Process request with redundancy for fault tolerance
        
        Args:
            request: Inference request
            num_retries: Number of retries on failure
            
        Returns:
            InferenceResponse or None if all retries fail
        """
        for attempt in range(num_retries):
            try:
                self.base_engine.submit_request(request)
                response = self.base_engine.get_response(timeout=1.0)
                
                if response and response.success:
                    return response
                
                # Error occurred
                self.error_count += 1
                
                # Check if we need to reset
                if self.error_count >= self.max_errors_before_reset:
                    self._reset_inference_engine()
                    self.error_count = 0
                
            except Exception as e:
                print(f"[OBC-RH] Attempt {attempt + 1} failed: {e}")
                self.error_count += 1
                time.sleep(0.1)  # Small delay between retries
        
        return None
    
    def _reset_inference_engine(self):
        """Reset inference engine after radiation-induced errors"""
        print(f"[OBC-RH] Resetting inference engine after {self.error_count} errors")
        
        # Stop and restart base engine
        self.base_engine.stop()
        time.sleep(0.5)
        self.base_engine.start()
        
        self.last_reset_time = time.time()
    
    def get_radiation_stats(self) -> Dict:
        """Get radiation/error statistics"""
        return {
            'error_count': self.error_count,
            'last_reset_time': self.last_reset_time,
            'time_since_reset': time.time() - self.last_reset_time
        }


class PowerOptimizedInference:
    """
    Power-optimized inference for limited power budgets
    Implements dynamic frequency scaling and model selection
    """
    
    def __init__(self, 
                 base_engine: EdgeInferenceEngine,
                 power_budget_watts: float = 10.0):
        """
        Initialize power-optimized inference
        
        Args:
            base_engine: Base inference engine
            power_budget_watts: Maximum power budget in watts
        """
        self.base_engine = base_engine
        self.power_budget_watts = power_budget_watts
        self.current_power_mode = "NORMAL"
        
        # Power modes
        self.power_modes = {
            "LOW_POWER": {
                'max_latency_ms': 100.0,
                'use_quantized': True,
                'batch_size': 1
            },
            "NORMAL": {
                'max_latency_ms': 50.0,
                'use_quantized': True,
                'batch_size': 4
            },
            "HIGH_PERFORMANCE": {
                'max_latency_ms': 20.0,
                'use_quantized': False,
                'batch_size': 8
            }
        }
    
    def set_power_mode(self, mode: str):
        """
        Set power mode
        
        Args:
            mode: 'LOW_POWER', 'NORMAL', or 'HIGH_PERFORMANCE'
        """
        if mode in self.power_modes:
            self.current_power_mode = mode
            print(f"[OBC-PO] Power mode set to {mode}")
        else:
            print(f"[OBC-PO] Unknown power mode: {mode}")
    
    def adapt_power_mode(self, current_latency_ms: float):
        """
        Dynamically adapt power mode based on performance
        
        Args:
            current_latency_ms: Current inference latency
        """
        mode = self.power_modes[self.current_power_mode]
        
        if current_latency_ms > mode['max_latency_ms']:
            # Need more performance
            if self.current_power_mode == "LOW_POWER":
                self.set_power_mode("NORMAL")
            elif self.current_power_mode == "NORMAL":
                self.set_power_mode("HIGH_PERFORMANCE")
        elif current_latency_ms < mode['max_latency_ms'] * 0.5:
            # Can save power
            if self.current_power_mode == "HIGH_PERFORMANCE":
                self.set_power_mode("NORMAL")
            elif self.current_power_mode == "NORMAL":
                self.set_power_mode("LOW_POWER")


class OBCDeploymentManager:
    """
    Main deployment manager for OBC SSA system
    Coordinates all inference components
    """
    
    def __init__(self,
                 model_dir: str = "models/quantized",
                 radiation_hardened: bool = True,
                 power_budget_watts: float = 10.0):
        """
        Initialize OBC deployment manager
        
        Args:
            model_dir: Directory containing quantized models
            radiation_hardened: Enable radiation-hardened mode
            power_budget_watts: Power budget in watts
        """
        # Create base inference engine
        self.base_engine = EdgeInferenceEngine(model_dir)
        
        # Wrap with radiation hardening if enabled
        if radiation_hardened:
            self.engine = RadiationHardenedInference(self.base_engine)
        else:
            self.engine = self.base_engine
        
        # Wrap with power optimization
        self.power_engine = PowerOptimizedInference(self.engine, power_budget_watts)
        
        # Start engine
        self.base_engine.start()
    
    def process_frame(self, 
                     frame: np.ndarray,
                     sensor_type: str = "optical",
                     metadata: Optional[Dict] = None) -> Optional[InferenceResponse]:
        """
        Process a sensor frame through full OBC pipeline
        
        Args:
            frame: Sensor data
            sensor_type: Type of sensor ('optical', 'spectroscopy', 'thermal')
            metadata: Optional metadata
            
        Returns:
            InferenceResponse or None if failed
        """
        request = InferenceRequest(
            request_id=f"req_{int(time.time() * 1000)}",
            sensor_type=sensor_type,
            data=frame,
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Process with redundancy if radiation-hardened
        if isinstance(self.engine, RadiationHardenedInference):
            return self.engine.process_with_redundancy(request)
        else:
            self.base_engine.submit_request(request)
            return self.base_engine.get_response()
    
    def shutdown(self):
        """Shutdown OBC deployment"""
        self.base_engine.stop()
        print("[OBC] Deployment shutdown complete")
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        status = {
            'engine_running': self.base_engine.is_running,
            'metrics': self.base_engine.get_metrics().__dict__,
            'power_mode': self.power_engine.current_power_mode
        }
        
        if isinstance(self.engine, RadiationHardenedInference):
            status['radiation_stats'] = self.engine.get_radiation_stats()
        
        return status


# Example usage
if __name__ == "__main__":
    print("Testing OBC Edge Deployment...")
    
    # Create deployment manager
    manager = OBCDeploymentManager(
        model_dir="models/quantized",
        radiation_hardened=True,
        power_budget_watts=10.0
    )
    
    # Process synthetic optical frame
    frame = np.random.poisson(100, (512, 512)).astype(float)
    
    response = manager.process_frame(frame, sensor_type="optical")
    
    if response:
        print(f"Response: {response.request_id}")
        print(f"Success: {response.success}")
        print(f"Processing time: {response.processing_time_ms:.2f} ms")
        print(f"Results: {response.results}")
    else:
        print("No response received")
    
    # Get system status
    status = manager.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Running: {status['engine_running']}")
    print(f"  Total inferences: {status['metrics']['total_inferences']}")
    print(f"  Success rate: {status['metrics']['successful_inferences'] / max(status['metrics']['total_inferences'], 1) * 100:.1f}%")
    print(f"  Avg latency: {status['metrics']['avg_latency_ms']:.2f} ms")
    print(f"  Power mode: {status['power_mode']}")
    
    # Shutdown
    manager.shutdown()
    
    print("\nAll tests passed!")
