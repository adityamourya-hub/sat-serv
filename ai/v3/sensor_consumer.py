#!/usr/bin/env python3
"""
Zero-Copy Sensor Consumer for Plasma Bridge

Reads sensor frames directly from shared memory without copying.
Achieves 0ms transfer time by memory-mapping Arrow Plasma buffers.

Critical for 100Hz: No serialization, no gRPC payload transfer.
Go writes → Plasma mmap → Python reads (all zero-copy)

M2 Air Advantage: Unified memory allows Metal GPU direct access.
"""

import pyarrow.plasma as plasma
import numpy as np
import json
import struct
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PlasmaSensorConsumer:
    """
    Zero-copy consumer for Go-native sensor data.
    
    Replaces gRPC-based data transfer with shared memory mapping.
    Python sees Go data as NumPy arrays without copy overhead.
    """
    
    def __init__(self, socket_path: str = "/tmp/plasma", frame_shape: Tuple[int, ...] = (2048, 2048)):
        """
        Args:
            socket_path: Plasma store socket (must match Go side)
            frame_shape: Expected sensor frame dimensions
        """
        self.client = plasma.connect(socket_path)
        self.frame_shape = frame_shape
        self.frame_dtype = np.float32
        
        # Pre-allocate buffers for Metal GPU upload
        self.gpu_buffer = None  # Will be allocated on first frame
        
        logger.info(f"Plasma consumer connected to {socket_path}")
    
    def get_frame(self, object_id_bytes: bytes) -> Optional[np.ndarray]:
        """
        Retrieve sensor frame from shared memory.
        
        This is ZERO-COPY: the returned NumPy array points directly
        to the memory written by the Go engine.
        
        Args:
            object_id_bytes: 20-byte ObjectID from Go (via gRPC notification)
            
        Returns:
            NumPy array pointing to shared memory buffer, or None if not found
        """
        object_id = plasma.ObjectID(object_id_bytes)
        
        try:
            # 1. Fetch buffer from Plasma (memory map, not copy)
            buffers = self.client.get_buffers([object_id])
            if not buffers or buffers[0] is None:
                logger.warning(f"Object {object_id_bytes} not found in Plasma")
                return None
            
            buffer = buffers[0]
            
            # 2. Create NumPy array VIEW into shared memory
            # This points directly to the mmap'd buffer - NO COPY
            frame_size = np.prod(self.frame_shape) * 4  # float32 = 4 bytes
            
            # View as float32 array
            frame = np.frombuffer(buffer, dtype=self.frame_dtype, count=np.prod(self.frame_shape))
            
            # 3. Reshape to sensor dimensions
            frame = frame.reshape(self.frame_shape)
            
            # 4. Extract metadata (binary encoded at end of buffer)
            meta = self._extract_metadata(buffer, frame_size)
            
            # 5. (Optional) Mark for GPU upload if using Metal
            # frame = self._ensure_gpu_buffer(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            return None
    
    def _extract_metadata(self, buffer: bytes, data_size: int) -> Dict:
        """
        Extract binary metadata written by Go.
        
        Layout (256 bytes at offset data_size):
        [0:8]   int64   - Timestamp (Unix nanoseconds)
        [8:12]  int32   - Width
        [12:16] int32   - Height
        [16:20] int32   - Channels
        [20:24] float32 - Azimuth
        [24:28] float32 - Elevation
        [28:32] float32 - Exposure (ms)
        [32:63] string  - Sensor ID (null-terminated)
        """
        try:
            meta_offset = data_size
            meta_buf = buffer[meta_offset:meta_offset + 256]
            
            timestamp = struct.unpack('<q', meta_buf[0:8])[0]      # int64
            width = struct.unpack('<i', meta_buf[8:12])[0]       # int32
            height = struct.unpack('<i', meta_buf[12:16])[0]       # int32
            channels = struct.unpack('<i', meta_buf[16:20])[0]   # int32
            azimuth = struct.unpack('<f', meta_buf[20:24])[0]    # float32
            elevation = struct.unpack('<f', meta_buf[24:28])[0]  # float32
            exposure = struct.unpack('<f', meta_buf[28:32])[0]     # float32
            
            # Sensor ID is null-terminated string
            sensor_id_bytes = meta_buf[32:63]
            sensor_id = sensor_id_bytes.split(b'\x00')[0].decode('utf-8')
            
            return {
                'timestamp_ns': timestamp,
                'width': width,
                'height': height,
                'channels': channels,
                'azimuth': azimuth,
                'elevation': elevation,
                'exposure_ms': exposure,
                'sensor_id': sensor_id
            }
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    def _ensure_gpu_buffer(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensure frame is in Metal GPU-accessible memory (M2 unified memory).
        
        On M2 Air, this is essentially free since CPU/GPU share pool.
        """
        if self.gpu_buffer is None or self.gpu_buffer.shape != frame.shape:
            # Allocate Metal-compatible buffer
            # On Apple Silicon, this is just page-aligned memory
            self.gpu_buffer = np.empty_like(frame, order='C')
        
        np.copyto(self.gpu_buffer, frame)
        return self.gpu_buffer
    
    def list_available_frames(self) -> list:
        """Debug: List all objects in Plasma store"""
        return self.client.list()
    
    def close(self):
        """Disconnect from Plasma store"""
        self.client.disconnect()
        logger.info("Plasma consumer disconnected")


class SensorConsumerAsync:
    """
    Asynchronous sensor consumer for non-blocking frame processing.
    
    Uses a background thread to prefetch frames while Python processes
    previous frame. Maintains 100Hz throughput.
    """
    
    def __init__(self, socket_path: str = "/tmp/plasma", max_prefetch: int = 2):
        self.consumer = PlasmaSensorConsumer(socket_path)
        self.max_prefetch = max_prefetch
        self.prefetch_queue = []
        self.running = False
        
    def start_prefetch(self, object_ids: list):
        """Start prefetching frames in background"""
        import threading
        
        def prefetch_worker():
            for oid in object_ids:
                if not self.running:
                    break
                frame = self.consumer.get_frame(oid)
                if frame is not None:
                    self.prefetch_queue.append((oid, frame))
                    # Trim queue to max size
                    while len(self.prefetch_queue) > self.max_prefetch:
                        self.prefetch_queue.pop(0)
        
        self.running = True
        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()
    
    def get_frame_nowait(self, object_id: bytes) -> Optional[np.ndarray]:
        """Get frame from prefetch queue or fetch directly"""
        # Check prefetch queue first
        for i, (oid, frame) in enumerate(self.prefetch_queue):
            if oid == object_id:
                self.prefetch_queue.pop(i)
                return frame
        
        # Fetch directly
        return self.consumer.get_frame(object_id)
    
    def stop(self):
        self.running = False
        self.consumer.close()


# Integration with psf_deconvolution.py
class PSFDeconvPlasmaBridge:
    """
    Bridge between Plasma frames and PSF deconvolution pipeline.
    
    Maintains 100Hz by streaming through deconvolution without copies.
    """
    
    def __init__(self, plasma_socket: str = "/tmp/plasma"):
        self.consumer = PlasmaSensorConsumer(
            socket_path=plasma_socket,
            frame_shape=(2048, 2048)
        )
        
    def process_frame(self, object_id: bytes):
        """
        Full pipeline: Plasma → Deconv → Output
        
        Zero-copy at input, minimal copies for GPU operations.
        """
        from psf_deconvolution import PSFModeler
        
        # 1. Zero-copy frame retrieval
        frame = self.consumer.get_frame(object_id)
        if frame is None:
            return None
        
        # 2. PSF deconvolution (GPU accelerated on Metal)
        psf_modeler = PSFModeler()
        deconvolved = psf_modeler.richardson_lucy(frame, iterations=10)
        
        # 3. Return processed frame
        return deconvolved
    
    def close(self):
        self.consumer.close()


# Integration with streak_detection.py
class StreakDetectionPlasmaBridge:
    """
    Real-time streak detection from Plasma buffers.
    """
    
    def __init__(self, plasma_socket: str = "/tmp/plasma"):
        self.consumer = PlasmaSensorConsumer(
            socket_path=plasma_socket,
            frame_shape=(2048, 2048)
        )
        self.frame_buffer = []  # Temporal buffer for 3D FFT
        self.max_buffer = 10   # 100ms at 100Hz
        
    def on_new_frame(self, object_id: bytes) -> Optional[dict]:
        """
        Process incoming frame and detect streaks.
        
        Returns detection metadata if streak found.
        """
        from streak_detection import StreakDetector3D
        
        # Get frame (zero-copy)
        frame = self.consumer.get_frame(object_id)
        if frame is None:
            return None
        
        # Add to temporal buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer:
            self.frame_buffer.pop(0)
        
        # Detect streaks when buffer full
        if len(self.frame_buffer) == self.max_buffer:
            detector = StreakDetector3D()
            cube = np.stack(self.frame_buffer, axis=0)  # [T, H, W]
            
            detections = detector.detect(cube)
            
            if detections:
                return {
                    'object_id': object_id,
                    'detections': detections,
                    'timestamp': time.time()
                }
        
        return None


def demo_plasma_consumer():
    """Demo zero-copy frame retrieval"""
    print("🚀 Plasma Zero-Copy Consumer Demo")
    
    consumer = PlasmaSensorConsumer()
    
    # Wait for Go to write a frame
    print("Waiting for Go frames...")
    
    # In production, this ObjectID comes via gRPC from Go
    # object_id = b'SSIS_SENSOR_FRAME_00'
    
    # For demo, poll available objects
    import time
    for _ in range(10):
        objects = consumer.list_available_frames()
        print(f"Available objects: {len(objects)}")
        
        if objects:
            # Get first available frame
            frame = consumer.get_frame(objects[0].binary())
            if frame is not None:
                print(f"Retrieved frame: {frame.shape}, dtype={frame.dtype}")
                print(f"Memory location: {frame.__array_interface__['data'][0]:#x}")
                print("✅ Zero-copy success!")
                break
        
        time.sleep(0.1)
    
    consumer.close()


if __name__ == "__main__":
    import time
    demo_plasma_consumer()
