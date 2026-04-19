"""
Apache Arrow Plasma consumer for Go-Python Zero-Copy IPC
Receives sensor frames via shared memory for MPDAF analytics
"""

import json
import time
import tempfile
import os
from dataclasses import dataclass
from typing import Callable, Optional, List
import logging

import zmq
import numpy as np

# Try to import pyarrow Plasma - may not be available in all environments
try:
    import pyarrow.plasma as plasma
    PLASMA_AVAILABLE = True
except ImportError:
    PLASMA_AVAILABLE = False
    logging.warning("pyarrow.plasma not available - using file fallback")

from mpdaf.obj import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorFrameMetadata:
    """Metadata received via ZeroMQ notification"""
    object_id: str
    sensor_id: str
    timestamp: str
    width: int
    height: int
    exposure_time_ms: float
    wcs_header: str
    dtype: str = "float32"
    byte_offset: int = 0
    byte_size: int = 0


class PlasmaConsumer:
    """
    High-performance consumer for sensor frame streaming
    Maps Plasma shared memory directly into NumPy without copying (when available)
    Falls back to temp file transfer for compatibility
    """

    def __init__(
        self,
        plasma_socket: str = "/tmp/plasma",
        zmq_endpoint: str = "ipc:///tmp/sensor_notify.sock",
        temp_dir: str = "/tmp/iaf_frames"
    ):
        self.plasma_client = None
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        if PLASMA_AVAILABLE:
            try:
                self.plasma_client = plasma.connect(plasma_socket)
                logger.info(f"Plasma consumer connected: {plasma_socket}")
            except Exception as e:
                logger.warning(f"Could not connect to Plasma: {e}")
                logger.warning("Falling back to file-based transfer")

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(zmq_endpoint)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

        self.callbacks: List[Callable[[np.ndarray, SensorFrameMetadata], None]] = []

        logger.info(f"ZeroMQ subscription: {zmq_endpoint}")
        logger.info(f"Temp directory: {temp_dir}")

    def register_callback(self, callback: Callable[[np.ndarray, SensorFrameMetadata], None]):
        """Register frame processing callback (e.g., MPDAF streak detection)"""
        self.callbacks.append(callback)

    def run_event_loop(self):
        """Main event loop - blocks and processes frames"""
        logger.info("Starting Plasma consumer event loop...")
        logger.info("Waiting for frames from Go producer...")

        while True:
            try:
                # Wait for ZeroMQ notification (blocking with timeout)
                if not self.zmq_socket.poll(timeout=1000):
                    continue

                # Receive metadata notification (~200 bytes)
                message = self.zmq_socket.recv_string()
                metadata = SensorFrameMetadata(**json.loads(message))

                logger.info(f"Received notification for frame {metadata.object_id[:16]}...")

                # Retrieve frame data (from Plasma or temp file)
                frame_array = self._get_frame_data(metadata)
                if frame_array is None:
                    logger.warning(f"Could not retrieve frame {metadata.object_id}")
                    continue

                # Process via callbacks
                for callback in self.callbacks:
                    try:
                        callback(frame_array, metadata)
                    except Exception as e:
                        logger.error(f"Callback error: {e}", exc_info=True)

                # Cleanup temp file if used
                if not PLASMA_AVAILABLE or self.plasma_client is None:
                    self._cleanup_temp_file(metadata)

            except KeyboardInterrupt:
                logger.info("Shutting down Plasma consumer...")
                break
            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)

    def _get_frame_data(self, metadata: SensorFrameMetadata) -> Optional[np.ndarray]:
        """Retrieve frame data from Plasma Store or temp file"""
        try:
            if PLASMA_AVAILABLE and self.plasma_client is not None:
                # Try Plasma first (zero-copy)
                return self._get_from_plasma(metadata)
            else:
                # Fallback to temp file
                return self._get_from_temp_file(metadata)
        except Exception as e:
            logger.error(f"Frame retrieval error: {e}")
            return None

    def _get_from_plasma(self, metadata: SensorFrameMetadata) -> Optional[np.ndarray]:
        """Retrieve frame from Plasma Store (zero-copy)"""
        try:
            obj_id = plasma.ObjectID(bytes.fromhex(metadata.object_id))

            # Get object buffers - returns list of memoryview (zero-copy)
            buffers = self.plasma_client.get_buffers([obj_id])

            if not buffers:
                logger.warning(f"Object not found in Plasma: {metadata.object_id}")
                return None

            # Convert to NumPy array (zero-copy via buffer protocol)
            buffer = buffers[0]
            frame_array = np.frombuffer(buffer, dtype=metadata.dtype).copy()
            frame_array = frame_array.reshape((metadata.height, metadata.width))

            # Release Plasma object reference
            self.plasma_client.release(obj_id)

            return frame_array

        except Exception as e:
            logger.error(f"Plasma get error: {e}")
            return None

    def _get_from_temp_file(self, metadata: SensorFrameMetadata) -> Optional[np.ndarray]:
        """Fallback: Retrieve frame from temp file"""
        temp_path = os.path.join(self.temp_dir, f"frame_{metadata.object_id}.npy")

        # Wait for file to be written (max 5 seconds)
        for _ in range(50):
            if os.path.exists(temp_path):
                break
            time.sleep(0.1)

        if not os.path.exists(temp_path):
            logger.warning(f"Temp file not found: {temp_path}")
            return None

        try:
            # Load NumPy array
            frame_array = np.load(temp_path)
            return frame_array
        except Exception as e:
            logger.error(f"Error loading temp file: {e}")
            return None

    def _cleanup_temp_file(self, metadata: SensorFrameMetadata):
        """Remove temp file after processing"""
        temp_path = os.path.join(self.temp_dir, f"frame_{metadata.object_id}.npy")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def close(self):
        """Cleanup resources"""
        self.zmq_socket.close()
        self.zmq_context.term()
        if self.plasma_client:
            self.plasma_client.disconnect()


# Example MPDAF Integration
def mpdaf_streak_detection_callback(frame: np.ndarray, metadata: SensorFrameMetadata):
    """
    Example callback: Run MPDAF 3D matched filtering

    Args:
        frame: NumPy array (height, width)
        metadata: Frame capture metadata
    """
    from streak_detection import StreakDetector3D

    # Create MPDAF Image from NumPy array
    image = Image(data=frame)

    # Run streak detection
    detector = StreakDetector3D()
    detections = detector.detect_in_frame(image)

    logger.info(f"Detected {len(detections)} streaks in frame from {metadata.sensor_id}")

    # Send results back to Go via results socket
    results = {
        "sensor_id": metadata.sensor_id,
        "timestamp": metadata.timestamp,
        "detections": len(detections),
        "details": [d.__dict__ for d in detections]
    }

    # TODO: Send results via ZMQ PUSH socket to Go
    print(f"RESULTS: {json.dumps(results)}")


if __name__ == "__main__":
    # Example usage
    consumer = PlasmaConsumer()
    consumer.register_callback(mpdaf_streak_detection_callback)

    try:
        consumer.run_event_loop()
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    finally:
        consumer.close()
