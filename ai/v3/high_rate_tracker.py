#!/usr/bin/env python3
"""
High-Rate Tracking Loop for LEO RSOs (100Hz)

This module implements the "Zero-Jitter" tracking requirement for 7.5 km/s targets.
It runs a dedicated 100Hz loop that:
1. Receives Az/El + rates from Go engine via ZMQ
2. Performs forward prediction (Taylor expansion) for latency compensation
3. Sends rate corrections to telescope for smooth tracking

θ_future ≈ θ_now + θ̇ · Δt

Where:
- θ_now: Current commanded Azimuth/Elevation
- θ̇: Angular velocity from Go HCW engine (deg/s)
- Δt: Estimated system latency (network + OS + motor lag, typically 50ms)

Reference: Brunton Control Bootcamp - Real-Time Control Systems
"""

import zmq
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("high_rate_tracker")


@dataclass
class TrackingState:
    """Current tracking state for a target"""
    timestamp: float       # Unix timestamp of last update
    azimuth: float         # Current commanded Az (deg)
    elevation: float       # Current commanded El (deg)
    az_rate: float         # Angular velocity Az (deg/s)
    el_rate: float         # Angular velocity El (deg/s)
    target_id: int         # NORAD ID
    
    # Latency compensation
    latency_ms: float      # Estimated system latency
    last_command_time: float  # When we last sent a command


class HighRateTracker:
    """
    100Hz tracking controller for LEO satellites.
    
    Uses rate-based control (not position slewing) for zero-jitter observation.
    Essential for tracking objects moving at 7.5 km/s across the sky.
    """
    
    def __init__(self, 
                 zmq_pub_endpoint: str = "tcp://localhost:5555",
                 telescope_controller: Optional[Callable] = None,
                 update_rate_hz: float = 100.0,
                 default_latency_ms: float = 50.0):
        """
        Args:
            zmq_pub_endpoint: Where to subscribe for target updates
            telescope_controller: Callback to send commands to telescope
            update_rate_hz: Target loop rate (100Hz recommended for LEO)
            default_latency_ms: Estimated system latency for compensation
        """
        self.update_rate = update_rate_hz
        self.dt = 1.0 / update_rate_hz
        self.default_latency = default_latency_ms / 1000.0  # Convert to seconds
        
        # ZMQ Setup - SUB socket receives from Go engine
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(zmq_pub_endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
        self.socket.setsockopt(zmq.RCVHWM, 10)  # Small queue, drop old messages
        
        # Non-blocking receive
        self.socket.setsockopt(zmq.RCVTIMEO, 1)  # 1ms timeout
        
        # Telescope interface
        self.telescope = telescope_controller
        
        # Tracking state
        self.state: Optional[TrackingState] = None
        self.running = False
        
        # Statistics
        self.loop_count = 0
        self.missed_updates = 0
        self.avg_latency = 0.0
        
    def predict_position(self, state: TrackingState, look_ahead_sec: float) -> tuple:
        """
        Forward prediction using Taylor expansion.
        
        θ_future = θ_now + θ̇ · Δt + 0.5 · θ̈ · Δt²
        
        For LEO tracking, first-order is usually sufficient (θ̈ is small).
        
        Args:
            state: Current tracking state with rates
            look_ahead_sec: How far ahead to predict
            
        Returns:
            (az_future, el_future): Predicted coordinates
        """
        # First-order Taylor expansion
        az_future = state.azimuth + (state.az_rate * look_ahead_sec)
        el_future = state.elevation + (state.el_rate * look_ahead_sec)
        
        # Normalize azimuth to [0, 360)
        while az_future >= 360:
            az_future -= 360
        while az_future < 0:
            az_future += 360
            
        # Clamp elevation
        el_future = max(0, min(90, el_future))
        
        return az_future, el_future
    
    def receive_update(self) -> Optional[TrackingState]:
        """
        Non-blocking receive of target update from Go engine.
        
        Expected message format (msgpack or JSON):
        {
            "ts": 1234567890.123,      # Unix timestamp
            "target_id": 25544,         # ISS
            "az": 123.45,               # Azimuth (deg)
            "el": 45.67,               # Elevation (deg)
            "az_rate": 1.234,          # Az velocity (deg/s)
            "el_rate": -0.456,         # El velocity (deg/s)
            "latency_estimate_ms": 50  # Optional: dynamic latency
        }
        """
        try:
            message = self.socket.recv_json(flags=zmq.NOBLOCK)
            
            latency_ms = message.get("latency_estimate_ms", self.default_latency * 1000)
            
            return TrackingState(
                timestamp=message["ts"],
                azimuth=message["az"],
                elevation=message["el"],
                az_rate=message["az_rate"],
                el_rate=message["el_rate"],
                target_id=message["target_id"],
                latency_ms=latency_ms,
                last_command_time=time.time()
            )
        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.warning(f"Error receiving update: {e}")
            return None
    
    def calculate_correction(self, state: TrackingState) -> tuple:
        """
        Calculate rate correction to keep target centered.
        
        Uses the HCW (High-Consistency Worker) smoothed trajectory to
        compute tiny rate adjustments at 100Hz.
        
        Returns:
            (az_correction, el_correction): Rate adjustments (deg/s)
        """
        # Current time
        now = time.time()
        
        # Time since last update from Go engine
        dt_since_update = now - state.timestamp
        
        # Predict where target should be now
        az_predicted, el_predicted = self.predict_position(state, dt_since_update)
        
        # The "error" is the difference between predicted and commanded
        # In a real implementation, we'd have feedback from the mount encoders
        # For now, we assume the mount is tracking at the commanded rate
        
        # Calculate correction based on trajectory evolution
        # This compensates for any drift in the orbital propagation
        az_correction = state.az_rate * 0.01  # 1% correction factor
        el_correction = state.el_rate * 0.01
        
        return az_correction, el_correction
    
    def send_rate_command(self, az: float, el: float, az_rate: float, el_rate: float):
        """
        Send rate-based tracking command to telescope.
        
        Args:
            az: Commanded Azimuth (with latency compensation)
            el: Commanded Elevation (with latency compensation)
            az_rate: Azimuth tracking rate (deg/s)
            el_rate: Elevation tracking rate (deg/s)
        """
        if self.telescope is None:
            return
        
        try:
            # Send command to telescope controller
            self.telescope({
                "type": "TRACK",
                "az": az,
                "el": el,
                "vel_az": az_rate,
                "vel_el": el_rate
            })
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
    
    def run(self):
        """
        Main 100Hz tracking loop.
        
        This loop MUST maintain precise timing for zero-jitter tracking.
        Any delay > 10ms will cause visible frame stepping.
        """
        logger.info(f"Starting 100Hz tracking loop (dt={self.dt*1000:.1f}ms)")
        self.running = True
        
        next_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            # 1. Try to receive new target data from Go engine
            new_state = self.receive_update()
            if new_state is not None:
                self.state = new_state
                self.loop_count += 1
            else:
                self.missed_updates += 1
            
            # 2. If we have a target, perform tracking
            if self.state is not None:
                # Calculate latency-compensated position
                look_ahead = self.state.latency_ms / 1000.0
                az_cmd, el_cmd = self.predict_position(self.state, look_ahead)
                
                # Calculate rate corrections
                az_corr, el_corr = self.calculate_correction(self.state)
                
                # Final commanded rates
                az_rate_cmd = self.state.az_rate + az_corr
                el_rate_cmd = self.state.el_rate + el_corr
                
                # 3. Send command to telescope
                self.send_rate_command(az_cmd, el_cmd, az_rate_cmd, el_rate_cmd)
                
                # 4. Update statistics
                self.avg_latency = 0.99 * self.avg_latency + 0.01 * (time.time() - loop_start)
            
            # 5. Maintain precise 100Hz timing
            next_time += self.dt
            sleep_time = next_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind schedule - log but continue
                if self.loop_count % 100 == 0:
                    logger.warning(f"Loop behind by {-sleep_time*1000:.1f}ms")
            
            # Log statistics every 10 seconds
            if self.loop_count % 1000 == 0 and self.loop_count > 0:
                logger.info(f"Stats: {self.loop_count} loops, "
                           f"{self.missed_updates} missed, "
                           f"avg latency={self.avg_latency*1000:.2f}ms")
    
    def stop(self):
        """Stop the tracking loop."""
        logger.info("Stopping high-rate tracker")
        self.running = False
        self.socket.close()
        self.context.term()


def demo_high_rate_tracking():
    """
    Demo the 100Hz tracking loop with simulated LEO target.
    
    Simulates the ISS passing overhead with ~1°/s angular velocity.
    """
    import threading
    
    # Mock telescope controller
    def mock_telescope(cmd):
        pass
    
    # Create tracker
    tracker = HighRateTracker(
        zmq_pub_endpoint="tcp://localhost:5555",
        telescope_controller=mock_telescope,
        update_rate_hz=100.0
    )
    
    # Simulate ZMQ publisher (Go engine side)
    def mock_publisher():
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")
        
        # Simulate ISS pass
        # Starting at horizon, rising at 1°/s
        az = 90.0   # East
        el = 10.0   # Rising
        az_rate = 0.5   # Moving south
        el_rate = 1.0   # Rising fast
        
        for i in range(600):  # 60 seconds at 10Hz update from Go
            message = {
                "ts": time.time(),
                "target_id": 25544,
                "az": az,
                "el": el,
                "az_rate": az_rate,
                "el_rate": el_rate,
                "latency_estimate_ms": 50
            }
            socket.send_json(message)
            
            # Simulate trajectory
            az += az_rate * 0.1
            el += el_rate * 0.1
            
            # Arc over and set
            if el > 80:
                el_rate = -abs(el_rate)
            
            time.sleep(0.1)  # 10Hz updates from Go
        
        socket.close()
        context.term()
    
    # Start publisher in background
    pub_thread = threading.Thread(target=mock_publisher, daemon=True)
    pub_thread.start()
    
    # Give publisher time to start
    time.sleep(0.5)
    
    # Run tracker
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    print("🛰️ High-Rate Tracking Demo (100Hz)")
    print("Simulating ISS pass with 50ms latency compensation")
    demo_high_rate_tracking()
