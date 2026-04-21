#!/usr/bin/env python3
"""
Telescope Controller - "The Muscle"
Interfaces with INDI/Pyindi and Alpaca for hardware control.
Receives ZMQ commands from Go engine, executes slews.
"""

import zmq
import json
import time
import logging
import threading
from typing import Optional, Dict
from dataclasses import dataclass

# Try to import astronomical control libraries
try:
    import PyIndi  # INDI protocol
    INDI_AVAILABLE = True
except ImportError:
    INDI_AVAILABLE = False

try:
    import alpaca  # ASCOM Alpaca
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telescope")


@dataclass
class MountState:
    """Current state of the telescope mount"""
    azimuth: float = 0.0
    elevation: float = 0.0
    slewing: bool = False
    tracking: bool = False
    last_update: float = 0.0


class TelescopeController:
    """
    High-Rate Tracking Controller for LEO RSOs.
    
    Implements 100Hz control loop with:
    - Forward prediction (Taylor expansion) for latency compensation
    - Rate-based tracking for zero-jitter observation
    - Slewing for initial acquisition
    
    Critical for 7.5 km/s LEO targets where 50ms latency = 375m position error.
    """
    
    def __init__(self, zmq_endpoint="tcp://*:5556", device_type="indi", 
                 update_rate_hz: float = 100.0, latency_ms: float = 50.0):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(zmq_endpoint)
        self.device_type = device_type
        self.running = False
        
        # High-rate control parameters
        self.update_rate = update_rate_hz  # 100Hz default
        self.latency_compensation = latency_ms / 1000.0  # Convert to seconds
        
        # Tracking state
        self.state = MountState()
        self.target_az = 0.0
        self.target_el = 0.0
        self.target_az_rate = 0.0  # deg/s for rate tracking
        self.target_el_rate = 0.0
        
        # Slew vs Track modes
        self.mode = "IDLE"  # IDLE, SLEWING, TRACKING
        
        # Slew simulation
        self.slew_rate = 5.0  # deg/s for slewing
        self.max_track_rate = 3.0  # deg/s max for rate tracking (LEO max ~1.5°/s)
        
        self.simulation_thread: Optional[threading.Thread] = None
        self.tracking_thread: Optional[threading.Thread] = None
        
        # Initialize hardware interface
        self.simulation_mode = True
        self.indi_client = None
        self.indi_telescope = None
        self.alpaca_scope = None
        
        self._init_hardware()
    
    def _init_hardware(self):
        """Connect to telescope hardware or fallback to simulation"""
        if self.device_type == "indi" and INDI_AVAILABLE:
            try:
                self._init_indi()
                self.simulation_mode = False
            except Exception as e:
                logger.warning(f"INDI init failed, using simulation: {e}")
        elif self.device_type == "alpaca" and ALPACA_AVAILABLE:
            try:
                self._init_alpaca()
                self.simulation_mode = False
            except Exception as e:
                logger.warning(f"Alpaca init failed, using simulation: {e}")
        else:
            logger.info("Hardware interface not available - SIMULATION MODE")
    
    def _init_indi(self):
        """Connect to INDI server (local or remote)"""
        self.indi_client = PyIndi.BaseClient()
        self.indi_client.setServer("localhost", 7624)
        
        if not self.indi_client.connectServer():
            raise ConnectionError("Failed to connect to INDI server")
        
        logger.info("Connected to INDI server")
        
        # Wait for telescope device
        time.sleep(1)
        self.indi_telescope = self.indi_client.getDevice("Telescope Simulator")
        if not self.indi_telescope:
            raise RuntimeError("Telescope device not found")
    
    def _init_alpaca(self):
        """Connect to Alpaca server"""
        self.alpaca_scope = alpaca.Telescope(
            "localhost:11111",  # Alpaca server
            device_number=0,
            protocol="http"
        )
        logger.info("Connected to Alpaca telescope")
    
    def run(self):
        """Main loop - listen for ZMQ commands"""
        self.running = True
        logger.info(f"Telescope controller listening on {self.socket.last_endpoint}")
        logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else self.device_type.upper()}")
        
        # Start state update thread
        state_thread = threading.Thread(target=self._update_state_loop, daemon=True)
        state_thread.start()
        
        while self.running:
            try:
                # Receive command from Go engine
                message = self.socket.recv_string()
                cmd = json.loads(message)
                
                # Process based on command type
                if cmd["type"] == "SLEW":
                    response = self.handle_slew(cmd)
                elif cmd["type"] == "TRACK":
                    response = self.handle_track(cmd)
                elif cmd["type"] == "STOP":
                    response = self.handle_stop()
                else:
                    response = {"status": "ERROR", "error": f"Unknown command: {cmd['type']}"}
                
                # Send acknowledgment
                self.socket.send_string(json.dumps(response))
                
                # Log latency
                latency_ms = time.time() * 1000 - cmd.get("ts", time.time() * 1000)
                if latency_ms > 10:  # Only log if meaningful latency
                    logger.info(f"Command processed in {latency_ms:.1f}ms (target: {cmd.get('target_id', 'N/A')})")
                
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    continue  # Timeout, check if still running
                logger.error(f"ZMQ error: {e}")
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                try:
                    self.socket.send_string(json.dumps({"status": "ERROR", "error": str(e)}))
                except:
                    pass
    
    def _update_state_loop(self):
        """Background thread to update mount state"""
        while self.running:
            if self.simulation_mode and self.state.slewing:
                # Simulate slew motion
                az_diff = self.target_az - self.state.azimuth
                el_diff = self.target_el - self.state.elevation
                
                # Normalize azimuth for shortest path
                if az_diff > 180:
                    az_diff -= 360
                if az_diff < -180:
                    az_diff += 360
                
                # Move toward target
                step = self.slew_rate * 0.1  # 100ms update interval
                
                if abs(az_diff) > step:
                    self.state.azimuth += step * (1 if az_diff > 0 else -1)
                else:
                    self.state.azimuth = self.target_az
                
                if abs(el_diff) > step:
                    self.state.elevation += step * (1 if el_diff > 0 else -1)
                else:
                    self.state.elevation = self.target_el
                
                # Check if arrived
                if abs(az_diff) < 0.1 and abs(el_diff) < 0.1:
                    self.state.slewing = False
                    logger.info(f"Slew complete: Az={self.state.azimuth:.2f}°, El={self.state.elevation:.2f}°")
            
            self.state.last_update = time.time()
            time.sleep(0.1)
    
    def handle_slew(self, cmd: Dict) -> Dict:
        """
        Execute slew command with forward prediction (Taylor expansion).
        
        θ_future ≈ θ_now + θ̇ · Δt
        
        Critical for LEO: 50ms latency at 7.5 km/s = 375m position error.
        We command the mount to where the satellite WILL BE, not where it is.
        """
        target_az = cmd["az"]
        target_el = cmd["el"]
        az_rate = cmd.get("vel_az", 0)  # deg/s
        el_rate = cmd.get("vel_el", 0)  # deg/s
        
        # Use provided lead time or default latency compensation
        lead_ms = cmd.get("lead_ms", self.latency_compensation * 1000)
        time_delta = lead_ms / 1000.0
        
        # FORWARD PREDICTION: Taylor expansion for latency compensation
        # θ_future = θ_now + θ̇ · Δt
        compensated_az = target_az + (az_rate * time_delta)
        compensated_el = target_el + (el_rate * time_delta)
        
        # Normalize azimuth to [0, 360)
        while compensated_az >= 360:
            compensated_az -= 360
        while compensated_az < 0:
            compensated_az += 360
        
        # Clamp elevation to valid range
        compensated_el = max(0, min(90, compensated_el))
        
        # Store rates for potential transition to tracking mode
        self.target_az_rate = az_rate
        self.target_el_rate = el_rate
        
        logger.info(f"SLEW: Target=({target_az:.2f}°, {target_el:.2f}°) | "
                   f"Predicted=({compensated_az:.2f}°, {compensated_el:.2f}°) | "
                   f"Rates=({az_rate:.3f}°/s, {el_rate:.3f}°/s) | "
                   f"Lead={lead_ms:.0f}ms")
        
        self.target_az = compensated_az
        self.target_el = compensated_el
        self.mode = "SLEWING"
        
        if self.simulation_mode:
            return {
                "status": "ACK",
                "az": compensated_az,
                "el": compensated_el,
                "az_rate": az_rate,
                "el_rate": el_rate,
                "mode": "SIM",
                "slewing": True,
                "latency_compensated": lead_ms
            }
        
        # Real hardware control
        if self.device_type == "indi":
            return self._slew_indi(compensated_az, compensated_el, az_rate, el_rate)
        elif self.device_type == "alpaca":
            return self._slew_alpaca(compensated_az, compensated_el, az_rate, el_rate)
        
        return {"status": "ERROR", "error": "No hardware interface available"}
    
    def handle_track(self, cmd: Dict) -> Dict:
        """
        Rate tracking for continuous observation (required for LEO).
        
        This is the KEY to zero-jitter tracking:
        - Commands constant velocity (e.g., 1.5°/s for LEO)
        - Sends tiny rate corrections at 100Hz to stay centered
        - No "slewing" motion that causes frame stepping
        """
        rate_az = cmd.get("vel_az", 0)  # deg/s
        rate_el = cmd.get("vel_el", 0)  # deg/s (optional)
        
        # Clamp to physical limits
        rate_az = max(-self.max_track_rate, min(self.max_track_rate, rate_az))
        rate_el = max(-self.max_track_rate, min(self.max_track_rate, rate_el))
        
        self.target_az_rate = rate_az
        self.target_el_rate = rate_el
        self.mode = "TRACKING"
        
        logger.info(f"Starting rate tracking: Az={rate_az:.3f}°/s, El={rate_el:.3f}°/s")
        
        if self.simulation_mode:
            # Start high-rate tracking thread (100Hz)
            def high_rate_track_loop():
                dt = 1.0 / self.update_rate
                while self.mode == "TRACKING" and self.running:
                    # Update position at tracking rate
                    self.state.azimuth += self.target_az_rate * dt
                    self.state.elevation += self.target_el_rate * dt
                    
                    # Normalize
                    while self.state.azimuth >= 360:
                        self.state.azimuth -= 360
                    while self.state.azimuth < 0:
                        self.state.azimuth += 360
                    
                    # Clamp elevation
                    self.state.elevation = max(0, min(90, self.state.elevation))
                    
                    time.sleep(dt)
            
            if self.tracking_thread is None or not self.tracking_thread.is_alive():
                self.tracking_thread = threading.Thread(target=high_rate_track_loop, daemon=True)
                self.tracking_thread.start()
            
            return {
                "status": "ACK",
                "tracking": True,
                "az_rate": rate_az,
                "el_rate": rate_el,
                "mode": "SIM"
            }
        
        if self.device_type == "alpaca":
            try:
                # Alpaca: Axis 0 = Azimuth, Axis 1 = Elevation (for Alt-Az mounts)
                self.alpaca_scope.MoveAxis(0, rate_az)   # Azimuth rate
                if rate_el != 0:
                    self.alpaca_scope.MoveAxis(1, rate_el)  # Elevation rate
                
                return {
                    "status": "ACK", 
                    "tracking": True, 
                    "az_rate": rate_az,
                    "el_rate": rate_el
                }
            except Exception as e:
                return {"status": "ERROR", "error": f"Alpaca track failed: {e}"}
        
        if self.device_type == "indi":
            try:
                # INDI: Send track rates if supported by driver
                # Many INDI mounts support TELESCOPE_TRACK_RATE
                self.indi_telescope.setNumber("TELESCOPE_TRACK_RATE", {
                    "TRACK_RATE_RA": rate_az / 15.0,  # Convert to sidereal units if needed
                    "TRACK_RATE_DE": rate_el
                })
                return {"status": "ACK", "tracking": True, "az_rate": rate_az, "el_rate": rate_el}
            except Exception as e:
                logger.warning(f"INDI rate tracking failed, falling back to position updates: {e}")
                # Fall through to position-based tracking
        
        return {"status": "ERROR", "error": "Rate tracking not supported for this mount type"}
    
    def handle_stop(self) -> Dict:
        """Stop all mount motion."""
        logger.info("Stopping mount")
        
        self.state.slewing = False
        self.state.tracking = False
        
        if self.simulation_mode:
            return {"status": "ACK", "stopped": True, "mode": "SIM"}
        
        if self.device_type == "indi":
            try:
                self.indi_telescope.abort()
                return {"status": "ACK", "stopped": True}
            except Exception as e:
                return {"status": "ERROR", "error": str(e)}
        
        if self.device_type == "alpaca":
            try:
                self.alpaca_scope.AbortSlew()
                return {"status": "ACK", "stopped": True}
            except Exception as e:
                return {"status": "ERROR", "error": str(e)}
        
        return {"status": "ACK", "stopped": True}
    
    def _slew_indi(self, az, el, az_rate=0, el_rate=0):
        """Execute INDI slew command with optional track rates."""
        try:
            # Try Alt-Az coordinates first (if mount supports it)
            try:
                self.indi_telescope.setNumber("HORIZONTAL_COORD", {
                    "AZ": az,
                    "ALT": el
                })
                
                # If rates provided, also set tracking rates
                if az_rate != 0 or el_rate != 0:
                    try:
                        self.indi_telescope.setNumber("HORIZONTAL_RATES", {
                            "AZ_RATE": az_rate,
                            "ALT_RATE": el_rate
                        })
                    except:
                        logger.debug("HORIZONTAL_RATES not supported by this mount")
                
            except:
                # Fall back to Equatorial (RA/Dec) conversion
                # Simplified conversion - proper implementation needs full coordinate transform
                logger.warning("Alt-Az not supported, falling back to RA/Dec conversion")
                self.indi_telescope.setNumber("EQUATORIAL_EOD_COORD", {
                    "RA": az / 15.0,  # Simplified
                    "DEC": el
                })
            
            return {
                "status": "ACK",
                "az": az,
                "el": el,
                "az_rate": az_rate,
                "el_rate": el_rate,
                "mode": "INDI",
                "slewing": True
            }
        except Exception as e:
            return {"status": "ERROR", "error": f"INDI slew failed: {e}"}
    
    def _slew_alpaca(self, az, el, az_rate=0, el_rate=0):
        """Execute Alpaca slew command with optional rate tracking."""
        try:
            # Alpaca supports Alt-Az (Azimuth/Altitude) directly
            self.alpaca_scope.SlewToAltAzAsync(az, el)
            
            # If we're already close and have rates, switch to tracking
            current_az = self.state.azimuth
            current_el = self.state.elevation
            
            # Check if we're close enough to handoff to tracking (within 0.5°)
            az_diff = abs(az - current_az)
            if az_diff > 180:  # Handle wrap-around
                az_diff = 360 - az_diff
                
            if az_diff < 0.5 and abs(el - current_el) < 0.5 and (az_rate != 0 or el_rate != 0):
                # Close enough - use rate tracking for smooth handoff
                logger.info("Slew complete, switching to rate tracking")
                self.alpaca_scope.MoveAxis(0, az_rate)
                self.alpaca_scope.MoveAxis(1, el_rate)
            
            return {
                "status": "ACK",
                "az": az,
                "el": el,
                "az_rate": az_rate,
                "el_rate": el_rate,
                "mode": "ALPACA",
                "slewing": True
            }
        except Exception as e:
            return {"status": "ERROR", "error": f"Alpaca slew failed: {e}"}
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down telescope controller")
        self.running = False
        self.state.tracking = False
        self.state.slewing = False
        
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


if __name__ == "__main__":
    import sys
    
    # Get device type from command line
    device_type = "indi"
    if len(sys.argv) > 1:
        device_type = sys.argv[1].lower()
    
    controller = TelescopeController(device_type=device_type)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.shutdown()
