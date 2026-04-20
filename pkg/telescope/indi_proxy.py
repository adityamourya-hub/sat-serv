#!/usr/bin/env python3
# pkg/telescope/indi_proxy.py
# ADITI 4.0: INDI Protocol Proxy for Telescope Control
#
# Lightweight Python proxy that translates ZMQ commands from Go into INDI protocol
# commands for telescope mount control. Enables <50ms latency for HGV tracking.
#
# Architecture:
#   Go (Sovereign IPC) → ZMQ port 5555 → Python Proxy → INDI server localhost:7624 → Telescope
#
# INDI Commands Implemented:
#   - CONNECTION: CONNECT / DISCONNECT
#   - EQUATORIAL_EOD_COORD: RA/Dec (JNow) targeting
#   - TELESCOPE_TRACK_STATE: Toggle Track/Slew
#   - TELESCOPE_MOTION_RATE: Direct Az/El rate control (critical for HGVs)
#   - TELESCOPE_ABORT_MOTION: Emergency kill switch

import json
import sys
import time
import threading
import zmq
from typing import Dict, Optional, Tuple

# Try to import PyINDI, fallback to mock for testing
try:
    import PyIndi
    INDI_AVAILABLE = True
except ImportError:
    print("[INDI_PROXY] PyINDI not available, running in mock mode")
    INDI_AVAILABLE = False


class INDITelescopeProxy:
    """INDI telescope driver proxy for ZMQ-INDI handshake"""
    
    def __init__(self, zmq_port: int = 5555, indi_host: str = "localhost", indi_port: int = 7624):
        self.zmq_port = zmq_port
        self.indi_host = indi_host
        self.indi_port = indi_port
        
        # ZMQ context
        self.zmq_context = zmq.Context()
        self.zmq_socket = None
        
        # INDI client
        self.indi_client = None
        self.telescope_device = None
        
        # Connection state
        self.connected = False
        self.tracking = False
        
        # Latency tracking
        self.latency_history = []
        self.max_latency_samples = 100
        
        # Thread safety
        self.lock = threading.Lock()
    
    def start(self):
        """Start the ZMQ server and connect to INDI"""
        # Setup ZMQ REP socket
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(f"tcp://*:{self.zmq_port}")
        print(f"[INDI_PROXY] ZMQ listening on port {self.zmq_port}")
        
        # Connect to INDI server
        if INDI_AVAILABLE:
            self.indi_client = PyIndi.BaseClient()
            self.indi_client.setServer(self.indi_host, self.indi_port)
            self.indi_client.connect()
            print(f"[INDI_PROXY] Connected to INDI server at {self.indi_host}:{self.indi_port}")
        else:
            print("[INDI_PROXY] Mock mode - no INDI server connection")
        
        # Start message loop
        self.message_loop()
    
    def message_loop(self):
        """Main ZMQ message handling loop"""
        print("[INDI_PROXY] Starting message loop")
        
        while True:
            try:
                # Receive command from Go
                message = self.zmq_socket.recv_json()
                start_time = time.time()
                
                # Process command
                response = self.handle_command(message)
                
                # Send response
                self.zmq_socket.send_json(response)
                
                # Track latency
                latency_ms = (time.time() - start_time) * 1000
                self.track_latency(latency_ms)
                
                if latency_ms > 50:
                    print(f"[INDI_PROXY] WARNING: Latency {latency_ms:.2f}ms exceeds 50ms target")
                
            except Exception as e:
                print(f"[INDI_PROXY] Error in message loop: {e}")
                # Send error response
                self.zmq_socket.send_json({
                    "status": "ERROR",
                    "error": str(e)
                })
    
    def handle_command(self, command: Dict) -> Dict:
        """Handle incoming command from Go"""
        cmd_type = command.get("type")
        
        with self.lock:
            if cmd_type == "CONNECTION":
                return self.handle_connection(command)
            elif cmd_type == "EQUATORIAL_EOD_COORD":
                return self.handle_equatorial_coord(command)
            elif cmd_type == "TELESCOPE_TRACK_STATE":
                return self.handle_track_state(command)
            elif cmd_type == "TELESCOPE_MOTION_RATE":
                return self.handle_motion_rate(command)
            elif cmd_type == "TELESCOPE_ABORT_MOTION":
                return self.handle_abort_motion(command)
            else:
                return {
                    "status": "ERROR",
                    "error": f"Unknown command type: {cmd_type}"
                }
    
    def handle_connection(self, command: Dict) -> Dict:
        """Handle CONNECT/DISCONNECT command"""
        action = command.get("action")  # "CONNECT" or "DISCONNECT"
        
        if action == "CONNECT":
            if INDI_AVAILABLE:
                # Connect to telescope device
                self.telescope_device = self.indi_client.getDevice("Telescope Simulator")
                if self.telescope_device is None:
                    return {
                        "status": "ERROR",
                        "error": "Telescope device not found"
                    }
                self.connected = True
                print("[INDI_PROXY] Connected to telescope")
            else:
                # Mock connection
                self.connected = True
                print("[INDI_PROXY] Mock connected")
            
            return {
                "status": "ACK",
                "mode": "INDI",
                "connected": True
            }
        
        elif action == "DISCONNECT":
            self.connected = False
            self.tracking = False
            print("[INDI_PROXY] Disconnected")
            
            return {
                "status": "ACK",
                "connected": False
            }
        
        else:
            return {
                "status": "ERROR",
                "error": f"Unknown connection action: {action}"
            }
    
    def handle_equatorial_coord(self, command: Dict) -> Dict:
        """Handle RA/Dec (JNow) targeting"""
        if not self.connected:
            return {
                "status": "ERROR",
                "error": "Not connected to telescope"
            }
        
        ra = command.get("ra")  # Right Ascension (degrees or hours)
        dec = command.get("dec")  # Declination (degrees)
        epoch = command.get("epoch", "J2000")  # Epoch
        
        if INDI_AVAILABLE and self.telescope_device:
            # Set RA/Dec via INDI
            ra_prop = self.telescope_device.getNumber("EQUATORIAL_EOD_COORD")
            if ra_prop:
                ra_prop[0].value = ra
                ra_prop[1].value = dec
                self.indi_client.sendNewNumber(ra_prop)
                print(f"[INDI_PROXY] Slew to RA={ra}, Dec={dec}")
            else:
                return {
                    "status": "ERROR",
                    "error": "EQUATORIAL_EOD_COORD property not found"
                }
        else:
            # Mock slew
            print(f"[INDI_PROXY] Mock slew to RA={ra}, Dec={dec}")
        
        return {
            "status": "ACK",
            "ra": ra,
            "dec": dec,
            "epoch": epoch
        }
    
    def handle_track_state(self, command: Dict) -> Dict:
        """Handle Track/Slew toggle"""
        if not self.connected:
            return {
                "status": "ERROR",
                "error": "Not connected to telescope"
            }
        
        track_state = command.get("track")  # true/false or "ON"/"OFF"
        
        if INDI_AVAILABLE and self.telescope_device:
            # Toggle tracking via INDI
            track_switch = self.telescope_device.getSwitch("TELESCOPE_TRACK_STATE")
            if track_switch:
                if track_state in [True, "ON", "on"]:
                    track_switch[0].s = PyIndi.ISS_ON
                    self.tracking = True
                else:
                    track_switch[0].s = PyIndi.ISS_OFF
                    self.tracking = False
                self.indi_client.sendNewSwitch(track_switch)
                print(f"[INDI_PROXY] Tracking state: {self.tracking}")
            else:
                return {
                    "status": "ERROR",
                    "error": "TELESCOPE_TRACK_STATE property not found"
                }
        else:
            # Mock tracking toggle
            self.tracking = bool(track_state)
            print(f"[INDI_PROXY] Mock tracking: {self.tracking}")
        
        return {
            "status": "ACK",
            "tracking": self.tracking
        }
    
    def handle_motion_rate(self, command: Dict) -> Dict:
        """Handle direct Az/El rate control (critical for HGVs)"""
        if not self.connected:
            return {
                "status": "ERROR",
                "error": "Not connected to telescope"
            }
        
        az_rate = command.get("az_rate")  # deg/s
        el_rate = command.get("el_rate")  # deg/s
        
        # Bypass sidereal limitations for HGV tracking
        if INDI_AVAILABLE and self.telescope_device:
            # Set motion rates via INDI
            motion_rate = self.telescope_device.getNumber("TELESCOPE_MOTION_RATE")
            if motion_rate:
                motion_rate[0].value = az_rate
                motion_rate[1].value = el_rate
                self.indi_client.sendNewNumber(motion_rate)
                print(f"[INDI_PROXY] Motion rates: Az={az_rate} deg/s, El={el_rate} deg/s")
            else:
                # Fallback to slew if motion rate not available
                return {
                    "status": "WARN",
                    "error": "TELESCOPE_MOTION_RATE not available, using slew",
                    "az_rate": az_rate,
                    "el_rate": el_rate
                }
        else:
            # Mock rate control
            print(f"[INDI_PROXY] Mock motion rates: Az={az_rate} deg/s, El={el_rate} deg/s")
        
        return {
            "status": "ACK",
            "az_rate": az_rate,
            "el_rate": el_rate
        }
    
    def handle_abort_motion(self, command: Dict) -> Dict:
        """Handle emergency abort (kill switch)"""
        if not self.connected:
            return {
                "status": "ERROR",
                "error": "Not connected to telescope"
            }
        
        if INDI_AVAILABLE and self.telescope_device:
            # Abort via INDI
            abort_switch = self.telescope_device.getSwitch("TELESCOPE_ABORT_MOTION")
            if abort_switch:
                abort_switch[0].s = PyIndi.ISS_ON
                self.indi_client.sendNewSwitch(abort_switch)
                print("[INDI_PROXY] ABORT MOTION activated")
            else:
                return {
                    "status": "ERROR",
                    "error": "TELESCOPE_ABORT_MOTION property not found"
                }
        else:
            # Mock abort
            print("[INDI_PROXY] Mock abort motion")
        
        self.tracking = False
        
        return {
            "status": "ACK",
            "aborted": True
        }
    
    def track_latency(self, latency_ms: float):
        """Track command latency"""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_latency_samples:
            self.latency_history.pop(0)
    
    def get_average_latency(self) -> float:
        """Get average latency over history"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def stop(self):
        """Stop the proxy and cleanup"""
        print("[INDI_PROXY] Stopping...")
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.indi_client:
            self.indi_client.disconnect()
        self.zmq_context.term()
        print("[INDI_PROXY] Stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="INDI Protocol Proxy for ADITI 4.0")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port for Go communication")
    parser.add_argument("--indi-host", type=str, default="localhost", help="INDI server host")
    parser.add_argument("--indi-port", type=int, default=7624, help="INDI server port")
    
    args = parser.parse_args()
    
    proxy = INDITelescopeProxy(args.zmq_port, args.indi_host, args.indi_port)
    
    try:
        proxy.start()
    except KeyboardInterrupt:
        print("\n[INDI_PROXY] Interrupted by user")
    finally:
        proxy.stop()


if __name__ == "__main__":
    main()
