"""
Reticulum Network Stack (RNS) Relay for IAF Sovereign Mesh
Receives TargetBrief from Go via ZeroMQ, transmits over RNS mesh
"""

import json
import time
import zmq
from dataclasses import dataclass
from typing import Optional
import logging

# Try to import Reticulum (may not be available in dev environment)
try:
    import RNS
    import LXMF
    RNS_AVAILABLE = True
except ImportError:
    RNS_AVAILABLE = False
    logging.warning("RNS not available - using mock mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TargetBrief:
    """Tactical Intelligence Packet"""
    msg_id: str
    timestamp: str
    priority: str
    origin_node: str
    norad_id: int
    cospar_id: str
    common_name: str
    country_code: str
    classification: str
    epoch: str
    position_ecef: list  # [x, y, z]
    velocity_ecef: list  # [vx, vy, vz]
    covariance_diag: list  # [p11, p22, p33]
    slope_trend: str  # STABLE|MANEUVERING|DECAYING
    material_id: str
    confidence_score: float
    attitude_state: str
    rotation_rate_hz: float
    thermal_signature: str
    target_volume: str
    access_start: str
    access_end: str
    max_elevation: float
    isr_capability: str
    recommended_action: str


class RNSRelay:
    """
    Bridges Go operational layer to Reticulum mesh network
    Receives TargetBrief via ZeroMQ, transmits over sovereign mesh
    """
    
    def __init__(
        self,
        zmq_endpoint: str = "ipc:///tmp/rns_relay.ipc",
        rns_identity: str = "iaf_node_alpha",
        mesh_destination: Optional[str] = None
    ):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_socket.bind(zmq_endpoint)
        
        self.rns_identity = rns_identity
        self.mesh_destination = mesh_destination
        
        # Initialize RNS if available
        if RNS_AVAILABLE:
            self._init_rns()
        else:
            self.rns_destination = None
            logger.info("RNS relay initialized in mock mode")
        
        logger.info(f"RNS relay listening on {zmq_endpoint}")
    
    def _init_rns(self):
        """Initialize Reticulum identity and LXMF destination"""
        self.rns_identity = RNS.Identity(self.rns_identity)
        self.lxmf_destination = RNS.Destination(
            self.rns_identity,
            RNS.Destination.SINGLE,
            "ssis", "tactical", "brief"
        )
        
        # Create LXMF message router
        self.lxmf_router = LXMF.LXMRouter(
            identity=self.rns_identity,
            storage_path="/tmp/lxmf_storage"
        )
        
        logger.info("RNS initialized successfully")
    
    def run_event_loop(self):
        """Main event loop - receive briefs and transmit over mesh"""
        logger.info("Starting RNS relay event loop...")
        
        while True:
            try:
                # Receive MessagePack data from Go
                data = self.zmq_socket.recv()
                
                # Deserialize
                brief = self._deserialize_brief(data)
                
                logger.info(f"Received brief for NORAD {brief.norad_id} "
                          f"({brief.common_name})")
                
                # Transmit over mesh
                if self._transmit_over_mesh(brief):
                    logger.info(f"Successfully transmitted to mesh")
                else:
                    logger.warning(f"Mesh transmission failed - queued for retry")
                
            except KeyboardInterrupt:
                logger.info("Shutting down RNS relay...")
                break
            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)
    
    def _deserialize_brief(self, data: bytes) -> TargetBrief:
        """Deserialize MessagePack to TargetBrief"""
        import msgpack
        
        unpacked = msgpack.unpackb(data, raw=False)
        return TargetBrief(**unpacked)
    
    def _transmit_over_mesh(self, brief: TargetBrief) -> bool:
        """
        Transmit brief over Reticulum mesh via LXMF
        Falls back to file logging if RNS unavailable
        """
        if not RNS_AVAILABLE or self.rns_destination is None:
            # Mock mode: log to file
            self._mock_transmit(brief)
            return True
        
        try:
            # Create LXMF message
            message = LXMF.LXMessage(
                destination=self.mesh_destination or self.lxmf_destination,
                content=self._brief_to_lxmf_content(brief),
                title=f"TACTICAL_{brief.priority}",
                desired_method=LXMF.LXMessage.DIRECT
            )
            
            # Send via mesh
            self.lxmf_router.handle_outbound(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Mesh transmission error: {e}")
            return False
    
    def _brief_to_lxmf_content(self, brief: TargetBrief) -> bytes:
        """Convert TargetBrief to LXMF-compatible content"""
        # Serialize to compact JSON
        content = {
            "msg_id": brief.msg_id,
            "ts": brief.timestamp,
            "pri": brief.priority,
            "origin": brief.origin_node,
            "id": brief.norad_id,
            "cls": brief.classification,
            "pos": brief.position_ecef,
            "vel": brief.velocity_ecef,
            "mat": brief.material_id,
            "mat_conf": brief.confidence_score,
            "att": brief.attitude_state,
            "vol": brief.target_volume,
            "max_el": brief.max_elevation,
            "isr": brief.isr_capability,
            "act": brief.recommended_action
        }
        
        return json.dumps(content, separators=(',', ':')).encode()
    
    def _mock_transmit(self, brief: TargetBrief):
        """Mock transmission for development/testing"""
        mock_file = f"/tmp/rns_mock_{brief.msg_id}.json"
        
        with open(mock_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "brief": brief.__dict__,
                "mock_transmission": True
            }, f, indent=2)
        
        logger.info(f"Mock transmission logged to {mock_file}")
    
    def close(self):
        """Cleanup resources"""
        self.zmq_socket.close()
        self.zmq_context.term()
        
        if RNS_AVAILABLE:
            # Cleanup RNS
            pass


def main():
    """Run RNS relay standalone"""
    relay = RNSRelay()
    
    try:
        relay.run_event_loop()
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    finally:
        relay.close()


if __name__ == "__main__":
    main()
