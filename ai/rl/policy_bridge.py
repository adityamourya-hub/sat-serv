"""
RL Policy Bridge - Python-side ZMQ REP server for Actor-Critic policy head
Receives state vectors from Go engine, runs PolicyHead inference, returns actions
"""

import zmq
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class PolicyHead(nn.Module):
    """
    Actor-Critic architecture for autonomous SSA decision-making
    Input: State vector (position, velocity, intent, threat indicators)
    Output: Action probabilities (Actor) + State value (Critic)
    
    Features Physics-Constrained Entropy Bonus:
    - Entropy weight β scales with Kalman Filter uncertainty
    - Prevents over-exploration in stable orbits
    - Forces exploration during high-uncertainty events
    """
    
    def __init__(self, state_dim: int = 15, action_dim: int = 6, hidden_dim: int = 128):
        super(PolicyHead, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function) - estimates state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Physics-Constrained Entropy parameters
        self.base_entropy_weight = 0.01  # β₀
        self.uncertainty_scale = 1.0      # k (normalization factor)
        self.max_entropy_weight = 0.1     # Cap to prevent over-exploration
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Actor-Critic network
        
        Args:
            state: Input state vector [batch_size, state_dim]
            
        Returns:
            action_probs: Probability distribution over actions [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy
        
        Args:
            state: Input state vector as numpy array
            deterministic: If True, select argmax; otherwise sample
            
        Returns:
            action: Selected action index
            action_prob: Probability of selected action
            value: State value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.forward(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()
            
            action_prob = action_probs[0, action].item()
            value = value.item()
            
        return action, action_prob, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, kalman_uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for policy gradient training
        
        Args:
            state: Batch of states
            action: Batch of actions taken
            kalman_uncertainty: Kalman Filter position covariance trace [batch_size]
            
        Returns:
            log_probs: Log probability of actions
            values: State value estimates
            entropy: Policy entropy for exploration bonus
        """
        action_probs, values = self.forward(state)
        
        # Log probability of taken actions
        log_probs = torch.log(action_probs.gather(1, action.unsqueeze(1)))
        
        # Policy entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        # Apply physics-constrained entropy scaling if uncertainty provided
        if kalman_uncertainty is not None:
            entropy_weight = self.compute_dynamic_entropy_weight(kalman_uncertainty)
            entropy = entropy * entropy_weight
        
        return log_probs, values.squeeze(1), entropy
    
    def compute_dynamic_entropy_weight(self, kalman_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic entropy weight based on Kalman Filter uncertainty
        
        β(t) = β₀ * exp(Trace(Σ_pos) / k)
        
        Args:
            kalman_uncertainty: Trace of position covariance [batch_size]
            
        Returns:
            entropy_weight: Scaled entropy weights [batch_size]
        """
        # Scale uncertainty by normalization factor
        scaled_uncertainty = kalman_uncertainty / self.uncertainty_scale
        
        # Exponential scaling: higher uncertainty → higher entropy weight
        entropy_weight = self.base_entropy_weight * torch.exp(scaled_uncertainty)
        
        # Cap at maximum to prevent over-exploration
        entropy_weight = torch.clamp(entropy_weight, max=self.max_entropy_weight)
        
        return entropy_weight


class PolicyBridge:
    """
    ZMQ bridge server that receives state vectors from Go and returns RL actions
    """
    
    def __init__(self, endpoint: str = "ipc:///tmp/policy_bridge.ipc", model_path: Optional[str] = None):
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # Initialize policy network
        self.policy = PolicyHead(state_dim=15, action_dim=6)
        
        # Load pretrained weights if available
        if model_path:
            self.load_model(model_path)
        
        self.running = False
        self.request_count = 0
        
    def load_model(self, model_path: str):
        """Load pretrained policy weights"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"[POLICY] Loaded model from {model_path}")
        except Exception as e:
            print(f"[POLICY] Failed to load model: {e}, using random initialization")
    
    def save_model(self, model_path: str):
        """Save current policy weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
        }, model_path)
        print(f"[POLICY] Saved model to {model_path}")
    
    def start(self):
        """Start ZMQ server and listen for requests"""
        self.socket.bind(self.endpoint)
        self.running = True
        
        print(f"[POLICY] Bridge active at {self.endpoint}")
        print("[POLICY] Awaiting state vectors from Go engine...")
        
        while self.running:
            try:
                # Receive state vector from Go
                message = self.socket.recv_string()
                self.request_count += 1
                
                # Parse state
                state_dict = json.loads(message)
                state_vector = self._dict_to_vector(state_dict)
                
                # Get action from policy
                action, action_prob, value = self.policy.get_action(state_vector)
                
                # Calculate entropy
                entropy = self._calculate_entropy(state_vector)
                
                # Prepare response
                response = {
                    'action': action,
                    'action_prob': action_prob,
                    'value': value,
                    'entropy': entropy,
                    'timestamp': int(time.time() * 1e9)  # nanoseconds
                }
                
                # Send response back to Go
                self.socket.send_string(json.dumps(response))
                
                # Log high-value decisions
                if value < 0.2:  # Low value = high risk
                    print(f"[POLICY] WARNING: Low state value ({value:.3f}) detected")
                    print(f"[POLICY] Action: {self._action_to_name(action)} (prob: {action_prob:.3f})")
                
            except zmq.ZMQError as e:
                print(f"[POLICY] ZMQ error: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"[POLICY] JSON decode error: {e}")
                # Send error response
                self.socket.send_string(json.dumps({'error': str(e)}))
            except Exception as e:
                print(f"[POLICY] Error processing request: {e}")
                self.socket.send_string(json.dumps({'error': str(e)}))
    
    def stop(self):
        """Stop the ZMQ server"""
        self.running = False
        self.socket.close()
        self.context.term()
        print("[POLICY] Bridge stopped")
    
    def _dict_to_vector(self, state_dict: Dict) -> np.ndarray:
        """Convert state dictionary to numpy vector"""
        # Extract fields in consistent order
        position = state_dict.get('position', [0.0, 0.0, 0.0])
        velocity = state_dict.get('velocity', [0.0, 0.0, 0.0])
        intent = state_dict.get('intent', 0)
        confidence = state_dict.get('confidence', 0.0)
        power_estimate = state_dict.get('power_estimate', -100.0)
        risk_score = state_dict.get('risk_score', 0.0)
        hcw_residual = state_dict.get('hcw_residual', 0.0)
        delta_v_recent = state_dict.get('delta_v_recent', 0.0)
        time_since_custody = state_dict.get('time_since_custody', 0.0)
        frames_in_custody = state_dict.get('frames_in_custody', 0)
        
        # Normalize and combine into vector
        vector = np.array([
            # Position (normalized to Earth radius)
            position[0] / 6371.0,
            position[1] / 6371.0,
            position[2] / 6371.0,
            # Velocity (normalized to typical LEO)
            velocity[0] / 7.5,
            velocity[1] / 7.5,
            velocity[2] / 7.5,
            # Intent (one-hot encoded)
            1.0 if intent == 0 else 0.0,  # PASSIVE
            1.0 if intent == 1 else 0.0,  # RPO_INSPECTION
            1.0 if intent == 2 else 0.0,  # ELECTRONIC_WARFARE
            1.0 if intent == 3 else 0.0,  # KINETIC_THREAT
            # Threat indicators (normalized)
            (power_estimate + 100.0) / 100.0,  # Normalize dBm
            risk_score / 100.0,
            hcw_residual,
            delta_v_recent / 100.0,  # Normalize to 100 m/s
            time_since_custody / 300.0,  # Normalize to 5 minutes
            float(frames_in_custody) / 100.0  # Normalize to 100 frames
        ], dtype=np.float32)
        
        return vector
    
    def _calculate_entropy(self, state_vector: np.ndarray) -> float:
        """Calculate policy entropy for exploration tracking"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_probs, _ = self.policy(state_tensor)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10)).item()
        return entropy
    
    def _action_to_name(self, action: int) -> str:
        """Convert action index to human-readable name"""
        names = {
            0: "PASSIVE_TRACK",
            1: "ACTIVE_SCAN",
            2: "FOLLOW_TARGET",
            3: "EMITTER_GEO",
            4: "TACTICAL_ALERT",
            5: "FUEL_OPTIMAL_EVADE"
        }
        return names.get(action, "UNKNOWN")


def main():
    """Main entry point for policy bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Policy Bridge for ADITI 4.0')
    parser.add_argument('--endpoint', type=str, default='ipc:///tmp/policy_bridge.ipc',
                        help='ZMQ endpoint')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--train', action='store_true',
                        help='Run in training mode (accepts reward feedback)')
    
    args = parser.parse_args()
    
    bridge = PolicyBridge(endpoint=args.endpoint, model_path=args.model)
    
    try:
        bridge.start()
    except KeyboardInterrupt:
        print("\n[POLICY] Shutting down...")
    finally:
        bridge.stop()


if __name__ == '__main__':
    main()
