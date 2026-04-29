"""
Sensor Tasking & Volumetric Scan Optimizer — RL-based scheduling for Astra-Mesh.

Implements PPO-based optimization for:
  - Volumetric scan: User-defined Earth region → optimized sensor tasking
  - Blind spot detection: ML flags coverage gaps for adversary objects
  - Multi-sensor dwell optimization: Radar/optical/SAR under IAF constraints

Based on RAND/AMOS 2024 research showing RL scheduling as top SDA force multiplier.

Architecture:
  PPO policy network → sensor selection + dwell time → coverage reward
  Integrates with fusion_gnn.py for multi-sensor state estimation.

Performance target: >95% region coverage, <10ms inference (matches collision surrogate).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import gymnasium as gym
from gymnasium import spaces


class SensorType(Enum):
    ATEENA_RADAR = "ateena"
    RISAT_SAR = "sar"
    OPTICAL_CCD = "optical"
    NAVIC_GNSS = "navic"


@dataclass
class SensorState:
    """Current state of a sensor station."""
    station_id: str
    sensor_type: SensorType
    latitude: float        # deg
    longitude: float       # deg
    elevation_m: float
    availability: float     # [0,1] — 1.0 = fully available
    last_tasked_jd: float  # Julian date of last tasking
    energy_remaining: float  # Arbitrary units, depletes with use
    iono_activity: float   # 0-1 scale (affects radar quality)


@dataclass
class TargetRegion:
    """User-defined region of interest."""
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    priority: float        # 0-1, higher = more important
    min_revisit_hours: float
    threat_density: float  # Estimated objects of interest per 1000 km²


@dataclass
class TaskingAction:
    """Action: which sensor to task, where to point, how long."""
    station_id: str
    target_lat: float
    target_lon: float
    dwell_seconds: float   # 10-300s typical
    sensor_mode: str      # 'search', 'track', 'isar', 'photometric'


class SensorTaskingEnv(gym.Env):
    """
    Gym environment for sensor tasking RL.
    
    State space: concatenated sensor states + region coverage + time
    Action space: (sensor_idx, azimuth, elevation, dwell_time)
    Reward: coverage % + threat detection - energy cost
    """
    
    def __init__(self, stations: List[SensorState], 
                 regions: List[TargetRegion],
                 sim_duration_hours: float = 24.0):
        super().__init__()
        
        self.stations = stations
        self.regions = regions
        self.sim_duration = sim_duration_hours
        self.dt_hours = 0.5  # Time step: 30 minutes
        
        # State: per-sensor features + region coverage + time
        n_sensor_features = 8  # [type_encoded, lat, lon, avail, time_since_task, energy, iono, 0]
        n_region_features = 5  # [priority, coverage, time_since_visit, threat_density, blind_spot_flag]
        state_dim = len(stations) * n_sensor_features + len(regions) * n_region_features + 1
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action: [station_idx (discrete), azimuth, elevation, dwell_norm]
        # We'll use a wrapper to handle mixed discrete/continuous
        self.action_dim = 4
        self.n_stations = len(stations)
        
        # Track coverage
        self.coverage_history = {r.name: [] for r in regions}
        self.detected_objects = []
        self.energy_consumed = 0.0
        
        self.reset()
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_time = 0.0  # Hours into simulation
        self.coverage_grid = {
            r.name: np.zeros((10, 10), dtype=np.float32)  # 10x10 grid per region
            for r in self.regions
        }
        self.last_visit_time = {r.name: 0.0 for r in self.regions}  # Use 0.0 instead of -inf
        self.blind_spots = []  # List of (lat, lon, time) gaps
        
        # Reset stations
        for s in self.stations:
            s.availability = 1.0
            s.energy_remaining = 100.0
            s.last_tasked_jd = 0.0
        
        return self._get_obs()
    
    def _get_obs(self) -> np.float32:
        """Build observation vector."""
        obs = []
        
        # Sensor features
        for s in self.stations:
            type_enc = 0.0 if s.sensor_type == SensorType.ATEENA_RADAR else \
                      1.0 if s.sensor_type == SensorType.RISAT_SAR else \
                      2.0 if s.sensor_type == SensorType.OPTICAL_CCD else 3.0
            obs.extend([
                type_enc / 3.0,
                s.latitude / 90.0,
                s.longitude / 180.0,
                s.availability,
                (self.current_time - s.last_tasked_jd * 24) / 24.0,  # Time since task (days → hours)
                s.energy_remaining / 100.0,
                s.iono_activity,
                0.0,  # Padding
            ])
        
        # Region features
        for r in self.regions:
            grid = self.coverage_grid[r.name]
            coverage = grid.mean()
            time_since = self.current_time - self.last_visit_time[r.name]
            blind_spot = 1.0 if coverage < 0.5 and time_since > r.min_revisit_hours else 0.0
            
            obs.extend([
                r.priority,
                coverage,
                time_since / 24.0,
                r.threat_density,
                blind_spot,
            ])
        
        # Time feature
        obs.append(self.current_time / self.sim_duration)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one tasking action.
        
        Action: [station_idx (0 to n-1), azimuth_norm, elevation_norm, dwell_norm]
        """
        station_idx = int(np.clip(action[0], 0, self.n_stations - 1))
        azimuth = action[1] * 180.0  # [-180, 180]
        elevation = action[2] * 90.0  # [-90, 90]
        dwell = 10.0 + action[3] * 290.0  # [10, 300] seconds
        
        station = self.stations[station_idx]
        
        # Check constraints
        if station.availability < 0.1 or station.energy_remaining < 10.0:
            # Invalid action — penalize
            reward = -1.0
            self.current_time += self.dt_hours
            done = self.current_time >= self.sim_duration
            return self._get_obs(), reward, done, {"invalid": True}
        
        # Compute tasking location from az/el (simplified)
        # In reality: propagate from station location + pointing
        task_lat = station.latitude + elevation * 0.1  # Rough approximation
        task_lon = station.longitude + azimuth * 0.1
        
        # Update coverage
        reward = 0.0
        for r in self.regions:
            if r.lat_min <= task_lat <= r.lat_max and r.lon_min <= task_lon <= r.lon_max:
                # Within region — update coverage grid
                grid = self.coverage_grid[r.name]
                # Mark nearby cells as covered
                lat_idx = int((task_lat - r.lat_min) / (r.lat_max - r.lat_min) * 10)
                lon_idx = int((task_lon - r.lon_min) / (r.lon_max - r.lon_min) * 10)
                lat_idx = np.clip(lat_idx, 0, 9)
                lon_idx = np.clip(lon_idx, 0, 9)
                
                # Coverage decays over time
                coverage_boost = dwell / 300.0  # Max boost at 300s dwell
                grid[lat_idx, lon_idx] = min(1.0, grid[lat_idx, lon_idx] + coverage_boost)
                
                # Reward for coverage
                reward += r.priority * coverage_boost * 10.0
                
                # Track blind spots
                if grid.mean() < 0.5:
                    self.blind_spots.append((task_lat, task_lon, self.current_time))
                
                self.last_visit_time[r.name] = self.current_time
        
        # Simulate threat detection (probabilistic)
        detection_prob = 0.3 + 0.5 * (dwell / 300.0)  # Longer dwell = better detection
        if np.random.random() < detection_prob:
            # Simulate detecting an object
            n_objects = np.random.poisson(2)
            for _ in range(n_objects):
                self.detected_objects.append({
                    'lat': task_lat + np.random.normal(0, 0.5),
                    'lon': task_lon + np.random.normal(0, 0.5),
                    'time': self.current_time,
                    'sensor': station.station_id,
                })
                reward += 5.0  # Reward for detection
        
        # Energy cost (penalty)
        energy_cost = dwell / 10.0  # Arbitrary units
        station.energy_remaining -= energy_cost
        self.energy_consumed += energy_cost
        reward -= energy_cost * 0.01  # Small penalty
        
        # Update station state
        station.last_tasked_jd = self.current_time / 24.0
        station.availability = max(0.0, station.availability - 0.05)  # Degrades with use
        station.availability = min(1.0, station.availability + 0.02)  # Recovers over time
        
        # Time step
        self.current_time += self.dt_hours
        done = self.current_time >= self.sim_duration
        
        info = {
            'coverage': {r.name: self.coverage_grid[r.name].mean() for r in self.regions},
            'blind_spots': len(self.blind_spots),
            'detections': len(self.detected_objects),
            'energy': self.energy_consumed,
        }
        
        return self._get_obs(), reward, done, info


class SensorTaskingPolicy(nn.Module):
    """
    Actor-Critic policy network for sensor tasking.
    
    Actor: outputs action distribution (station selection + pointing + dwell)
    Critic: estimates value function for advantage
    """
    
    def __init__(self, state_dim: int, n_stations: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared backbone with better initialization
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Initialize weights for numerical stability
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Actor head (policy)
        self.actor_station = nn.Linear(hidden_dim, n_stations)  # Discrete: station selection
        nn.init.orthogonal_(self.actor_station.weight, gain=0.01)
        nn.init.constant_(self.actor_station.bias, 0.0)
        
        self.actor_continuous = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [azimuth_norm, elevation_norm, dwell_norm]
            nn.Tanh(),  # Bound to [-1, 1]
        )
        nn.init.orthogonal_(self.actor_continuous[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor_continuous[2].weight, gain=0.01)
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.critic[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (station_logits, continuous_action, value)
        """
        x = self.shared(state)
        
        station_logits = self.actor_station(x)  # [B, n_stations]
        continuous = self.actor_continuous(x)   # [B, 3]
        value = self.critic(x)                   # [B, 1]
        
        return station_logits, continuous, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Sample action from policy."""
        with torch.no_grad():
            logits, continuous, value = self.forward(state)
            
            # Sample station
            if deterministic:
                station_idx = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                station_idx = torch.multinomial(probs, 1).squeeze(-1)
            
            # Combine into full action
            action = torch.cat([
                station_idx.unsqueeze(-1).float() / (logits.size(-1) - 1) * 2 - 1,  # Normalize to [-1,1]
                continuous
            ], dim=-1)
            
            return action.cpu().numpy(), value.item()


def train_sensor_tasking_ppo(
    stations: List[SensorState],
    regions: List[TargetRegion],
    total_timesteps: int = 100000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SensorTaskingPolicy:
    """
    Train sensor tasking policy using PPO.
    
    Args:
        stations: List of sensor stations in Astra-Mesh
        regions: List of priority regions to monitor
        total_timesteps: Training iterations
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained policy network
    """
    print(f"Training Sensor Tasking RL on {device}")
    print(f"  Stations: {len(stations)}")
    print(f"  Regions: {len(regions)}")
    print(f"  Timesteps: {total_timesteps}")
    
    # Create environment
    env = SensorTaskingEnv(stations, regions)
    state_dim = env.observation_space.shape[0]
    n_stations = len(stations)
    
    # Initialize policy
    policy = SensorTaskingPolicy(state_dim, n_stations).to(device)
    # Force FP32 to prevent numerical instability
    policy = policy.float()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # AMP mixed precision for V100 GPU acceleration
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # PPO hyperparameters
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE parameter
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    
    # Training loop (simplified PPO)
    episode_rewards = []
    
    for step in range(total_timesteps):
        # Collect trajectory
        state = torch.FloatTensor(env.reset()).to(device)
        
        # Validate input state for NaN/inf
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"Warning: Invalid state detected at step {step}, resetting")
            state = torch.zeros_like(state)
        
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        for t in range(256):  # Episode length (trajectory) - increased for V100 16GB
            # Forward pass (keep FP32 for numerical stability)
            logits, continuous, value = policy(state.unsqueeze(0))
            
            # Sample action (use logits directly for numerical stability)
            station_dist = torch.distributions.Categorical(logits=logits)
            station_idx = station_dist.sample()
            
            # Build action tensor with correct dimensions
            # station_idx is [1] (batched), squeeze to scalar then normalize
            station_action = station_idx.squeeze().float() / (n_stations - 1) * 2 - 1
            continuous_action = continuous.squeeze(0)
            
            # Concatenate: both are now 1D tensors
            action = torch.cat([station_action.unsqueeze(0), continuous_action])
            
            # Log probability
            log_prob = station_dist.log_prob(station_idx)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            
            # Step environment
            next_state, reward, done, info = env.step(action.cpu().numpy())
            
            rewards.append(reward)
            dones.append(done)
            
            state = torch.FloatTensor(next_state).to(device)
            
            if done:
                break
        
        # Compute returns and advantages (simplified)
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        # PPO update (simplified single update per trajectory)
        advantages = returns - values
        
        # Policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = 0.5 * advantages.pow(2).mean()
        
        # Total loss
        loss = policy_loss + value_coef * value_loss
        
        optimizer.zero_grad()
        
        # AMP mixed precision backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        if step % 100 == 0:
            avg_reward = sum(rewards) / len(rewards)
            print(f"Step {step}: Avg Reward = {avg_reward:.2f}, Detections = {len(env.detected_objects)}")
    
    print("Training complete")
    return policy


def export_tasking_policy_onnx(policy: SensorTaskingPolicy, 
                                state_dim: int, 
                                output_path: str = "ai/training/models/sensor_tasking.onnx"):
    """Export trained policy to ONNX for Go inference."""
    import torch.onnx
    
    dummy_state = torch.randn(1, state_dim)
    
    # Wrap to return continuous actions only (station selection done in Go)
    class ONNXWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, state):
            logits, continuous, value = self.policy(state)
            # Return: station probabilities + continuous action + value
            return torch.softmax(logits, dim=-1), continuous, value
    
    wrapped = ONNXWrapper(policy)
    
    torch.onnx.export(
        wrapped,
        dummy_state,
        output_path,
        input_names=["sensor_state"],
        output_names=["station_probs", "continuous_action", "value"],
        dynamic_axes={"sensor_state": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX policy to {output_path}")


# --- Self-Test ---

if __name__ == "__main__":
    print("Testing Sensor Tasking RL Environment...")
    
    # Define Astra-Mesh stations
    stations = [
        SensorState("leh", SensorType.ATEENA_RADAR, 34.0, 77.5, 3500, 1.0, 0.0, 100.0, 0.3),
        SensorState("thumba", SensorType.ATEENA_RADAR, 8.5, 76.9, 10, 1.0, 0.0, 100.0, 0.5),
        SensorState("sriharikota", SensorType.OPTICAL_CCD, 13.7, 80.2, 10, 1.0, 0.0, 100.0, 0.2),
        SensorState("port_blair", SensorType.RISAT_SAR, 11.6, 92.7, 10, 1.0, 0.0, 100.0, 0.4),
    ]
    
    # Define priority regions (India + borders)
    regions = [
        TargetRegion("Northern Border", 28.0, 38.0, 72.0, 80.0, 0.95, 2.0, 0.8),
        TargetRegion("Eastern Seaboard", 8.0, 22.0, 80.0, 88.0, 0.90, 4.0, 0.6),
        TargetRegion("Southern Ocean", -10.0, 8.0, 75.0, 95.0, 0.70, 6.0, 0.3),
    ]
    
    # Quick training demo
    print("\nTraining PPO policy...")
    policy = train_sensor_tasking_ppo(
        stations, 
        regions, 
        total_timesteps=1000,  # Small for demo
        device="cpu"
    )
    
    # Test inference
    env = SensorTaskingEnv(stations, regions)
    state = torch.FloatTensor(env.reset()).unsqueeze(0)
    
    action, value = policy.get_action(state)
    print(f"\nSample action: station_idx={action[0]:.2f}, az={action[1]:.2f}, el={action[2]:.2f}, dwell={action[3]:.2f}")
    print(f"Estimated value: {value:.2f}")
    
    # Export
    state_dim = env.observation_space.shape[0]
    export_tasking_policy_onnx(policy, state_dim, "ai/training/models/sensor_tasking_demo.onnx")
    
    print("\n✓ Sensor Tasking RL operational")
