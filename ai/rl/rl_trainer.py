"""
RL Trainer - PPO/A2C training loop for ADITI 4.0 policy head
Integrates with physics-informed reward functions and replay buffer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import json
import time
from pathlib import Path

from policy_bridge import PolicyHead


class OUNoise:
    """
    Ornstein-Uhlenbeck Noise for temporally correlated exploration
    Ensures smooth exploration that respects hardware inertia (INDI mount)
    
    dx_t = θ(μ - x_t)dt + σdW_t
    """
    
    def __init__(
        self,
        action_dim: int = 6,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1.0
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta  # Mean reversion rate
        self.sigma = sigma  # Volatility
        self.dt = dt
        self.x = np.ones(action_dim) * mu  # Current noise state
        self.reset()
    
    def reset(self):
        """Reset noise state to mean"""
        self.x = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """
        Sample OU noise with temporal correlation
        
        Returns:
            noise: Correlated noise vector [action_dim]
        """
        # dx_t = θ(μ - x_t)dt + σdW_t
        dx = self.theta * (self.mu - self.x) * self.dt
        dw = np.random.normal(0, self.sigma * np.sqrt(self.dt))
        self.x = self.x + dx + dw
        return self.x
    
    def get_action_with_noise(self, action: int, noise_scale: float = 0.1) -> int:
        """
        Apply OU noise to discrete action selection
        Temporally correlated: if exploring, continue exploring for several steps
        
        Args:
            action: Original action from policy
            noise_scale: Scale of noise application
            
        Returns:
            noisy_action: Action with OU noise applied
        """
        noise = self.sample()
        
        # Convert discrete action to one-hot, apply noise, convert back
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        
        # Apply noise
        noisy_onehot = action_onehot + noise_scale * noise
        
        # Add small random perturbation for exploration
        noisy_onehot += np.random.normal(0, 0.05, self.action_dim)
        
        # Convert back to discrete action (argmax)
        noisy_action = np.argmax(noisy_onehot)
        
        # Clip to valid range
        noisy_action = np.clip(noisy_action, 0, self.action_dim - 1)
        
        return int(noisy_action)


@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    value: float
    log_prob: float


class ReplayBuffer:
    """Experience replay buffer for policy gradient methods"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
    def push(self, experience: Experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)


class CuriosityGovernor:
    """
    Dynamic Curiosity Threshold scaling by UCT density, innovation residuals, and tactical pressure
    Ensures exploration scales with environmental chaos and strategic threat levels
    
    C_T(ρ_UCT, ν, Ψ) = C_base + α·log(1 + ρ_UCT) + β·‖ν‖ + γ·Ψ
    """
    
    def __init__(
        self,
        base_epsilon: float = 0.1,
        alpha: float = 0.05,  # UCT sensitivity factor
        beta: float = 0.1,    # Physics surprise factor
        gamma: float = 0.02,  # Tactical pressure factor
        max_epsilon: float = 0.5
    ):
        self.base_epsilon = base_epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        
        self.uct_density = 0.0
        self.innovation_residuals = []
        self.tactical_pressure = 0.0  # Ψ from Go priority filter
        self.current_threshold = base_epsilon
    
    def update_threshold(self, uct_count: float, innovation_residuals: List[float], tactical_pressure: float = 0.0):
        """
        Update dynamic curiosity threshold based on environmental state and tactical pressure
        
        Args:
            uct_count: Number of Uncorrelated Tracks in view
            innovation_residuals: List of Kalman innovation residuals
            tactical_pressure: Tactical pressure (Ψ) from Go priority filter
        """
        # Store UCT density
        self.uct_density = uct_count
        self.innovation_residuals = innovation_residuals
        self.tactical_pressure = tactical_pressure
        
        # 1. UCT factor (log-scaled to prevent saturation)
        uct_factor = np.log1p(uct_count)
        
        # 2. Physics Innovation Factor (from Yoshida-4 residuals)
        physics_factor = 0.0
        if innovation_residuals:
            physics_factor = np.clip(np.mean(innovation_residuals), 0, 5.0)
        
        # 3. Tactical Pressure Factor (from Go priority filter)
        # Scaled by gamma to prevent overwhelming other factors
        tactical_factor = np.clip(tactical_pressure / 10.0, 0, 2.0)
        
        # 4. Compute Dynamic Threshold
        dynamic_eps = (
            self.base_epsilon +
            self.alpha * uct_factor +
            self.beta * physics_factor +
            self.gamma * tactical_factor
        )
        
        # Cap at maximum to prevent total chaos
        self.current_threshold = np.min(dynamic_eps, self.max_epsilon)
        
        return self.current_threshold
    
    def should_explore(self, policy_entropy: float = 0.0) -> bool:
        """
        Decide whether to explore based on current threshold
        
        Args:
            policy_entropy: Current policy entropy (optional)
            
        Returns:
            explore: True if should explore, False if should exploit
        """
        # Combine threshold with policy entropy for decision
        exploration_score = self.current_threshold + policy_entropy * 0.1
        return np.random.random() < exploration_score
    
    def get_curiosity_state(self) -> Dict:
        """Return current curiosity state for logging"""
        return {
            'uct_density': self.uct_density,
            'innovation_mean': np.mean(self.innovation_residuals) if self.innovation_residuals else 0.0,
            'tactical_pressure': self.tactical_pressure,
            'current_threshold': self.current_threshold,
            'mode': 'EXPLORATION' if self.current_threshold > 0.2 else 'EXPLOITATION'
        }


class PhysicsInformedReward:
    """
    Physics-informed reward function incorporating:
    - Custody time rewards
    - Fuel consumption penalties
    - Threat miss penalties (heavy)
    - Yoshida-4 integrator residuals
    """
    
    def __init__(self, w_custody: float = 1.0, w_fuel: float = 0.5, w_threat: float = 10.0):
        self.w_custody = w_custody
        self.w_fuel = w_fuel
        self.w_threat = w_threat
        
    def compute(self, state: Dict, next_state: Dict, action: int) -> float:
        """
        Compute reward: R = w1*Custody - w2*sum|Δv| - w3*ThreatMiss
        
        Args:
            state: Previous state dictionary
            next_state: Current state dictionary
            action: Action taken (0-5)
            
        Returns:
            reward: Scalar reward value
        """
        # Custody reward (frames in custody)
        custody_reward = next_state.get('frames_in_custody', 0) * self.w_custody
        
        # Fuel penalty (recent delta-V)
        delta_v = next_state.get('delta_v_recent', 0.0)
        fuel_penalty = delta_v * self.w_fuel
        
        # Threat miss penalty (heavy for missed kinetic threats)
        intent = next_state.get('intent', 0)
        threat_penalty = 0.0
        
        if intent == 3:  # KINETIC_THREAT
            if action not in [4, 5]:  # Not TACTICAL_ALERT or FUEL_OPTIMAL_EVADE
                threat_penalty = self.w_threat
                
        # Additional shaping based on intent
        intent_bonus = 0.0
        if intent == 2 and action == 3:  # Electronic warfare + EMITTER_GEO
            intent_bonus = 2.0
        elif intent == 1 and action == 2:  # RPO inspection + FOLLOW_TARGET
            intent_bonus = 1.0
            
        return custody_reward - fuel_penalty - threat_penalty + intent_bonus


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer
    More stable than vanilla A2C for defense applications
    """
    
    def __init__(
        self,
        policy: PolicyHead,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_ou_noise: bool = True,
        use_curiosity_governor: bool = True
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE parameter
        self.clip_epsilon = clip_epsilon  # PPO clipping
        self.value_coef = value_coef  # Value loss weight
        self.entropy_coef = entropy_coef  # Entropy bonus weight
        self.max_grad_norm = max_grad_norm  # Gradient clipping
        
        self.reward_fn = PhysicsInformedReward()
        self.episode_rewards = []
        
        # Exploration components
        self.use_ou_noise = use_ou_noise
        self.ou_noise = OUNoise(action_dim=6) if use_ou_noise else None
        
        self.use_curiosity_governor = use_curiosity_governor
        self.curiosity_governor = CuriosityGovernor() if use_curiosity_governor else None
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of state value estimates
            dones: List of episode termination flags
            
        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates
        """
        advantages = []
        gae = 0.0
        
        # Work backwards through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Compute returns (advantage + value)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def select_action_with_exploration(
        self,
        state: np.ndarray,
        kalman_uncertainty: float = 0.0,
        uct_count: float = 0.0,
        innovation_residuals: List[float] = None,
        tactical_pressure: float = 0.0
    ) -> Tuple[int, float, float, bool]:
        """
        Select action with advanced exploration strategies
        
        Args:
            state: Current state vector
            kalman_uncertainty: Kalman filter position covariance trace
            uct_count: Number of Uncorrelated Tracks
            innovation_residuals: Kalman innovation residuals
            tactical_pressure: Tactical pressure (Ψ) from Go priority filter
            
        Returns:
            action: Selected action index
            action_prob: Probability of selected action
            value: State value estimate
            explored: Whether exploration was used
        """
        # Update curiosity threshold if enabled
        if self.curiosity_governor:
            innovation_residuals = innovation_residuals or []
            self.curiosity_governor.update_threshold(uct_count, innovation_residuals, tactical_pressure)
        
        # Get base action from policy
        action, action_prob, value = self.policy.get_action(state, deterministic=False)
        
        explored = False
        
        # Apply OU noise if enabled
        if self.ou_noise and self.curiosity_governor:
            if self.curiosity_governor.should_explore():
                action = self.ou_noise.get_action_with_noise(action, noise_scale=0.1)
                explored = True
        
        return action, action_prob, value, explored
    
    def update(self, experiences: List[Experience], epochs: int = 4, batch_size: int = 64) -> Dict:
        """
        Update policy using PPO
        
        Args:
            experiences: List of experiences from rollout
            epochs: Number of optimization epochs
            batch_size: Batch size for updates
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Extract data
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences])
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(epochs):
            # Mini-batch updates
            indices = torch.randperm(len(experiences))
            
            for start in range(0, len(experiences), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, state_values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # PPO policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.MSELoss()(state_values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = epochs * (len(experiences) // batch_size + 1)
        
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards)
        }
        
        self.episode_rewards.append(np.sum(rewards))
        
        return metrics
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }
        torch.save(checkpoint, path)
        print(f"[TRAINER] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"[TRAINER] Loaded checkpoint from {path}")
        return checkpoint.get('episode', 0)


class RLTrainer:
    """
    Main training orchestrator for ADITI 4.0 RL policy
    Coordinates data collection, training, and evaluation
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 6,
        lr: float = 3e-4,
        save_dir: str = "./checkpoints"
    ):
        # Initialize policy and trainer
        self.policy = PolicyHead(state_dim=state_dim, action_dim=action_dim)
        self.trainer = PPOTrainer(self.policy, lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Configuration
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.episode = 0
        self.step = 0
        
    def collect_experience(
        self,
        env,
        max_steps: int = 1000,
        deterministic: bool = False
    ) -> List[Experience]:
        """
        Collect experience by running policy in environment
        
        Args:
            env: Environment that provides states and accepts actions
            max_steps: Maximum steps per episode
            deterministic: Use deterministic policy (no exploration)
            
        Returns:
            experiences: List of collected experiences
        """
        experiences = []
        state = env.reset()
        
        for step in range(max_steps):
            # Get action from policy
            action, action_prob, value = self.policy.get_action(state, deterministic)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            experience = Experience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                value=value,
                log_prob=np.log(action_prob + 1e-10)
            )
            experiences.append(experience)
            
            state = next_state
            
            if done:
                break
        
        return experiences
    
    def train_episode(
        self,
        env,
        max_steps: int = 1000,
        update_epochs: int = 4
    ) -> Dict:
        """
        Run one training episode
        
        Args:
            env: Training environment
            max_steps: Maximum steps per episode
            update_epochs: Number of policy update epochs
            
        Returns:
            metrics: Training metrics for this episode
        """
        # Collect experience
        experiences = self.collect_experience(env, max_steps)
        
        # Add to replay buffer
        for exp in experiences:
            self.replay_buffer.push(exp)
        
        # Update policy if enough experiences
        if len(experiences) >= 32:
            metrics = self.trainer.update(experiences, epochs=update_epochs)
        else:
            metrics = {'policy_loss': 0.0, 'value_loss': 0.0}
        
        self.episode += 1
        self.step += len(experiences)
        
        # Periodic checkpointing
        if self.episode % 100 == 0:
            checkpoint_path = self.save_dir / f"policy_episode_{self.episode}.pt"
            self.trainer.save_checkpoint(str(checkpoint_path), self.episode)
        
        return metrics
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict:
        """
        Evaluate current policy
        
        Args:
            env: Evaluation environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            metrics: Evaluation metrics
        """
        rewards = []
        
        for _ in range(num_episodes):
            experiences = self.collect_experience(env, deterministic=True)
            episode_reward = sum([e.reward for e in experiences])
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    def train(self, env, num_episodes: int = 1000, eval_interval: int = 50):
        """
        Main training loop
        
        Args:
            env: Training environment
            num_episodes: Total training episodes
            eval_interval: Evaluate every N episodes
        """
        print("[TRAINER] Starting training loop")
        print(f"[TRAINER] Episodes: {num_episodes}, Eval interval: {eval_interval}")
        
        for episode in range(num_episodes):
            # Train one episode
            metrics = self.train_episode(env)
            
            # Logging
            if episode % 10 == 0:
                print(f"[TRAINER] Episode {episode}/{num_episodes}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate(env)
                print(f"[TRAINER] Evaluation at episode {episode}:")
                print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        
        # Save final model
        final_path = self.save_dir / "policy_final.pt"
        self.trainer.save_checkpoint(str(final_path), self.episode)
        print("[TRAINER] Training complete!")


def main():
    """Main entry point for RL training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Trainer for ADITI 4.0')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RLTrainer(lr=args.lr, save_dir=args.save_dir)
    
    # Load checkpoint if resuming
    start_episode = 0
    if args.resume:
        start_episode = trainer.trainer.load_checkpoint(args.resume)
        trainer.episode = start_episode
    
    # TODO: Initialize actual environment
    # For now, placeholder
    class DummyEnv:
        def reset(self):
            return np.random.randn(15)
        def step(self, action):
            return np.random.randn(15), np.random.randn(), False, {}
    
    env = DummyEnv()
    
    # Start training
    trainer.train(env, num_episodes=args.episodes)


if __name__ == '__main__':
    main()
