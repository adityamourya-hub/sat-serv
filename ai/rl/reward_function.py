"""
Physics-Informed Reward Function for ADITI 4.0 RL Policy
Implements: R = w1*CustodyTime - w2*sum|Δv| - w3*ThreatMiss
"""

import numpy as np
from typing import Dict, Tuple


class PhysicsInformedRewardFunction:
    """
    Reward function incorporating:
    - Custody time rewards (maintaining target lock)
    - Fuel consumption penalties (Yoshida-4 integrator residuals)
    - Threat miss penalties (heavy for KINETIC threats)
    - Intent-specific shaping
    """
    
    def __init__(
        self,
        w_custody: float = 1.0,
        w_fuel: float = 0.5,
        w_threat: float = 10.0,
        w_intent: float = 2.0,
        custody_threshold: int = 10,  # frames
        fuel_threshold: float = 5.0,   # m/s per episode
        hcw_threshold: float = 0.1     # HCW residual threshold
    ):
        self.w_custody = w_custody
        self.w_fuel = w_fuel
        self.w_threat = w_threat
        self.w_intent = w_intent
        
        self.custody_threshold = custody_threshold
        self.fuel_threshold = fuel_threshold
        self.hcw_threshold = hcw_threshold
        
        # Intent label mapping
        self.INTENT_PASSIVE = 0
        self.INTENT_RPO = 1
        self.INTENT_EW = 2
        self.INTENT_KINETIC = 3
        
        # Action mapping
        self.ACTION_PASSIVE = 0
        self.ACTION_SCAN = 1
        self.ACTION_FOLLOW = 2
        self.ACTION_GEO = 3
        self.ACTION_ALERT = 4
        self.ACTION_EVADE = 5
    
    def compute(
        self,
        state: Dict,
        next_state: Dict,
        action: int,
        info: Dict = None
    ) -> Tuple[float, Dict]:
        """
        Compute physics-informed reward
        
        R = w1*Custody - w2*Fuel - w3*ThreatMiss + w4*IntentBonus
        
        Args:
            state: Previous state (dict with orbital parameters)
            next_state: Current state after action
            action: Action taken (0-5)
            info: Additional info (e.g., Yoshida-4 residuals)
            
        Returns:
            reward: Total reward value
            components: Dict of reward components for logging
        """
        components = {}
        
        # 1. Custody Reward
        custody_reward = self._compute_custody_reward(next_state)
        components['custody'] = custody_reward
        
        # 2. Fuel Penalty
        fuel_penalty = self._compute_fuel_penalty(next_state)
        components['fuel'] = -fuel_penalty
        
        # 3. Threat Miss Penalty
        threat_penalty = self._compute_threat_penalty(next_state, action)
        components['threat'] = -threat_penalty
        
        # 4. Intent-Specific Shaping
        intent_bonus = self._compute_intent_bonus(next_state, action)
        components['intent'] = intent_bonus
        
        # 5. Physics Residuals (Yoshida-4)
        physics_bonus = self._compute_physics_bonus(next_state, info)
        components['physics'] = physics_bonus
        
        # Total reward
        total_reward = (
            self.w_custody * custody_reward -
            self.w_fuel * fuel_penalty -
            self.w_threat * threat_penalty +
            self.w_intent * intent_bonus +
            physics_bonus
        )
        
        components['total'] = total_reward
        
        return total_reward, components
    
    def _compute_custody_reward(self, state: Dict) -> float:
        """Reward for maintaining target custody"""
        frames_in_custody = state.get('frames_in_custody', 0)
        time_since_custody = state.get('time_since_custody', 0.0)
        
        # Linear reward for consecutive custody frames
        if frames_in_custody > 0:
            return float(frames_in_custody)
        
        # Penalty for losing custody
        if time_since_custody > 60.0:  # Lost custody for > 60 seconds
            return -1.0
        
        return 0.0
    
    def _compute_fuel_penalty(self, state: Dict) -> float:
        """Penalty for fuel consumption (recent delta-V)"""
        delta_v = state.get('delta_v_recent', 0.0)
        
        # Linear penalty for delta-V
        # Cheap maneuvers (Keplerian drift) have low Δv
        # Expensive maneuvers (adversarial thrust) have high Δv
        return delta_v
    
    def _compute_threat_penalty(self, state: Dict, action: int) -> float:
        """Heavy penalty for missing kinetic threats"""
        intent = state.get('intent', self.INTENT_PASSIVE)
        confidence = state.get('confidence', 0.0)
        
        # Only penalize high-confidence detections
        if confidence < 0.7:
            return 0.0
        
        # KINETIC_THREAT requires immediate action
        if intent == self.INTENT_KINETIC:
            if action not in [self.ACTION_ALERT, self.ACTION_EVADE]:
                # Heavy penalty for not alerting or evading
                return 1.0
        
        # ELECTRONIC_WARFARE should trigger geolocation
        if intent == self.INTENT_EW:
            if action != self.ACTION_GEO:
                return 0.3  # Moderate penalty
        
        # RPO_INSPECTION should trigger active scanning
        if intent == self.INTENT_RPO:
            if action not in [self.ACTION_SCAN, self.ACTION_FOLLOW]:
                return 0.2  # Small penalty
        
        return 0.0
    
    def _compute_intent_bonus(self, state: Dict, action: int) -> float:
        """Bonus for correct intent-action matching"""
        intent = state.get('intent', self.INTENT_PASSIVE)
        
        # Optimal action-intent pairs
        optimal_pairs = {
            self.INTENT_PASSIVE: [self.ACTION_PASSIVE],
            self.INTENT_RPO: [self.ACTION_SCAN, self.ACTION_FOLLOW],
            self.INTENT_EW: [self.ACTION_GEO, self.ACTION_ALERT],
            self.INTENT_KINETIC: [self.ACTION_ALERT, self.ACTION_EVADE]
        }
        
        if action in optimal_pairs.get(intent, []):
            return 1.0
        
        return 0.0
    
    def _compute_physics_bonus(self, state: Dict, info: Dict = None) -> float:
        """Bonus for physics-grounded decisions"""
        if info is None:
            return 0.0
        
        bonus = 0.0
        
        # HCW residual indicates Keplerian drift (good)
        hcw_residual = info.get('hcw_residual', 0.0)
        if hcw_residual < self.hcw_threshold:
            # Low HCW residual = following natural orbital mechanics
            bonus += 0.5
        
        # Yoshida-4 integrator precision
        yoshida_error = info.get('yoshida_error', 0.0)
        if yoshida_error < 1e-6:
            # High precision physics integration
            bonus += 0.5
        
        return bonus
    
    def compute_advantage(
        self,
        reward: float,
        value: float,
        next_value: float,
        done: bool,
        gamma: float = 0.99
    ) -> float:
        """
        Compute advantage for policy gradient
        
        A(s,a) = r + γV(s') - V(s)  (TD error)
        """
        if done:
            return reward - value
        else:
            return reward + gamma * next_value - value


class IntentAwareRewardShaping:
    """
    Advanced reward shaping that considers adversary intent patterns
    Provides curriculum learning for different threat scenarios
    """
    
    def __init__(self, base_reward_fn: PhysicsInformedRewardFunction):
        self.base_reward = base_reward_fn
        self.episode_count = 0
        self.difficulty_level = 0
    
    def set_difficulty(self, level: int):
        """Adjust reward shaping based on training progress"""
        self.difficulty_level = level
        
        # Increase threat penalties as agent improves
        if level > 5:
            self.base_reward.w_threat *= 1.5
        if level > 10:
            self.base_reward.w_threat *= 2.0
    
    def compute(
        self,
        state: Dict,
        next_state: Dict,
        action: int,
        info: Dict = None,
        episode_step: int = 0
    ) -> Tuple[float, Dict]:
        """
        Compute shaped reward with curriculum
        """
        base_reward, components = self.base_reward.compute(
            state, next_state, action, info
        )
        
        # Curriculum shaping: early episodes get bonus for exploration
        if self.episode_count < 100 and episode_step < 10:
            exploration_bonus = 0.1
            components['exploration'] = exploration_bonus
            base_reward += exploration_bonus
        
        # Difficulty-based modifiers
        if self.difficulty_level > 5:
            # Higher difficulty: require more precise custody
            if components['custody'] < 5.0:
                base_reward -= 0.5
        
        return base_reward, components
    
    def on_episode_end(self, total_reward: float):
        """Called at end of each episode"""
        self.episode_count += 1
        
        # Increase difficulty every 100 episodes
        if self.episode_count % 100 == 0:
            self.set_difficulty(self.episode_count // 100)
            print(f"[REWARD] Difficulty increased to level {self.difficulty_level}")


def test_reward_function():
    """Test the reward function with sample states"""
    reward_fn = PhysicsInformedRewardFunction()
    
    # Test case 1: Passive tracking with custody
    state = {
        'frames_in_custody': 20,
        'intent': 0,  # PASSIVE
        'confidence': 0.9,
        'delta_v_recent': 0.5
    }
    next_state = {
        'frames_in_custody': 21,
        'intent': 0,
        'confidence': 0.9,
        'delta_v_recent': 0.5
    }
    action = 0  # PASSIVE_TRACK
    
    reward, components = reward_fn.compute(state, next_state, action)
    print(f"Test 1 (Passive + Custody): {reward:.2f}")
    print(f"  Components: {components}")
    
    # Test case 2: Missed kinetic threat
    state = {
        'frames_in_custody': 10,
        'intent': 3,  # KINETIC_THREAT
        'confidence': 0.9,
        'delta_v_recent': 2.0
    }
    next_state = {
        'frames_in_custody': 10,
        'intent': 3,
        'confidence': 0.9,
        'delta_v_recent': 2.0
    }
    action = 0  # PASSIVE_TRACK (WRONG!)
    
    reward, components = reward_fn.compute(state, next_state, action)
    print(f"\nTest 2 (Missed Kinetic): {reward:.2f}")
    print(f"  Components: {components}")
    print(f"  Large negative reward for missing threat!")
    
    # Test case 3: Correct EW response
    state = {
        'frames_in_custody': 5,
        'intent': 2,  # ELECTRONIC_WARFARE
        'confidence': 0.8,
        'delta_v_recent': 1.0
    }
    next_state = {
        'frames_in_custody': 6,
        'intent': 2,
        'confidence': 0.8,
        'delta_v_recent': 1.0
    }
    action = 3  # EMITTER_GEO (correct!)
    
    reward, components = reward_fn.compute(state, next_state, action)
    print(f"\nTest 3 (Correct EW Response): {reward:.2f}")
    print(f"  Components: {components}")


if __name__ == '__main__':
    test_reward_function()
