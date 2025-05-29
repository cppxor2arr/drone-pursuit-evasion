import numpy as np
import warnings
from typing import Dict
from gymnasium.spaces import Discrete, Box
import os
import torch

from .base_agent import BaseRLAgent


class RandomAgent(BaseRLAgent):
    """Random action agent following unified interface"""
    
    def __init__(self, observation_space: Box, action_space: Discrete, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select random action"""
        return self.action_space.sample()
    
    def process(self, transition: tuple) -> Dict[str, float]:
        """Process transition (no learning for random agent)"""
        self.total_steps += 1
        # Random agents don't learn, so just return empty metrics
        return {"loss": 0.0}
    
    def save(self, path: str) -> None:
        """Save agent state (minimal for random agent)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'agent_type': 'random',
            'total_steps': self.total_steps
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load agent state"""
        if not os.path.exists(path):
            return  # Random agent doesn't need to load anything critical
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.total_steps = checkpoint.get('total_steps', 0)
        except:
            pass  # Graceful fallback for random agent


class HoveringAgent(BaseRLAgent):
    """Hovering agent that always selects action 0"""
    
    def __init__(self, observation_space: Box, action_space: Discrete, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Always hover (action 0)"""
        return 0
    
    def process(self, transition: tuple) -> Dict[str, float]:
        """Process transition (no learning for hovering agent)"""
        self.total_steps += 1
        # Hovering agents don't learn, so just return empty metrics
        return {"loss": 0.0}
    
    def save(self, path: str) -> None:
        """Save agent state (minimal for hovering agent)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'agent_type': 'hovering',
            'total_steps': self.total_steps
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load agent state"""
        if not os.path.exists(path):
            return  # Hovering agent doesn't need to load anything critical
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.total_steps = checkpoint.get('total_steps', 0)
        except:
            pass  # Graceful fallback for hovering agent 