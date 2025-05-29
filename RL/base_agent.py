from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from gymnasium.spaces import Box, Discrete


class BaseRLAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, observation_space: Box, action_space: Discrete, device: str = "cpu"):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.total_steps = 0
        self.training = True
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action given observation
        
        Args:
            observation: Current observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action (integer)
        """
        pass
    
    @abstractmethod
    def process(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> Dict[str, float]:
        """Process a transition (state, action, reward, next_state, done)
        
        Args:
            transition: (state, action, reward, next_state, done)
            
        Returns:
            Dictionary of metrics (e.g., loss, epsilon, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state"""
        pass
    
    def train(self) -> None:
        """Set to training mode"""
        self.training = True
        
    def eval(self) -> None:
        """Set to evaluation mode"""
        self.training = False
    
    def to(self, device: str) -> None:
        """Move to device"""
        self.device = device
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "total_steps": self.total_steps,
            "training": self.training
        } 