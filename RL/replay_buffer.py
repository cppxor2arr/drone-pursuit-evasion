import torch
import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """Replay buffer for experience replay in off-policy algorithms"""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        return (
            torch.from_numpy(np.array(states)).float().to(self.device),
            torch.from_numpy(np.array(actions)).long().to(self.device),
            torch.from_numpy(np.array(rewards)).float().to(self.device),
            torch.from_numpy(np.array(next_states)).float().to(self.device),
            torch.from_numpy(np.array(dones)).float().to(self.device)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def to(self, device: str) -> None:
        """Move buffer to specified device"""
        self.device = device 