import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
from typing import Dict, Union, Any
from gymnasium.spaces import Discrete, Box
from omegaconf import DictConfig

from .base_agent import BaseRLAgent
from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DroneDQNAgent(BaseRLAgent):
    """DQN Agent for drone control with clean API"""
    
    def __init__(self, observation_space: Box, action_space: Discrete, 
                 config: DictConfig, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
        
        # Store config
        self.config = config
        
        # Extract hyperparameters from config
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self._epsilon_start = config.epsilon_start
        self._epsilon_end = config.epsilon_end
        self._epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.warmup_steps = getattr(config, 'warmup_steps', 1000)
        self.target_update_interval = getattr(config, 'target_update_interval', 1000)
        
        # Internal state
        self._epsilon = self._epsilon_start
        
        # Network dimensions
        obs_dim = np.prod(observation_space.shape)
        action_dim = action_space.n
        
        # Initialize networks
        self.q_net = QNetwork(obs_dim, action_dim, config.network.hidden_dims).to(device)
        self.target_net = QNetwork(obs_dim, action_dim, config.network.hidden_dims).to(device)
        self._update_target_network()
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Embedded replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, device)
        
        # Epsilon decay rate
        self.epsilon_decay_rate = (self._epsilon_start - self._epsilon_end) / self._epsilon_decay
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or deterministic policy"""
        # Exploration case: epsilon-greedy, training mode, or warmup period
        if training and ((np.random.rand() < self._epsilon) or (self.total_steps < self.warmup_steps)):
            return self.action_space.sample()
        else:
            return self._exploit(observation)
    
    def process(self, transition: tuple) -> Dict[str, float]:
        """Process a transition and update the agent"""
        state, action, reward, next_state, done = transition
        
        # Increment step count
        self.total_steps += 1
        
        # Add to buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        metrics = {"loss": 0.0, "epsilon": self._epsilon, "steps": self.total_steps}
        
        # Train if enough samples and past warmup
        if self.total_steps > self.warmup_steps and len(self.replay_buffer) > self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self._compute_td_loss(batch)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            metrics["loss"] = loss.item()
        
        # Update target network periodically
        if self.total_steps % self.target_update_interval == 0:
            self._update_target_network()
        
        # Update epsilon
        if self.total_steps > self.warmup_steps:
            self._epsilon = max(self._epsilon_end, self._epsilon - self.epsilon_decay_rate)
        
        metrics["epsilon"] = self._epsilon
        return metrics
    
    def save(self, path: str) -> None:
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self._epsilon,
            'config': self.config
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model state"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore training state
        self.total_steps = checkpoint.get('total_steps', 0)
        self._epsilon = checkpoint.get('epsilon', self._epsilon_end)
    
    def train(self) -> None:
        """Set to training mode"""
        super().train()
        self.q_net.train()
        
    def eval(self) -> None:
        """Set to evaluation mode"""
        super().eval()
        self.q_net.eval()
    
    def to(self, device: str) -> None:
        """Move to device"""
        super().to(device)
        self.q_net.to(device)
        self.target_net.to(device)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = super().get_stats()
        stats.update({
            "epsilon": self._epsilon,
            "buffer_size": len(self.replay_buffer),
            "warmup_complete": self.total_steps > self.warmup_steps
        })
        return stats
    
    # Private methods
    def _exploit(self, observation: np.ndarray) -> int:
        """Select greedy action"""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        with torch.no_grad():
            q_values = self.q_net(obs)
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def _compute_td_loss(self, samples) -> torch.Tensor:
        """Compute temporal difference loss"""
        states, actions, rewards, next_states, dones = samples
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        # Compute loss
        loss = self.criterion(current_q, target_q)
        return loss
    
    def _update_target_network(self) -> None:
        """Hard update target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())

# TODO: observation이 input으로 들어올 때엔 batch로 들어올 수 있으므로, 조심하기 
# replace buffer는 stable baseline에서 구현 된 것 가져오기 
# vstack은 supersuit으로 된 것 가져오기 
    
if __name__ == "__main__":
    # This is just for testing/debugging the agent, not used in actual training
    print("DQN Agent module")
    # Uncomment and modify to test functionality as needed
    # from task.lider_drone_base import LidarDroneBaseEnv
    # env = LidarDroneBaseEnv(...)
    # agent = DroneDQNAgent(...)
