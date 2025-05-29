import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Union, List, Tuple, Any
from gymnasium.spaces import Discrete, Box
from omegaconf import DictConfig

from .base_agent import BaseRLAgent
from .networks import ActorNetwork, CriticNetwork


class PPOBuffer:
    """PPO experience buffer"""
    
    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((capacity,), dtype=torch.float32, device=device)
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        # Add last value for bootstrapping
        values = torch.cat([self.values, torch.tensor([last_value], device=self.device)])
        
        gae = 0
        for step in reversed(range(self.size)):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        
        # Normalize advantages
        self.advantages[:self.size] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.returns[:self.size] = returns
    
    def get_batch(self, batch_size: int):
        """Get random batch of experiences"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
    
    def get_all(self):
        """Get all experiences from buffer"""
        return (
            self.observations[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.values[:self.size],
            self.log_probs[:self.size],
            self.dones[:self.size]
        )
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size


class DronePPOAgent(BaseRLAgent):
    """PPO Agent for drone control with clean API"""
    
    def __init__(self, observation_space: Box, action_space: Discrete, 
                 config: DictConfig, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
        
        # Store config
        self.config = config
        
        # Extract hyperparameters
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.clip_range = config.clip_range
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.gae_lambda = config.gae_lambda
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.ppo_epochs = config.ppo_epochs
        self.batch_size = config.batch_size
        
        # Network dimensions
        obs_dim = np.prod(observation_space.shape)
        action_dim = action_space.n
        
        # Initialize networks
        self.actor = ActorNetwork(obs_dim, action_dim, config.network.hidden_dims).to(device)
        self.critic = CriticNetwork(obs_dim, config.network.hidden_dims).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )
        
        # Embedded buffer
        self.buffer = PPOBuffer(config.buffer_size, obs_dim, device)
        
        # For action selection
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action and store necessary info for PPO"""
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            
            # Get action probabilities and value
            action_probs = self.actor(obs)
            value = self.critic(obs)
            
            if training:
                # Sample action from policy
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # Deterministic action (for evaluation)
                action = action_probs.argmax(dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(action)
            
            # Store for later use in process()
            self.last_action = action.item()
            self.last_log_prob = log_prob.item()
            self.last_value = value.item()
            
            return self.last_action
    
    def process(self, transition: tuple) -> Dict[str, float]:
        """Process a transition and update the agent"""
        state, action, reward, next_state, done = transition
        
        # Increment step count
        self.total_steps += 1
        
        # Add experience to buffer (using stored values from select_action)
        if self.last_action is not None:
            self.buffer.add(
                state,
                self.last_action,
                reward,
                self.last_value,
                self.last_log_prob,
                done
            )
        
        metrics = {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        
        # Update if buffer is full
        if len(self.buffer) >= self.buffer.capacity:
            # Compute final value for bootstrapping
            with torch.no_grad():
                next_obs = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
                if next_obs.ndim == 1:
                    next_obs = next_obs.unsqueeze(0)
                final_value = self.critic(next_obs).item() if not done else 0.0
            
            # Update networks
            update_metrics = self._update_networks(final_value)
            metrics.update(update_metrics)
            
            # Clear buffer
            self.buffer.clear()
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': self.config
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model state"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        # Use weights_only=False for backward compatibility with older models
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback for older model files
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore training state
        self.total_steps = checkpoint.get('total_steps', 0)
    
    def train(self) -> None:
        """Set to training mode"""
        super().train()
        self.actor.train()
        self.critic.train()
        
    def eval(self) -> None:
        """Set to evaluation mode"""
        super().eval()
        self.actor.eval()
        self.critic.eval()
    
    def to(self, device: str) -> None:
        """Move to device"""
        super().to(device)
        self.actor.to(device)
        self.critic.to(device)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = super().get_stats()
        stats.update({
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity
        })
        return stats
    
    # Private methods (unchanged implementation)
    def _update_networks(self, final_value: float) -> Dict[str, float]:
        """Update actor and critic networks using PPO"""
        # Get all experiences from buffer
        experiences = self.buffer.get_all()
        states, actions, rewards, values, log_probs, dones = experiences
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, final_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors - states is already a tensor from buffer
        states = states.to(self.device)  # states is already a tensor
        actions = actions.to(self.device)  # actions is already a tensor
        old_log_probs = log_probs.to(self.device)  # log_probs is already a tensor
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        # PPO update for multiple epochs
        for _ in range(self.ppo_epochs):
            # Forward pass
            action_probs = self.actor(states)
            current_values = self.critic(states).squeeze()
            
            # Compute policy loss
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(current_values, returns)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
        
        # Average over epochs
        return {
            "loss": total_loss / self.ppo_epochs,
            "actor_loss": total_actor_loss / self.ppo_epochs,
            "critic_loss": total_critic_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs
        }
    
    def _compute_gae(self, rewards, values, dones, final_value: float):
        """Compute Generalized Advantage Estimation"""
        # Convert tensors to lists if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().tolist()
        if isinstance(values, torch.Tensor):
            values = values.cpu().tolist()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().tolist()
            
        advantages = []
        returns = []
        gae = 0
        
        # Add final value to values list for bootstrapping
        values = values + [final_value]
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Convert to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Compute returns
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        
        return advantages, returns 