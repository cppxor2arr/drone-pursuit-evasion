import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple
import os
from omegaconf import DictConfig

from .base_agent import BaseRLAgent
from .replay_buffer import ReplayBuffer


class SACActorNetwork(nn.Module):
    """SAC Actor (Policy) Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list, 
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.backbone(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds using tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class SACCriticNetwork(nn.Module):
    """SAC Critic (Q-value) Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value"""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DroneSACAgent(BaseRLAgent):
    """SAC Agent for drone control following unified interface"""
    
    def __init__(self, observation_space, action_space, config: DictConfig, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
        
        self.config = config
        self.state_dim = observation_space.shape[0]
        
        # For discrete action spaces, we'll treat it as continuous and discretize
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.discrete_actions = True
        else:
            self.action_dim = action_space.shape[0]
            self.discrete_actions = True  # Assume discrete for drone control
        
        # SAC hyperparameters
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha  # Temperature parameter
        self.target_update_interval = config.get('target_update_interval', 1)
        self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', True)
        
        # Networks
        hidden_dims = config.network.hidden_dims
        
        # Actor
        self.actor = SACActorNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        
        # Critics (twin critics)
        self.critic1 = SACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        self.critic2 = SACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        
        # Target critics
        self.target_critic1 = SACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        self.target_critic2 = SACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dims
        ).to(device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            device
        )
        
        # Training parameters
        self.batch_size = config.batch_size
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        # Metrics tracking
        self.last_metrics = {}
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using SAC policy"""
        if training and self.total_steps < self.warmup_steps:
            # Random action during warmup
            if self.discrete_actions:
                return np.random.randint(0, self.action_dim)
            else:
                return np.random.uniform(-1, 1, self.action_dim)
        
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                # Sample action for exploration
                action, _ = self.actor.sample(state)
            else:
                # Use mean action for evaluation
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        
        action = action.cpu().numpy()[0]
        
        if self.discrete_actions:
            # Convert continuous action to discrete
            # Scale from [-1, 1] to [0, action_dim-1]
            action = np.clip(action, -1, 1)
            action = ((action + 1) / 2 * self.action_dim).astype(int)
            action = np.clip(action, 0, self.action_dim - 1)
            return action[0] if len(action) == 1 else action
        
        return action
    
    def process(self, transition: tuple) -> Dict[str, float]:
        """Process transition and update networks"""
        state, action, reward, next_state, done = transition
        
        # Store transition in replay buffer
        if self.discrete_actions:
            # Convert discrete action to continuous for storage
            action_cont = np.array([action], dtype=np.float32)
            action_cont = (action_cont / (self.action_dim - 1)) * 2 - 1  # Scale to [-1, 1]
        else:
            action_cont = action
        
        self.replay_buffer.add(state, action_cont, reward, next_state, done)
        self.total_steps += 1
        
        metrics = {}
        
        # Update networks if enough samples
        if len(self.replay_buffer) >= self.batch_size and self.total_steps > self.warmup_steps:
            metrics = self._update_networks()
        
        self.last_metrics = metrics
        return metrics
    
    def _update_networks(self) -> Dict[str, float]:
        """Update SAC networks"""
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        # Update critics
        critic_loss = self._update_critics(state, action, reward, next_state, done)
        
        # Update actor
        actor_loss = self._update_actor(state)
        
        # Update temperature
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = self._update_temperature(state)
        
        # Update target networks
        if self.total_steps % self.target_update_interval == 0:
            self._soft_update_targets()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
            'loss': critic_loss + actor_loss  # For compatibility
        }
    
    def _update_critics(self, state, action, reward, next_state, done) -> float:
        """Update critic networks"""
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Critic 1 loss
        current_q1 = self.critic1(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        # Critic 2 loss
        current_q2 = self.critic2(state, action)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss + critic2_loss).item() / 2
    
    def _update_actor(self, state) -> float:
        """Update actor network"""
        action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_temperature(self, state) -> float:
        """Update temperature parameter"""
        action, log_prob = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str) -> None:
        """Save agent state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps
        }
        
        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
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
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.total_steps = checkpoint.get('total_steps', 0) 