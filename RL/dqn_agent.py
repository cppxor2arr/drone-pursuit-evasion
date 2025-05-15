import torch
import torch.nn as nn

import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from gymnasium.spaces import Discrete,Box
from enum import Enum
import random
import os
import math
from collections import deque
from datetime import datetime

class QNetwork(nn.Module):
    def __init__(self, num_obs:int, num_action:int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_obs, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 84),
            nn.ReLU(),
            nn.Linear(84, num_action),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, limit, device: str = "cpu"):
        self.buffer = deque(maxlen=limit)
        self.device = device

    def add(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, done = zip(*batch)
        
        # Convert lists to numpy arrays then to tensors and move to device
        return (
            torch.from_numpy(np.array(s)).float().to(self.device),
            torch.from_numpy(np.array(a)).long().to(self.device),
            torch.from_numpy(np.array(r)).float().to(self.device),
            torch.from_numpy(np.array(s_prime)).float().to(self.device),
            torch.from_numpy(np.array(done)).float().to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

    def to(self, device: str) -> None:
        """Move buffer to specified device"""
        self.device = device

# batch로 input이 들어오면 batch로 대응할 수 있는가? 
# supersuit에 vframe으로 모았을 때, 아니면 bootstrapping한 sample을 여러개 batch로 돌릴때,
# input observation:
class DroneDQNAgent():
    def __init__(
        self, 
        observation_space: Box, 
        action_space: Discrete, 
        device: str,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        learning_rate: float = 0.0005
    ):
        # 환경 공간 저장
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        
        # 하이퍼파라미터
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # 현재 스텝 및 엡실론 값
        self.steps_done = 0
        self.epsilon = epsilon_start
        
        # 네트워크 초기화
        num_obs = np.prod(observation_space.shape)
        num_action = action_space.n
        self.q_net = QNetwork(num_obs, num_action).to(device)  # Move to device immediately
        self.target_network = QNetwork(num_obs, num_action).to(device)  # Move to device immediately
        self.update_target_network()
        
        # 옵티마이저 초기화
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        # 손실 함수
        self.criterion = nn.MSELoss()

    def to(self, device: str) -> None:
        """모델을 지정된 디바이스로 이동"""
        self.device = device
        self.target_network.to(device)
        self.q_net.to(device)

    def update_target_network(self) -> None:
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_net.state_dict())
    
    def soft_update_target_network(self, tau: float = 0.005) -> None:
        """타겟 네트워크 소프트 업데이트"""
        for target_param, param in zip(self.target_network.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_epsilon(self) -> float:
        """현재 엡실론 값 반환"""
        return self.epsilon
    
    def update_epsilon(self) -> None:
        """엡실론 값 업데이트 (선형 감소)"""
        self.steps_done += 1
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps_done / self.epsilon_decay
        )
    
    def update_epsilon_cosine(self) -> None:
        """엡실론 값 업데이트 (코사인 감소)"""
        self.steps_done += 1
        if self.steps_done < self.epsilon_decay:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * self.steps_done / self.epsilon_decay))
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * cosine_decay
        else:
            self.epsilon = self.epsilon_end
    
    def select_action(self, observation: np.ndarray) -> int:
        """엡실론-그리디 정책에 따라 행동 선택"""
        # 엡실론으로 탐색 또는 활용 결정
        if random.random() < self.epsilon:
            return self.explore()
        else:
            return self.exploit(observation)[0]
    
    def explore(self) -> int:
        """무작위 행동 선택 (탐색)"""
        return self.action_space.sample()
    
    def exploit(self, observation: np.ndarray) -> np.ndarray:
        """현재 정책에 따라 최적 행동 선택 (활용)"""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) → (1, obs_dim)

        with torch.no_grad():
            _, action_idx = self.q_net(obs).max(dim=1)  # shape: (batch_size,)
        
        return action_idx.cpu().numpy()  # return np.ndarray of ints
    
    def predict_q_values(self, observation: np.ndarray) -> np.ndarray:
        """주어진 상태에 대한 Q-값 예측"""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) → (1, obs_dim)
            
        with torch.no_grad():
            q_values = self.q_net(obs)
            
        return q_values.cpu().numpy()
    
    def compute_td_loss(self, samples) -> torch.Tensor:
        """시간차 학습 손실 계산"""
        states, actions, rewards, next_states, dones = samples
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 현재 Q 값 계산
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 타겟 Q 값 계산
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        # 손실 계산
        loss = self.criterion(current_q, target_q)
        
        return loss
    
    def update(self, samples) -> float:
        """샘플로 네트워크 업데이트"""
        loss = self.compute_td_loss(samples)
        
        # 옵티마이저 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str, step: int = None, is_best: bool = False) -> str:
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if step is not None:
            save_path = f"{path}_{step}.pt"
        elif is_best:
            save_path = f"{path}_best.pt"
        else:
            save_path = f"{path}_final.pt"
        
        torch.save(self.q_net.state_dict(), save_path)
        return save_path
    
    def load(self, path: str) -> None:
        """모델 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()
        
    def train(self) -> None:
        """학습 모드 설정"""
        self.q_net.train()
        
    def eval(self) -> None:
        """평가 모드 설정"""
        self.q_net.eval()

    def create_buffer(self, buffer_size: int) -> ReplayBuffer:
        """Create a new replay buffer with the specified size"""
        return ReplayBuffer(buffer_size, self.device)

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
