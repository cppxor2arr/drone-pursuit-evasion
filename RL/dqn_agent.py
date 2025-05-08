import torch
import torch.nn as nn

import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from gymnasium.spaces import Discrete,Box
from enum import Enum
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, num_obs:int, num_action:int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_obs, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_action),
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, limit):
        self.buffer = deque(maxlen=limit)

    def add(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, done = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_prime, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# batch로 input이 들어오면 batch로 대응할 수 있는가? 
# supersuit에 vframe으로 모았을 때, 아니면 bootstrapping한 sample을 여러개 batch로 돌릴때,
# input observation:
class DroneDQNAgent():
    def __init__(self,observation_space:Box, action_space:Discrete, device:str): # gymnasium env or Petting zoo env, env:ParallelEnv
        #assert(len(observation_space.shape) == len(action_space.shape) == 1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        num_obs = np.prod(observation_space.shape)
        num_action = action_space.n
        self.q_net = QNetwork(num_obs, num_action)
        self.target_network = QNetwork(num_obs, num_action)
        self.update_target_network()

    def to(self, device:str) -> None:
        self.device = device
        self.target_network.to(device)
        self.q_net.to(device)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_net.state_dict())
    
    def explore(self) -> int:
        return self.action_space.sample()
    
    # def exploit(self, observation:np.ndarray) -> Discrete:
    #     with torch.no_grad():
    #         obs = torch.tensor(observation, dtype=torch.float32).to(self.device)
    #         target_max, target_idx = self.q_net(obs).max(dim=1)
    #     return target_idx
    def exploit(self, observation: ReplayBufferSamples) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) → (1, obs_dim)

        with torch.no_grad():
            _, action_idx = self.q_net(obs).max(dim=1)  # shape: (batch_size,)
        
        return action_idx.cpu().numpy()  # return np.ndarray of ints


    # use stable baseline replace buffer, and frame_stack_v1 supersuit. 
    # in order to support batch system, 
    def target_value(self, observation:np.ndarray):
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32).to(self.device)
            target_max, target_idx = self.target_network(obs).max(dim=1)
        return target_max
    
    def q_value(self, observation):
        return 
    

# TODO: observation이 input으로 들어올 때엔 batch로 들어올 수 있으므로, 조심하기 
# replace buffer는 stable baseline에서 구현 된 것 가져오기 
# vstack은 supersuit으로 된 것 가져오기 
    
if __name__ == "__main__":
    from .. import task
    # LidarDroneBaseEnv
    from supersuit import frame_stack_v1
    from stable_baselines3.common.buffers import ReplayBuffer

    env = LidarDroneBaseEnv(render_mode="human")
    env = frame_stack_v1(env)
    env.reset(flight_mode=7, start_pos=np.array([[1,1,1], [-1,-1,1]]),start_orn=np.array([[0,0,0], [0,0,0]]))
    drone1 = DroneDQNAgent()
