from gymnasium import spaces
import numpy as np
from typing import Any, Literal
import pybullet as p
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv #MAQuadXBaseEnv
import math

class Actions:
    def __init__(self, length: float = 1.):
        self.norm = 1 / np.sqrt(3)
        self.length = length
        self.vectors = [
            np.array([+1, +1, +1]) * self.norm,
            np.array([+1, +1, -1]) * self.norm,
            np.array([+1, -1, +1]) * self.norm,
            np.array([+1, -1, -1]) * self.norm,
            np.array([-1, +1, +1]) * self.norm,
            np.array([-1, +1, -1]) * self.norm,
            np.array([-1, -1, +1]) * self.norm,
            np.array([-1, -1, -1]) * self.norm,
            np.array([+1, 0, 0]),
            np.array([0, +1, 0]),
            np.array([0, 0, +1]),
            np.array([-1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, -1]),
        ]

    def __getitem__(self, idx):
        return self.vectors[idx] * self.length

    def __len__(self):
        return len(self.vectors)

    def __iter__(self):
        return (v * self.length for v in self.vectors)

class LidarDroneBaseEnv(MAQuadXHoverEnv):
    def __init__(self, lidar_reach: float, num_ray: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.lidar_reach = lidar_reach
        self.num_ray = num_ray

        translation_dim = velocity_dim = 3
        target_position = 3
        self._observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(translation_dim + velocity_dim + target_position + self.num_ray, ),
                    dtype=np.float64,
                )
        self._action_space = spaces.Discrete(len(Actions(self.lidar_reach)))

    def laycast(self, position:np.ndarray, quaternion:np.ndarray) -> np.ndarray:
        # You need to change this term to change lidar observation.
        R = p.getMatrixFromQuaternion(quaternion)
        R = np.array(R).reshape(3,3)
        ray_from = [position for _ in range(self.num_ray)]
        ray_to = [ position + R @ np.array([
                                self.lidar_reach * math.sin(2. * math.pi * float(d_theta) / self.num_ray),
                                self.lidar_reach * math.cos(2. * math.pi * float(d_theta) / self.num_ray),
                                0
                                ])
                                for d_theta in range(self.num_ray)]
        
        # code for visualize
        # line_color = [0, 1, 0]
        # for i in range(self.num_ray):
        #     p.addUserDebugLine(ray_from[i], ray_to[i], line_color)
        # ---
        
        NUM_THREAD = 1
        distances = [normalized_dist* self.lidar_reach 
                     for ray_id, body_id, normalized_dist, xyz, direction in p.rayTestBatch(ray_from,ray_to,NUM_THREAD)]
        return np.array(distances)
    
    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        raw_state = self.compute_attitude_by_id(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

        # depending on angle representation, return the relevant thing

        if self.angle_representation == 0:
            return np.concatenate(
                [
                    lin_vel,
                    lin_pos,
                    self.laycast(lin_pos, quaternion),
                    self.past_actions[agent_id],
                    self.start_pos[agent_id],
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            return np.concatenate(
                [
                    lin_vel,
                    lin_pos,
                    self.laycast(lin_pos, quaternion),
                    self.past_actions[agent_id],
                    self.start_pos[agent_id],
                ],
                axis=-1,
            )
        else:
            raise AssertionError("Not supposed to end up here!")
