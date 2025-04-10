from gymnasium import spaces
import numpy as np
from typing import Any, Literal
import pybullet as p
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv #MAQuadXBaseEnv
import math

class LidarDroneBaseEnv(MAQuadXHoverEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def laycast(self, position:np.ndarray, quaternion:np.ndarray, lidar_reach: float) -> np.ndarray:
        # You need to change this term to change lidar observation.
        NUM_RAY = 4 
        R = p.getMatrixFromQuaternion(quaternion)
        R = np.array(R).reshape(3,3)
        ray_from = [position for _ in range(NUM_RAY)]
        ray_to = [ R@ np.array([
                        lidar_reach * math.sin(2. * math.pi * float(d_theta) / NUM_RAY),
                        lidar_reach * math.cos(2. * math.pi * float(d_theta) / NUM_RAY),
                        1
                        ])
                  for d_theta in range(NUM_RAY)]
        
        # code for visualize
        line_color = [0, 1, 0]
        for i in range(NUM_RAY):
            p.addUserDebugLine(ray_from[i], ray_to[i], line_color)
        # ---
        
        distances = [normalized_dist* lidar_reach 
                     for ray_id, body_id, normalized_dist, xyz, direction in p.rayTestBatch(ray_from,ray_to,4)]
        return np.array(distances)
    
    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        raw_state = self.compute_attitude_by_id(agent_id)
        aux_state = self.aviary.aux_state(agent_id)

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
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.laycast(lin_pos, quaternion,4),
                    self.past_actions[agent_id],
                    self.start_pos[agent_id],
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            return np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.laycast(lin_pos, quaternion,4),
                    self.past_actions[agent_id],
                    self.start_pos[agent_id],
                ],
                axis=-1,
            )
        else:
            raise AssertionError("Not supposed to end up here!")
