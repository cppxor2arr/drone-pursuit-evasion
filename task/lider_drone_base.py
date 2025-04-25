from gymnasium import spaces
import numpy as np
from typing import Any, Literal
import pybullet as p
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import (
    MAQuadXHoverEnv,
)  # MAQuadXBaseEnv
import math


class Actions:
    def __init__(self, length: float = 1.0):
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
    def __init__(
        self,
        lidar_reach: float,
        num_ray: int,
        angle_representations: str = "quaternion",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lidar_reach = lidar_reach
        self.num_ray = num_ray

        translation_dim = velocity_dim = 3
        target_position = 3
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(translation_dim + velocity_dim + target_position + self.num_ray,),
            dtype=np.float64,
        )
        self.actions = Actions(self.lidar_reach)
        self._action_space = spaces.Discrete(len(self.actions))

    def laycast(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        # You need to change this term to change lidar observation.
        R = p.getMatrixFromQuaternion(quaternion)
        R = np.array(R).reshape(3, 3)
        ray_from = [position for _ in range(self.num_ray)]
        ray_to = [
            position
            + R
            @ np.array(
                [
                    self.lidar_reach
                    * math.sin(2.0 * math.pi * float(d_theta) / self.num_ray),
                    self.lidar_reach
                    * math.cos(2.0 * math.pi * float(d_theta) / self.num_ray),
                    0,
                ]
            )
            for d_theta in range(self.num_ray)
        ]

        # code for visualize
        # line_color = [0, 1, 0]
        # for i in range(self.num_ray):
        #     p.addUserDebugLine(ray_from[i], ray_to[i], line_color)
        # ---

        NUM_THREAD = 1
        distances = [
            normalized_dist * self.lidar_reach
            for ray_id, body_id, normalized_dist, xyz, direction in p.rayTestBatch(
                ray_from, ray_to, NUM_THREAD
            )
        ]
        return np.array(distances)

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        raw_state = self.compute_attitude_by_id(agent_id)

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

        return np.concatenate(
            [
                lin_pos,
                lin_vel,
                self.laycast(lin_pos, quaternion),
            ],
            axis=-1,
        )

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int, prev_observations: np.ndarray
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        # initialize
        reward = 0.0
        term = False
        trunc = self.step_count > self.max_steps
        info = dict()

        # collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 10.0
            info["collision"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 10.0
            info["out_of_bounds"] = True
            term |= True

        # reward
        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.aviary.state(agent_id)[-1] - self.start_pos[agent_id]
            )

            # how far are we from 0 roll pitch
            angular_distance = np.linalg.norm(self.aviary.state(agent_id)[1][:2])

            reward -= float(linear_distance + angular_distance * 0.1)
            reward += 1.0

        return term, trunc, reward, info

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        NUM_AGENT = len(actions.items())
        assert NUM_AGENT == 2
        for k, v in actions.items():
            agent_idx = self.agent_name_mapping[k]
            target_idx = int((agent_idx + 1) % NUM_AGENT)
            x, y, z = self.compute_observation_by_id(target_idx)[:3]
            dxdydz = self.actions[v]

            next_x = x + dxdydz[0]
            next_y = y + dxdydz[1]
            next_z = z + dxdydz[2]

            yaw_angle = 1.0
            next_pose = np.array([next_x, next_y, yaw_angle, next_z])

            self.aviary.set_setpoint(agent_idx, next_pose)

        observations = dict()
        terminations = {k: False for k in self.agents}
        truncations = {k: False for k in self.agents}
        rewards = {k: 0.0 for k in self.agents}
        infos = {k: dict() for k in self.agents}

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):
            self.aviary.step()
            self.update_states()

            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(
                    ag_id, self.prev_observations
                )
                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag].update(info)

                # compute observations
                observations[ag] = np.concatenate(
                    [
                        self.compute_observation_by_id(ag_id),
                        self.compute_observation_by_id((ag_id + 1) % NUM_AGENT)[:3],
                    ]  # append pursuit and observation target
                )
        # increment step count and cull dead agents for the next round
        self.prev_observations = observations.copy()
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    def reset(
        self, seed=None, options=dict()
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observations, infos = super().reset(seed, options)
        observations = {
            ag: np.concatenate([
                    self.compute_observation_by_id(self.agent_name_mapping[ag]),
                    self.compute_observation_by_id((self.agent_name_mapping[ag]+1) % self.num_agents)[:3] 
                ] # append pursuit and observation target
                )
            for ag in self.agents
        }
        self.prev_observations = observations.copy()
        return observations, infos
