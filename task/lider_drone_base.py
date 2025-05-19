from gymnasium import spaces
import numpy as np
from typing import Any, Literal, Dict, Optional, List, Tuple, Union
import pybullet as p
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import (
    MAQuadXHoverEnv,
)  # MAQuadXBaseEnv
import math
from enum import Enum


class DroneRole(Enum):
    PURSUER = 0
    EVADER = 1
    HOVER = 2
    RANDOM = 3
    GREEDY = 4


class DroneConfig:
    """Configuration class for a drone in the environment"""
    def __init__(
        self,
        role: DroneRole,
        start_pos: np.ndarray = None,
        start_orn: np.ndarray = None,
        action_length: float = 7.0,
        is_training: bool = True,
        resume_from: str = None,
        name: str = None
    ):
        self.role = role
        self.start_pos = start_pos  # 3D position [x, y, z]
        self.start_orn = start_orn  # 3D orientation [roll, pitch, yaw]
        self.action_length = action_length  # Length of actions for this drone
        self.is_training = is_training  # Whether this drone is being trained
        self.resume_from = resume_from  # Path to model to resume from
        self.name = name  # Optional name (will be auto-assigned if None)


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
        drone_configs: Optional[List[DroneConfig]] = None,
        render_simulation: bool = False,  # Whether to render the simulation
        *args,
        **kwargs,
    ):
        # Set render_mode based on render_simulation flag
        if not render_simulation and 'render_mode' in kwargs and kwargs['render_mode'] == 'human':
            kwargs['render_mode'] = None
        
        # Handle start positions and orientations from drone_configs
        if drone_configs is not None:
            start_pos = np.array([config.start_pos for config in drone_configs if config.start_pos is not None])
            if len(start_pos) > 0:
                kwargs['start_pos'] = start_pos
            
            start_orn = np.array([config.start_orn for config in drone_configs if config.start_orn is not None])
            if len(start_orn) > 0:
                kwargs['start_orn'] = start_orn
        
        # Initialize base environment
        super().__init__(*args, **kwargs)

        self.lidar_reach = lidar_reach
        self.num_ray = num_ray
        self.render_simulation = render_simulation

        velocity_dim = 3
        target_position = 3
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(velocity_dim + target_position + self.num_ray,),
            dtype=np.float64,
        )
        self.actions = Actions()# 1 10 5 3 7 # default ation length 1 
        self._action_space = spaces.Discrete(len(self.actions))
        
        # Store drone configurations
        self.drone_configs = drone_configs
        
        # Agent roles will be initialized on reset
        self.agent_roles = {}
        
        # Reward parameters
        self.capture_threshold = 0.1  # 10cm threshold for capture
        self.pursuer_capture_reward = 20.0   # Reward for pursuer when capturing evader
        
        # Modified rewards for evader
        self.evader_capture_penalty = -10.0  # Reduced penalty for evader when captured
        self.collision_penalty = -30.0       # Increased penalty for collision with objects
        self.out_of_bounds_penalty = -30.0   # Increased penalty for going out of bounds
        
        self.distance_reward_coef = 2.0      # Coefficient for distance-based rewards
        self.evader_survival_reward = 0.1    # Small reward for evader surviving each step
        self.evader_safe_distance = 2.0      # Distance at which evader gets extra reward
        self.evader_safe_distance_reward = 0.5 # Reward for maintaining safe distance
        self.pursuer_proximity_threshold = 1.0 # Threshold for pursuer proximity bonus
        self.pursuer_proximity_coef = 1.0    # Coefficient for pursuer proximity bonus
        
        # Episode time rewards to encourage longer episodes
        self.time_reward_coef = 0.01         # Small reward for each step
        self.max_episode_steps = 500         # Maximum episode length

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

        # Visualize rays if simulation is rendered
        if self.render_simulation:
            line_color = [0, 1, 0]
            for i in range(self.num_ray):
                p.addUserDebugLine(ray_from[i], ray_to[i], line_color, lifeTime=0.1)

        NUM_THREAD = 1
        try:
            ray_results = p.rayTestBatch(ray_from, ray_to, NUM_THREAD)
            distances = [
                normalized_dist * self.lidar_reach
                for ray_id, body_id, normalized_dist, xyz, direction in ray_results
            ]
            return np.array(distances)
        except (TypeError, AttributeError):
            # Handle the case where p might be None or rayTestBatch fails
            return np.ones(self.num_ray) * self.lidar_reach  # Return max distances

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        try:
            raw_state = self.compute_attitude_by_id(agent_id)
            ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

            return np.concatenate(
                [
                    lin_vel,
                    self.laycast(lin_pos, quaternion),
                ],
                axis=-1,
            )
        except Exception as e:
            print(f"Error computing observation for agent {agent_id}: {e}")
            # Return a safe fallback observation with the correct shape
            return np.zeros((3 + self.num_ray,), dtype=np.float64)
    
    def compute_distance_between_agents(self, agent_id: int, other_agent_id: int) -> float:
        """Compute the distance between two agents"""
        try:
            linear_pos = self.aviary.state(agent_id)[-1]
            other_linear_pos = self.aviary.state(other_agent_id)[-1]
            return np.linalg.norm(other_linear_pos - linear_pos)
        except Exception as e:
            print(f"Error computing distance between agents: {e}")
            return float('inf')  # Return a large distance on error
    
    def compute_prev_distance_between_agents(self, agent_id: int, other_agent_id: int) -> float:
        """Compute the previous distance between two agents using prev_observations"""
        agent_name = f"uav_{agent_id}"
        other_agent_name = f"uav_{other_agent_id}"
        
        if not hasattr(self, 'prev_observations') or agent_name not in self.prev_observations or other_agent_name not in self.prev_observations:
            # If previous observations are not available, return current distance
            return self.compute_distance_between_agents(agent_id, other_agent_id)
        
        try:
            prev_linear_pos = self.prev_observations[other_agent_name][-3:]
            prev_other_linear_pos = self.prev_observations[agent_name][-3:]
            return np.linalg.norm(prev_other_linear_pos - prev_linear_pos)
        except Exception as e:
            print(f"Error computing previous distance: {e}")
            return float('inf')
    
    def compute_pursuer_reward(
        self, agent_id: int, other_agent_id: int, collision: bool, out_of_bounds: bool
    ) -> tuple[float, bool]:
        """Compute reward for pursuer agent"""
        reward = 0.0
        
        # Get current and previous distances
        current_distance = self.compute_distance_between_agents(agent_id, other_agent_id)
        prev_distance = self.compute_prev_distance_between_agents(agent_id, other_agent_id)
        
        # Check if capture occurred
        capture = current_distance < self.capture_threshold
        
        # Penalties
        if collision:
            reward += self.collision_penalty
        if out_of_bounds:
            reward += self.out_of_bounds_penalty
        
        # Capture reward
        if capture:
            reward += self.pursuer_capture_reward
        
        # Distance-based reward (reward for getting closer)
        reward += self.distance_reward_coef * (prev_distance - current_distance)
        
        # Proximity reward (more reward as pursuer gets closer)
        if current_distance < self.pursuer_proximity_threshold:
            reward += self.pursuer_proximity_coef * (self.pursuer_proximity_threshold - current_distance)
        
        return reward, capture
    
    def compute_evader_reward(
        self, agent_id: int, other_agent_id: int, collision: bool, out_of_bounds: bool
    ) -> tuple[float, bool]:
        """Compute reward for evader agent"""
        reward = 0.0
        
        # Get current and previous distances
        current_distance = self.compute_distance_between_agents(agent_id, other_agent_id)
        prev_distance = self.compute_prev_distance_between_agents(agent_id, other_agent_id)
        
        # Check if capture occurred
        capture = current_distance < self.capture_threshold
        
        # Add time-based survival reward
        reward += self.time_reward_coef * self.step_count
        
        # Penalties - increased for collision and out-of-bounds to make capture preferable
        if collision:
            reward += self.collision_penalty
            print(f"Evader collision penalty: {self.collision_penalty}")
        if out_of_bounds:
            reward += self.out_of_bounds_penalty
            print(f"Evader out-of-bounds penalty: {self.out_of_bounds_penalty}")
        
        # Capture penalty - reduced to make it less bad than collisions
        if capture:
            reward += self.evader_capture_penalty
            print(f"Evader capture penalty: {self.evader_capture_penalty}")
        
        # Distance-based reward (reward for getting farther)
        distance_reward = self.distance_reward_coef * (current_distance - prev_distance)
        reward += distance_reward
        
        # Survival bonus
        reward += self.evader_survival_reward
        
        # Bonus for maintaining safe distance
        if current_distance > self.evader_safe_distance:
            reward += self.evader_safe_distance_reward
        
        return reward, capture
    
    def assign_agent_roles(self):
        """Assign roles to agents based on configuration"""
        self.agent_roles = {}
        
        # If we have drone configs, use them to assign roles
        for i, config in enumerate(self.drone_configs):
            if i < len(self.agents):
                agent_name = self.agents[i]
                self.agent_roles[agent_name] = config.role

        

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int, prev_observations: dict[str, Any]
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        # initialize
        reward = 0.0
        term = False
        trunc = self.step_count > self.max_steps
        info = dict()

        try:
            # collision check
            if self.aviary and hasattr(self.aviary, 'contact_array') and hasattr(self.aviary, 'drones'):
                collision = np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id])
                if collision:
                    info["collision"] = True
                    term |= True

            # exceed flight dome check
            if self.aviary:
                agent_pos = self.aviary.state(agent_id)[-1]
                out_of_bounds = np.linalg.norm(agent_pos) > self.flight_dome_size
                if out_of_bounds:
                    info["out_of_bounds"] = True
                    term |= True
            else:
                out_of_bounds = False

            # Get the other agent ID (assuming 2 agents)
            other_agent_id = 1 if agent_id == 0 else 0
            
            # Assign roles if not already assigned
            if len(self.agent_roles) == 0:
                self.assign_agent_roles()
            
            agent_name = [name for name, id in self.agent_name_mapping.items() if id == agent_id][0]
            
            # Compute role-based rewards
            if agent_name in self.agent_roles:
                if self.agent_roles[agent_name] == DroneRole.PURSUER:
                    reward, capture = self.compute_pursuer_reward(agent_id, other_agent_id, collision, out_of_bounds)
                else:  # EVADER
                    reward, capture = self.compute_evader_reward(agent_id, other_agent_id, collision, out_of_bounds)
                
                # Add capture information to info
                if capture:
                    info["capture"] = True
                    term |= True  # End episode on capture
            else:
                # Fallback to basic reward
                linear_distance = self.compute_distance_between_agents(agent_id, other_agent_id)
                
                # Basic penalties
                if collision:
                    reward -= 10.0
                if out_of_bounds:
                    reward -= 10.0
                
                # Basic capture reward/penalty
                if linear_distance < self.capture_threshold:
                    if agent_id == 0:  # Assume pursuer is agent 0
                        reward += 10.0
                    else:  # Assume evader is agent 1
                        reward -= 10.0
                    info["capture"] = True
                    term |= True
        except Exception as e:
            print(f"Error in compute_term_trunc_reward_info_by_id: {e}")
            # In case of error, return safe values
            info["error"] = str(e)

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
        try:
            NUM_AGENT = len(actions.items())
            # 하나의 에이전트만 있는 경우를 확인하고 처리
            if NUM_AGENT < 2:
                # 하나의 에이전트만 있을 경우 에피소드를 종료시킵니다
                observations = {}
                rewards = {}
                terminations = {}
                truncations = {}
                infos = {}
                
                for agent_name in list(actions.keys()):
                    observations[agent_name] = np.zeros((6 + self.num_ray,), dtype=np.float64)
                    rewards[agent_name] = 0.0
                    terminations[agent_name] = True  # 에피소드 종료
                    truncations[agent_name] = False
                    infos[agent_name] = {"error": "Not enough agents"}
                
                return observations, rewards, terminations, truncations, infos
            
            assert NUM_AGENT == 2, f"Expected 2 agents, got {NUM_AGENT}"
            
            for k, v in actions.items():
                agent_idx = self.agent_name_mapping[k]
                x, y, z = self.compute_observation_by_id((agent_idx + 1) % NUM_AGENT)[-3:]
                
                # Handle different drone roles
                agent_name = k
                drone_role = self.agent_roles.get(agent_name, DroneRole.PURSUER)  # Default to PURSUER if no role
                
                # For HOVER role, don't move
                if drone_role == DroneRole.HOVER:
                    # Set target position as current position for hovering
                    current_pos = self.aviary.state(agent_idx)[-1]
                    current_yaw = 0.0  # Use default yaw
                    next_pose = np.array([current_pos[0], current_pos[1], current_yaw, current_pos[2]])
                    self.aviary.set_setpoint(agent_idx, next_pose)
                    continue  # Skip the rest of the processing for hovering drones
                    
                # For RANDOM role, choose a random action
                if drone_role == DroneRole.RANDOM:
                    v = np.random.randint(0, len(self.actions))
                
                # Get action length for this drone from config if available
                action_length = 1.0
                if self.drone_configs is not None:
                    for config in self.drone_configs:
                        if config.role == drone_role:
                            action_length = config.action_length
                            break
                            
                # Scale actions based on action_length
                dxdydz = self.actions[v] * action_length

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
                if self.aviary is not None:
                    self.aviary.step()
                self.update_states()

            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(
                    ag_id, getattr(self, 'prev_observations', {})
                )
                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag].update(info)

                # compute observations
                try:
                    observations[ag] = np.concatenate(
                        [
                            self.compute_observation_by_id(ag_id),
                            self.compute_attitude_by_id((ag_id + 1) % NUM_AGENT)[3],
                        ]  # append pursuit and observation target
                    )
                except Exception as e:
                    print(f"Error computing observations: {e}")
                    # Return a safe fallback observation
                    observations[ag] = np.zeros((6 + self.num_ray,), dtype=np.float64)
                        
            # increment step count and cull dead agents for the next round
            self.prev_observations = observations.copy()
            self.step_count += 1
            self.agents = [
                agent
                for agent in self.agents
                if not (terminations[agent] or truncations[agent])
            ]

            # Transform other drone's position to relative position (reference frame: self)
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                left_ag_id = (ag_id - 1) % NUM_AGENT
                left_ag = f"uav_{left_ag_id}"

                observations[ag][-3:] -= self.prev_observations[left_ag][-3:]

            return observations, rewards, terminations, truncations, infos
        
        except Exception as e:
            print(f"Error in step method: {e}")
            # Return safe fallback values
            observations = {agent: np.zeros((6 + self.num_ray,), dtype=np.float64) for agent in self.agents}
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}  # End episode on error
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {"error": str(e)} for agent in self.agents}
            
            return observations, rewards, terminations, truncations, infos

    def reset(
        self, seed=None, options=dict()
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        try:
            observations, infos = super().reset(seed, options)

            concaveSphereCollisionId = self.aviary.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName="hi_res_sphere.obj",
                meshScale=[-self.flight_dome_size] * 3,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            concaveSphereVisualId = self.aviary.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName="hi_res_sphere.obj",
                meshScale=[-self.flight_dome_size] * 3,
                rgbaColor=[0.2, 0.2, 1.0, 0.8],
                specularColor=[0.4, 0.4, 0.4],
            )
            concaveSphereId = self.aviary.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=concaveSphereCollisionId,
                baseVisualShapeIndex=concaveSphereVisualId,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )
            convexSphereVisualId = self.aviary.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName="hi_res_sphere.obj",
                meshScale=[self.flight_dome_size] * 3,
                rgbaColor=[0.2, 0.2, 1.0, 0.8],
                specularColor=[0.4, 0.4, 0.4],
            )
            convexSphereId = self.aviary.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=convexSphereVisualId,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )
            obs_radius = 0.5
            obs_height = 7.0
            obstacles = [[2*i,2*j] for i in range(-3, 3) for j in range(-3, 3)]
            for obs_loc in obstacles:
                obsCollisionId = self.aviary.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=obs_radius, height=obs_height)
                obsId = self.aviary.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=obsCollisionId,
                basePosition=obs_loc + [obs_height / 2.0],
                )
            self.aviary.register_all_new_bodies()

            NUM_AGENT = 2

            try:
                # Compute observations with target positions
                observations = {
                    ag: np.concatenate(
                        [
                            self.compute_observation_by_id(self.agent_name_mapping[ag]),
                            self.compute_attitude_by_id((self.agent_name_mapping[ag] + 1) % self.num_agents)[3],
                        ]  # append pursuit and observation target
                    )
                    for ag in self.agents
                }
            except Exception as e:
                print(f"Error computing initial observations: {e}")
                # Return safe fallback observations
                observations = {
                    ag: np.zeros((6 + self.num_ray,), dtype=np.float64)
                    for ag in self.agents
                }
            
            self.prev_observations = observations.copy()
            
            # Assign agent roles based on configuration
            self.assign_agent_roles()
            
            # Log the roles
            role_info = {}
            for agent_name, role in self.agent_roles.items():
                role_info[agent_name] = role.name
            print(f"Agent roles: {role_info}")

            # Transform other drone's position to relative position (reference frame: self)
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                left_ag_id = (ag_id - 1) % NUM_AGENT
                left_ag = f"uav_{left_ag_id}"

                observations[ag][-3:] -= self.prev_observations[left_ag][-3:]

            return observations, infos
        
        except Exception as e:
            print(f"Error in reset method: {e}")
            # Return safe fallback values
            observations = {agent: np.zeros((6 + self.num_ray,), dtype=np.float64) for agent in self.agents}
            infos = {"error": str(e)}
            
            return observations, infos
