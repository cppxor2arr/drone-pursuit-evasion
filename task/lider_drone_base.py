from gymnasium import spaces
import numpy as np
from typing import Any, Literal, Dict, Optional, List, Tuple, Union
import pybullet as p
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import (
    MAQuadXHoverEnv,
)  # MAQuadXBaseEnv
import math
from enum import Enum
import os


class DroneRole(Enum):
    PURSUER = 0
    EVADER = 1

class AgentType(Enum):
    DQN = "dqn"
    PPO = "ppo"
    SAC = "sac"
    RANDOM = "random"
    HOVERING = "hovering"

class EnvironmentStage(Enum):
    """Environment complexity stages"""
    OPEN = "open"          # Stage 1: No obstacles (open space)
    SINGLE_OBSTACLE = "single"  # Stage 2: One cylinder in the middle
    MULTIPLE_OBSTACLES = "multiple"  # Stage 3: Current setting (grid of cylinders)


class DroneConfig:
    """Configuration class for a drone in the environment"""
    def __init__(
        self,
        role: DroneRole,
        agent_type: AgentType = AgentType.DQN,  # Type of agent using enum
        start_pos: np.ndarray = None,
        start_orn: np.ndarray = None,
        action_length: float = 7.0,
        is_training: bool = True,
        resume_from: str = None,
        name: str = None
    ):
        self.role = role
        self.agent_type = agent_type  # Now using AgentType enum
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
            np.array([0, 0, 0]),  # Action 0: No movement (for hovering)
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
        env_config,  # Environment configuration object
        scenario_config,  # Scenario configuration object
        drone_configs: Optional[List[DroneConfig]],
        *args,
        **kwargs,
    ):
        # Use config objects (new approach)
        self.lidar_reach = env_config.lidar_reach
        self.num_ray = env_config.num_ray
        self.render_simulation = env_config.render_simulation
        self.environment_stage = EnvironmentStage(env_config.get('stage', 'multiple'))
        
        # Extract reward parameters from config
        self.capture_threshold = env_config.get('capture_threshold', 0.1)
        self.pursuer_capture_reward = env_config.get('pursuer_capture_reward', 20.0)
        self.evader_capture_penalty = env_config.get('evader_capture_penalty', -10.0)
        self.collision_penalty = env_config.get('collision_penalty', -30.0)
        self.out_of_bounds_penalty = env_config.get('out_of_bounds_penalty', -30.0)
        self.distance_reward_coef = env_config.get('distance_reward_coef', 2.0)
        self.evader_survival_reward = env_config.get('evader_survival_reward', 0.1)
        self.evader_safe_distance = env_config.get('evader_safe_distance', 2.0)
        self.evader_safe_distance_reward = env_config.get('evader_safe_distance_reward', 0.5)
        self.pursuer_proximity_threshold = env_config.get('pursuer_proximity_threshold', 1.0)
        self.pursuer_proximity_coef = env_config.get('pursuer_proximity_coef', 1.0)
        self.time_reward_coef = env_config.get('time_reward_coef', 0.01)
        self.max_episode_steps = env_config.get('max_episode_steps', 500)
        
        # Get drone configs from scenario_config if available
        if scenario_config is not None and hasattr(scenario_config, 'drones'):
            drone_configs = self._create_drone_configs_from_scenario(scenario_config)


        # Set render_mode based on render_simulation flag
        if not self.render_simulation and 'render_mode' in kwargs and kwargs['render_mode'] == 'human':
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
        super().__init__(*args, **kwargs,flight_mode=7) ## DON"T CHANGE FLIGHT MODE
        self.num_possible_agents = len(drone_configs)
        self.possible_agents = [config.name for config in drone_configs]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        velocity_dim = 3
        relative_position = 3
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(velocity_dim + self.num_ray + relative_position,),
            dtype=np.float64,
        )
        self.actions = Actions()
        self._action_space = spaces.Discrete(len(self.actions))
        
        # Store drone configurations
        self.drone_configs = drone_configs
        
        # Agent roles will be initialized on reset
        self.agent_roles = {}

    def _create_drone_configs_from_scenario(self, scenario_config):
        """Create DroneConfig objects from scenario configuration"""
        configs = []
        for drone in scenario_config.drones:
            role = getattr(DroneRole, drone.role)
            agent_type = AgentType(drone.agent_type)
            configs.append(DroneConfig(
                role=role,
                agent_type=agent_type,
                start_pos=np.array(drone.start_pos),
                start_orn=np.array(drone.start_orn),
                action_length=drone.action_length,
                is_training=drone.is_training,
                resume_from=drone.resume_from,
                name=drone.name
            ))
        return configs

    def create_obstacles(self):
        """Create obstacles based on environment stage"""
        if self.environment_stage == EnvironmentStage.OPEN:
            # Stage 1: No obstacles (open space)
            print("Environment Stage 1: Open space (no obstacles)")
            return
        
        elif self.environment_stage == EnvironmentStage.SINGLE_OBSTACLE:
            # Stage 2: Single cylinder in the middle
            print("Environment Stage 2: Single obstacle in center")
            obs_radius = 0.5
            obs_height = 7.0
            obs_loc = [0.0, 0.0]  # Center position
            
            obsCollisionId = self.aviary.createCollisionShape(
                shapeType=p.GEOM_CYLINDER, 
                radius=obs_radius, 
                height=obs_height
            )
            obsId = self.aviary.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=obsCollisionId,
                basePosition=obs_loc + [obs_height / 2.0],
            )
        
        elif self.environment_stage == EnvironmentStage.MULTIPLE_OBSTACLES:
            # Stage 3: Multiple obstacles in grid pattern (current setting)
            print("Environment Stage 3: Multiple obstacles (grid pattern)")
            obs_radius = 0.5
            obs_height = 7.0
            obstacles = [[2*i, 2*j] for i in range(-3, 3) for j in range(-3, 3)]
            
            for obs_loc in obstacles:
                obsCollisionId = self.aviary.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER, 
                    radius=obs_radius, 
                    height=obs_height
                )
                obsId = self.aviary.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=obsCollisionId,
                    basePosition=obs_loc + [obs_height / 2.0],
                )

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

        # # Visualize rays if simulation is rendered
        # if self.render_simulation:
        #     line_color = [0, 1, 0]
        #     for i in range(self.num_ray):
        #         p.addUserDebugLine(ray_from[i], ray_to[i], line_color, lifeTime=0.1)

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
        raw_state = self.compute_attitude_by_id(agent_id)
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state
        other_lin_pos = self.compute_attitude_by_id((agent_id + 1) % 2)[3]

        return np.concatenate(
            [
                lin_vel,  # Linear velocity (3D)
                self.laycast(lin_pos, quaternion),  # LIDAR data
                other_lin_pos - lin_pos,  # Other drone's relative position
            ],
            axis=-1,
        )
    
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
        
        # ADDED: Altitude maintenance reward - penalize deviation from target altitude
        try:
            current_pos = self.aviary.state(agent_id)[-1]
            altitude_deviation = abs(current_pos[2] - self.target_altitude)
            altitude_penalty = -self.altitude_penalty_coef * altitude_deviation
            reward += altitude_penalty
        except Exception as e:
            pass  # Skip altitude penalty if position unavailable
        
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
        
        # ADDED: Altitude maintenance reward - penalize deviation from target altitude
        try:
            current_pos = self.aviary.state(agent_id)[-1]
            altitude_deviation = abs(current_pos[2] - self.target_altitude)
            altitude_penalty = -self.altitude_penalty_coef * altitude_deviation
            reward += altitude_penalty
        except Exception as e:
            pass  # Skip altitude penalty if position unavailable
        
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
        NUM_AGENT = len(actions.items())

        #assert NUM_AGENT == 2, f"Expected 2 agents, got {NUM_AGENT}
        for agent_name, action_input in actions.items():
            agent_idx = self.agent_name_mapping[agent_name]
            # Get THIS agent's current position, not the other agent's position
            raw_state = self.compute_attitude_by_id(agent_idx)
            ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state
            x, y, z = lin_pos
            
            # Handle different drone roles for action length only
            drone_role = self.agent_roles.get(agent_name, DroneRole.PURSUER)  # Default to PURSUER if no role
            
            drone_config = next(config for config in self.drone_configs if config.name == agent_name)
            action_length = drone_config.action_length
            
            if drone_config.agent_type == AgentType.HOVERING:
                next_x, next_y, next_z = drone_config.start_pos
            else:
                dxdydz = self.actions[action_input] * action_length
                next_x = x + dxdydz[0]
                next_y = y + dxdydz[1]
                next_z = z + dxdydz[2]

            current_yaw = ang_pos[2]  # yaw is the Z component of angular position
            next_pose = np.array([next_x, next_y, current_yaw, next_z])
            
            self.aviary.set_setpoint(agent_idx, next_pose)

        observations = dict()
        terminations = {agent_name: False for agent_name in self.agents}
        truncations = {agent_name: False for agent_name in self.agents}
        rewards = {agent_name: 0.0 for agent_name in self.agents}
        infos = {agent_name: dict() for agent_name in self.agents}

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
            observations[ag] = self.compute_observation_by_id(ag_id)
                    
        # increment step count and cull dead agents for the next round
        self.prev_observations = observations.copy()
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]
        # If any agent terminated, terminate all remaining agents
        if any(terminations.values()):
            for agent in self.agents:
                terminations[agent] = True
                infos[agent]["forced_termination"] = True

        return observations, rewards, terminations, truncations, infos
        

    def reset(
        self, seed=None, options=dict()
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observations, infos = super().reset(seed, options)

        # Use absolute path for mesh file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_file = os.path.join(os.path.dirname(script_dir), "hi_res_sphere.obj")
        
        # Create dome boundaries
        concaveSphereCollisionId = self.aviary.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_file,
            meshScale=[-self.flight_dome_size] * 3,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        concaveSphereVisualId = self.aviary.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_file,
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
            fileName=mesh_file,
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
        
        # Create obstacles based on environment stage
        self.create_obstacles()
        
        # Register all new bodies
        self.aviary.register_all_new_bodies()

        # Compute observations with target positions
        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
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

        return observations, infos
