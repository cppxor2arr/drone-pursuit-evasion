#!/usr/bin/env python3
"""
Visualization script for drone pursuit-evasion scenarios.
Loads pretrained models and demonstrates behaviors with visual simulation.
"""

import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time

from task.lider_drone_base import LidarDroneBaseEnv, DroneRole, DroneConfig, AgentType
from RL import DroneDQNAgent, DronePPOAgent, DroneSACAgent, RandomAgent, HoveringAgent


def setup_device(device_config: str) -> str:
    """Setup device based on configuration"""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def create_agent(agent_type: str, config: DictConfig, obs_space, act_space, device: str):
    """Create agent based on type specified in scenario"""
    from RL import DroneDQNAgent, DronePPOAgent, DroneSACAgent, RandomAgent, HoveringAgent
    
    # Convert string to enum if needed
    if isinstance(agent_type, str):
        agent_type = AgentType(agent_type)
    
    if agent_type == AgentType.DQN:
        agent_config = OmegaConf.create({
            "gamma": config.agent.gamma,
            "learning_rate": config.agent.learning_rate,
            "epsilon_start": 0.0,  # No exploration during visualization
            "epsilon_end": 0.0,
            "epsilon_decay": 1,
            "batch_size": config.agent.batch_size,
            "network": config.agent.network,
            "buffer_size": config.agent.buffer_size
        })
        return DroneDQNAgent(obs_space, act_space, agent_config, device)
    
    elif agent_type == AgentType.PPO:
        ppo_config = getattr(config, 'ppo', None)
        if ppo_config is None:
            ppo_config = OmegaConf.create({
                "gamma": 0.99,
                "learning_rate": 0.0003,
                "clip_range": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
                "ppo_epochs": 10,
                "batch_size": 256,
                "buffer_size": 2048,
                "network": {"hidden_dims": [256, 256]}
            })
        return DronePPOAgent(obs_space, act_space, ppo_config, device)
    
    elif agent_type == AgentType.SAC:
        sac_config = getattr(config, 'sac', None)
        if sac_config is None:
            sac_config = OmegaConf.create({
                "gamma": 0.99,
                "learning_rate": 0.0003,
                "tau": 0.005,
                "alpha": 0.2,
                "automatic_entropy_tuning": True,
                "target_update_interval": 1,
                "batch_size": 256,
                "buffer_size": 100000,
                "warmup_steps": 1000,
                "network": {"hidden_dims": [256, 256]}
            })
        return DroneSACAgent(obs_space, act_space, sac_config, device)
    
    elif agent_type == AgentType.RANDOM:
        return RandomAgent(obs_space, act_space, device)
    
    elif agent_type == AgentType.HOVERING:
        return HoveringAgent(obs_space, act_space, device)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_drone_configs(scenario_config):
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


def load_pretrained_agents(config: DictConfig, env, agent_names: list, device: str) -> dict:
    """Load pretrained agents from specified model paths"""
    agents = {}
    drone_configs = create_drone_configs(config.scenario)
    
    print("\n🤖 Loading pretrained agents...")
    
    for i, agent_name in enumerate(agent_names):
        role = env.agent_roles[agent_name]
        drone_config = drone_configs[i] if i < len(drone_configs) else None
        
        if drone_config is None:
            print(f"  ⚠️ No configuration found for agent {agent_name}")
            continue
            
        # Get spaces
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        # Create agent
        agent = create_agent(drone_config.agent_type, config, obs_space, act_space, device)
        
        # Load pretrained weights if specified
        model_path = None
        if hasattr(config.visualization, 'model_paths') and agent_name in config.visualization.model_paths:
            model_path = config.visualization.model_paths[agent_name]
        elif hasattr(config.visualization, 'weights_dir') and config.visualization.weights_dir:
            # Use specified weights directory
            weights_dir = Path(config.visualization.weights_dir)
        else:
            # Auto-detect latest weights from experiment
            experiment_name = getattr(config, 'experiment_name', 'drone_pursuit_evasion')
            weights_base = Path(getattr(config, 'paths', {}).get('weights_dir', 'weights'))
            
            # Try to find latest directory
            latest_dir = weights_base / experiment_name / "latest"
            if latest_dir.exists():
                weights_dir = latest_dir
                print(f"  🔍 Found latest weights: {weights_dir}")
            else:
                # Fallback: find most recent timestamped directory
                experiment_dir = weights_base / experiment_name
                if experiment_dir.exists():
                    timestamp_dirs = [d for d in experiment_dir.iterdir() 
                                    if d.is_dir() and d.name != "latest"]
                    if timestamp_dirs:
                        weights_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
                        print(f"  🔍 Found recent weights: {weights_dir}")
                    else:
                        weights_dir = None
                        print(f"  ⚠️ No weight directories found in {experiment_dir}")
                else:
                    weights_dir = None
                    print(f"  ⚠️ Experiment directory not found: {experiment_dir}")
        
        if weights_dir:
            # Auto-detect model path based on role
            role_name = role.name.lower()
            
            # Try different naming patterns
            possible_names = [
                f"{role_name}_final.pt",
                f"{role_name}_best.pt", 
                f"{role_name}_0.pt"
            ]
            
            for name in possible_names:
                potential_path = weights_dir / name
                if potential_path.exists():
                    model_path = potential_path
                    break
        
        if model_path and os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"  ✅ Loaded {drone_config.agent_type.value.upper()} {role.name} from: {model_path}")
            except Exception as e:
                print(f"  ❌ Failed to load model for {agent_name}: {e}")
                print(f"  ℹ️ Using randomly initialized {drone_config.agent_type.value.upper()} agent")
        else:
            if weights_dir:
                print(f"  ⚠️ No model found for {agent_name} ({role.name}) in {weights_dir}")
            print(f"  ℹ️ Using randomly initialized {drone_config.agent_type.value.upper()}")
        
        # Set to evaluation mode
        agent.eval()
        agents[agent_name] = agent
    
    return agents


def run_visualization(config: DictConfig):
    """Run the visualization simulation"""
    print(f"\n🎬 Starting Visualization: {config.visualization.get('name', 'Drone Pursuit-Evasion')}")
    print(f"Environment Stage: {config.environment.get('stage', 'multiple')}")
    print(f"Episodes to run: {config.visualization.episodes}")
    print(f"Episode length limit: {config.visualization.max_episode_steps}")
    
    # Get visualization settings
    slow_motion = config.visualization.get('slow_motion', False)
    sleep_time = config.visualization.get('sleep_time', 0.1)
    pause_between_episodes = config.visualization.get('pause_between_episodes', False)
    
    if slow_motion:
        print(f"🐌 Slow motion enabled: {sleep_time} seconds between steps")
    if pause_between_episodes:
        print(f"⏸️ Manual pause between episodes enabled")
    
    # Check if rendering is enabled
    render_enabled = config.environment.get('render_simulation', True)
    print(f"🎨 Rendering: {'enabled' if render_enabled else 'disabled'}")
    
    # Setup device
    device = setup_device(config.device)
    print(f"Device: {device}")
    
    # Create environment with visualization enabled
    env_config = config.environment.copy()
    # Force rendering for visualization (only use valid config keys)
    env_config.render_simulation = True
    
    env = LidarDroneBaseEnv(
        env_config=env_config,
        scenario_config=config.scenario,
        drone_configs=create_drone_configs(config.scenario),
        render_mode="human",
    )
    
    # Initialize environment
    observations, _ = env.reset()
    agent_names = list(observations.keys())
    
    if not env.agent_roles:
        env.assign_agent_roles()
    
    # Load pretrained agents
    agents = load_pretrained_agents(config, env, agent_names, device)
    
    # Run visualization episodes
    for episode in range(config.visualization.episodes):
        print(f"\n📺 Episode {episode + 1}/{config.visualization.episodes}")
        
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_rewards = {agent: 0 for agent in agent_names}
        
        while not done and step_count < config.visualization.max_episode_steps:
            actions = {}
            
            # Select actions from agents
            for agent_name in obs.keys():
                if agent_name in agents:
                    action = agents[agent_name].select_action(obs[agent_name], training=False)
                    # Ensure action is a scalar integer
                    if isinstance(action, (list, np.ndarray)):
                        action = int(action[0]) if len(action) > 0 else 0
                    elif not isinstance(action, (int, np.integer)):
                        action = int(action)
                    actions[agent_name] = action
                else:
                    actions[agent_name] = env.action_space(agent_name).sample()
            
            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Track rewards
            for agent_name, reward in rewards.items():
                episode_rewards[agent_name] += reward
            
            step_count += 1
            
            # Check for episode end events
            episode_outcome = "ongoing"
            for agent_name, info in infos.items():
                if info.get("capture", False):
                    episode_outcome = "capture"
                    if render_enabled:
                        print(f"  🎯 Capture occurred at step {step_count}!")
                elif info.get("collision", False):
                    episode_outcome = "collision"
                    if render_enabled:
                        print(f"  💥 Collision occurred at step {step_count}!")
                elif info.get("out_of_bounds", False):
                    episode_outcome = "out_of_bounds"
                    if render_enabled:
                        print(f"  🚫 Out of bounds at step {step_count}!")
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            obs = next_obs
            
            # Add small delay for better visualization
            if slow_motion and render_enabled and sleep_time > 0:
                time.sleep(sleep_time)
        
        # Episode summary
        if step_count >= config.visualization.max_episode_steps:
            episode_outcome = "timeout"
        
        print(f"  📊 Episode {episode + 1} Results:")
        print(f"     Length: {step_count} steps")
        print(f"     Outcome: {episode_outcome}")
        for agent_name in agent_names:
            role = env.agent_roles[agent_name]
            print(f"     {agent_name} ({role.name}): {episode_rewards[agent_name]:.2f}")
        
        # Wait between episodes if specified
        if pause_between_episodes:
            input("Press Enter to continue to next episode...")
    
    print(f"\n🎉 Visualization complete!")
    env.close()


@hydra.main(version_base=None, config_path="conf", config_name="visualization")
def visualize(config: DictConfig) -> None:
    """Main visualization function"""
    print("🚁 Drone Pursuit-Evasion Visualization")
    print("=" * 50)
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    try:
        run_visualization(config)
    except KeyboardInterrupt:
        print("\n⚠️ Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        raise


if __name__ == "__main__":
    visualize() 