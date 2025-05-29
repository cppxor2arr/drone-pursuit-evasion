import os
import numpy as np
import torch
import random
import datetime
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Tuple

from task.lider_drone_base import LidarDroneBaseEnv, DroneRole, DroneConfig, AgentType
from RL import DroneDQNAgent, ReplayBuffer, DronePPOAgent, RandomAgent, HoveringAgent, DroneSACAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
            "epsilon_start": config.agent.epsilon_start,
            "epsilon_end": config.agent.epsilon_end,
            "epsilon_decay": config.agent.epsilon_decay,
            "batch_size": config.agent.batch_size,
            "network": config.agent.network,
            "buffer_size": config.agent.buffer_size
        })
        return DroneDQNAgent(obs_space, act_space, agent_config, device)
    
    elif agent_type == AgentType.PPO:
        # Use PPO config if available, otherwise create default
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
        # Use SAC config if available, otherwise create default
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


def setup_wandb(config: DictConfig, run_name: str):
    """Initialize Weights & Biases logging"""
    if config.wandb.mode == "disabled":
        return
    
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.wandb.mode,
        tags=config.wandb.tags,
        notes=config.wandb.notes
    )


def evaluate_agents(agents: dict, config: DictConfig, global_step: int, scenario_config) -> dict:
    """Evaluate agents performance"""
    print(f"\nEvaluating at step {global_step}...")
    
    # Create evaluation environment using config objects
    eval_env = LidarDroneBaseEnv(
        env_config=config.environment,
        scenario_config=scenario_config,
        drone_configs=create_drone_configs(scenario_config)
    )
    
    # Initialize
    observations, _ = eval_env.reset()
    agent_names = list(observations.keys())
    
    if not eval_env.agent_roles:
        eval_env.assign_agent_roles()
    
    # Set agents to eval mode
    original_modes = {}
    for agent_name, agent in agents.items():
        original_modes[agent_name] = agent.training if hasattr(agent, 'training') else True
        agent.eval()
    
    # Run evaluation episodes
    total_rewards = {agent: 0 for agent in agent_names}
    total_lengths = 0
    capture_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    timeout_count = 0
    
    # Detailed episode tracking for WandB
    episode_details = []
    
    for episode in range(config.training.eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = {agent: 0 for agent in agent_names}
        step_count = 0
        episode_outcome = "timeout"
        outcome_step = -1
        
        while not done and step_count < 500:
            actions = {}
            
            # Select actions
            for agent_name in obs.keys():
                if agent_name in agents:
                    actions[agent_name] = agents[agent_name].select_action(obs[agent_name], training=False)
                else:
                    actions[agent_name] = eval_env.action_space(agent_name).sample()
            
            next_obs, rewards, terminations, truncations, infos = eval_env.step(actions)
            
            for agent_name, reward in rewards.items():
                episode_reward[agent_name] += reward
            
            step_count += 1
            
            # Check for events
            for agent_name, info in infos.items():
                if info.get("capture", False):
                    episode_outcome = "capture"
                    outcome_step = step_count
                    capture_count += 1
                elif info.get("collision", False):
                    episode_outcome = "collision"
                    outcome_step = step_count
                    collision_count += 1
                elif info.get("out_of_bounds", False):
                    episode_outcome = "out_of_bounds"
                    outcome_step = step_count
                    out_of_bounds_count += 1
            
            done = all(terminations.values()) or all(truncations.values())
            obs = next_obs
        
        # Finalize episode
        if step_count >= 500 and episode_outcome == "timeout":
            timeout_count += 1
            outcome_step = step_count
        
        # Store episode details
        episode_info = {
            'episode_num': episode + 1,
            'length': step_count,
            'outcome': episode_outcome,
            'outcome_step': outcome_step,
            'rewards': episode_reward.copy()
        }
        episode_details.append(episode_info)
        
        # Update totals
        for agent_name in agent_names:
            total_rewards[agent_name] += episode_reward[agent_name]
        total_lengths += step_count
        
        print(f"  Eval Episode {episode + 1}: Length={step_count}, Outcome={episode_outcome}")
    
    # Calculate averages
    avg_rewards = {agent: total_rewards[agent] / config.training.eval_episodes for agent in agent_names}
    avg_length = total_lengths / config.training.eval_episodes
    capture_rate = capture_count / config.training.eval_episodes
    collision_rate = collision_count / config.training.eval_episodes
    out_of_bounds_rate = out_of_bounds_count / config.training.eval_episodes
    timeout_rate = timeout_count / config.training.eval_episodes
    
    # Log results to WandB
    if config.wandb.mode != "disabled":
        # Main metrics
        log_dict = {
            "eval/avg_episode_length": avg_length,
            "eval/capture_rate": capture_rate,
            "eval/collision_rate": collision_rate,
            "eval/out_of_bounds_rate": out_of_bounds_rate,
            "eval/timeout_rate": timeout_rate,
            "eval/total_captures": capture_count,
            "eval/total_collisions": collision_count,
            "eval/total_out_of_bounds": out_of_bounds_count,
            "eval/total_timeouts": timeout_count,
            "global_step": global_step
        }
        
        # Agent rewards
        for agent_name in agent_names:
            role = eval_env.agent_roles[agent_name]
            log_dict[f"eval/{agent_name}_{role.name.lower()}_reward"] = avg_rewards[agent_name]
        
        # Log main metrics
        wandb.log(log_dict)
        
        # Log detailed episode information
        for ep_info in episode_details:
            episode_log = {
                f"eval/episode_{ep_info['episode_num']}_length": ep_info['length'],
                f"eval/episode_{ep_info['episode_num']}_outcome": ep_info['outcome'],
                f"eval/episode_{ep_info['episode_num']}_outcome_step": ep_info['outcome_step'],
                "global_step": global_step
            }
            
            # Add episode rewards for each agent
            for agent_name in agent_names:
                role = eval_env.agent_roles[agent_name]
                episode_log[f"eval/episode_{ep_info['episode_num']}_{agent_name}_{role.name.lower()}_reward"] = ep_info['rewards'][agent_name]
            
            wandb.log(episode_log)
        
        # Create a summary table for this evaluation
        episode_table_data = []
        for ep_info in episode_details:
            row = [
                ep_info['episode_num'],
                ep_info['length'],
                ep_info['outcome'],
                ep_info['outcome_step']
            ]
            # Add rewards for each agent
            for agent_name in agent_names:
                row.append(round(ep_info['rewards'][agent_name], 2))
            
            episode_table_data.append(row)
        
        # Create table headers
        headers = ["Episode", "Length", "Outcome", "Outcome_Step"]
        for agent_name in agent_names:
            role = eval_env.agent_roles[agent_name]
            headers.append(f"{agent_name}_{role.name}_Reward")
        
        # Log evaluation summary table
        eval_table = wandb.Table(data=episode_table_data, columns=headers)
        wandb.log({f"eval/episode_summary_step_{global_step}": eval_table, "global_step": global_step})
    
    print(f"Evaluation results:")
    print(f"  Average episode length: {avg_length:.1f}")
    print(f"  Capture rate: {capture_rate:.2f} ({capture_count}/{config.training.eval_episodes})")
    print(f"  Collision rate: {collision_rate:.2f} ({collision_count}/{config.training.eval_episodes})")
    print(f"  Out of bounds rate: {out_of_bounds_rate:.2f} ({out_of_bounds_count}/{config.training.eval_episodes})")
    print(f"  Timeout rate: {timeout_rate:.2f} ({timeout_count}/{config.training.eval_episodes})")
    for agent_name in agent_names:
        role = eval_env.agent_roles[agent_name]
        print(f"  {agent_name} ({role.name}): {avg_rewards[agent_name]:.2f}")
    
    # Restore training modes
    for agent_name, agent in agents.items():
        if original_modes[agent_name]:
            agent.train()
    
    eval_env.close()
    return avg_rewards


def create_drone_configs(scenario_config):
    """Create DroneConfig objects from scenario configuration"""
    from task.lider_drone_base import DroneRole, DroneConfig, AgentType
    import numpy as np
    
    configs = []
    for drone in scenario_config.drones:
        role = getattr(DroneRole, drone.role)
        agent_type = AgentType(drone.agent_type)  # Convert string to enum
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


def setup_training(config: DictConfig) -> Tuple[str, str, str, Path]:
    """Setup training environment and configuration"""
    # Set up experiment
    set_seed(config.seed)
    device = setup_device(config.device)
    
    # Create timestamped run name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{timestamp}"
    
    # Setup logging
    setup_wandb(config, run_name)
    
    print(f"Starting training: {run_name}")
    print(f"Device: {device}")
    print(f"Config:\n{OmegaConf.to_yaml(config)}")
    
    # Create weights directory with better organization
    # Use absolute path to avoid Hydra's working directory change
    original_cwd = hydra.utils.get_original_cwd()
    weights_dir = Path(original_cwd) / config.paths.weights_dir / config.experiment_name / timestamp
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create a "latest" symlink for easy access
    latest_dir = Path(original_cwd) / config.paths.weights_dir / config.experiment_name / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        latest_dir.unlink()
    latest_dir.symlink_to(timestamp, target_is_directory=True)
    
    print(f"Weights will be saved to: {weights_dir}")
    print(f"Latest weights available at: {latest_dir}")
    
    return device, run_name, timestamp, weights_dir


def create_training_environment(config: DictConfig, drone_configs: list):
    """Create and initialize the training environment"""
    # Create environment using config objects
    env = LidarDroneBaseEnv(
        env_config=config.environment,
        scenario_config=config.scenario,
        drone_configs=drone_configs
    )
    
    # Initialize environment
    observations, _ = env.reset()
    agent_names = list(observations.keys())
    
    # If agent_roles is empty, manually call assign_agent_roles
    if not env.agent_roles:
        env.assign_agent_roles()
    
    return env, observations, agent_names


def create_training_agents(config: DictConfig, drone_configs: list, env, agent_names: list, device: str, weights_dir: Path) -> dict:
    """Create agents for ALL drones (including non-trainable ones)"""
    agents = {}
    agent_roles = {}
    
    print("\nCreating agents...")
    for i, agent_name in enumerate(agent_names):
        role = env.agent_roles[agent_name]
        agent_roles[agent_name] = role
        
        print(f"Agent {agent_name}: {role.name}")
        
        # Get drone config by index (environment assigns agents by order)
        drone_config = drone_configs[i] if i < len(drone_configs) else None
        if drone_config is None:
            print(f"  ⚠️ No configuration found for agent {agent_name}")
            continue
            
        # Get spaces
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        # Create agent based on drone config
        agent = create_agent(drone_config.agent_type, config, obs_space, act_space, device)
        agents[agent_name] = agent
        
        # Save initial model
        agent.save(weights_dir / f"{role.name.lower()}_0.pt")
        
        if drone_config.is_training:
            print(f"  ✅ Created {drone_config.agent_type.value.upper()} agent (TRAINABLE)")
        else:
            print(f"  ℹ️ Created {drone_config.agent_type.value.upper()} agent (NON-TRAINABLE)")
    
    return agents, agent_roles


def run_training_loop(config: DictConfig, env, agents: dict, agent_names: list, drone_configs: list, weights_dir: Path):
    """Execute the main training loop"""
    print("\nStarting training...")
    obs, _ = env.reset()
    episode_rewards = {agent: 0 for agent in agent_names}
    episode_count = 0
    
    for global_step in range(config.training.total_timesteps):
        actions = {}
        
        # Select actions for ALL agents using unified interface
        for agent_name in obs.keys():
            if agent_name in agents:
                actions[agent_name] = agents[agent_name].select_action(obs[agent_name], training=True)
            else:
                # Fallback for missing agents
                actions[agent_name] = env.action_space(agent_name).sample()
        
        # Environment step
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Process experiences for ALL agents
        for agent_name in agent_names:
            if agent_name in obs and agent_name in next_obs and agent_name in agents:
                episode_rewards[agent_name] += rewards[agent_name]
                
                # Create transition
                done = terminations[agent_name] or truncations[agent_name]
                transition = (
                    obs[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_obs[agent_name],
                    done
                )
                
                # Process transition using unified interface
                metrics = agents[agent_name].process(transition)
                
                # Log metrics for trainable agents
                if global_step % config.training.log_interval == 0 and config.wandb.mode != "disabled":
                    # Check if this is a trainable agent (has meaningful loss)
                    if metrics.get("loss", 0) > 0 or hasattr(agents[agent_name], 'replay_buffer'):
                        for metric_name, metric_value in metrics.items():
                            wandb.log({
                                f"train/{agent_name}_{metric_name}": metric_value,
                                "global_step": global_step
                            })
            elif agent_name in episode_rewards:
                # Still track episode rewards for missing agents
                episode_rewards[agent_name] += rewards.get(agent_name, 0)
        
        # Save models periodically (only for trainable agents)
        if global_step > 0 and global_step % config.training.save_interval == 0:
            save_periodic_models(agents, env, drone_configs, weights_dir, global_step)
        
        # Evaluate agents
        if global_step > 0 and global_step % config.training.evaluate_interval == 0:
            evaluate_agents(agents, config, global_step, config.scenario)
        
        # Check episode done
        done = all(terminations.values()) or all(truncations.values())
        
        if done:
            # Log episode
            log_episode_results(config, episode_rewards, env, episode_count, global_step)
            
            # Print episode results
            print(f"Episode {episode_count}, Step {global_step}:")
            for agent_name in episode_rewards:
                role = env.agent_roles[agent_name]
                print(f"  {agent_name} ({role.name}): {episode_rewards[agent_name]:.2f}")
            
            # Reset
            episode_rewards = {agent: 0 for agent in agent_names}
            episode_count += 1
            obs, _ = env.reset()
        else:
            obs = next_obs


def save_periodic_models(agents: dict, env, drone_configs: list, weights_dir: Path, global_step: int):
    """Save models periodically during training"""
    for i, agent_name in enumerate(agents.keys()):
        role = env.agent_roles[agent_name]
        drone_config = drone_configs[i] if i < len(drone_configs) else None
        
        if drone_config and drone_config.is_training:
            agents[agent_name].save(weights_dir / f"{role.name.lower()}_{global_step}.pt")
    print(f"Saved models at step {global_step}")


def save_final_models(agents: dict, env, drone_configs: list, weights_dir: Path):
    """Save final models after training completion"""
    print("\nSaving final models...")
    for i, agent_name in enumerate(agents.keys()):
        role = env.agent_roles[agent_name]
        drone_config = drone_configs[i] if i < len(drone_configs) else None
        
        if drone_config and drone_config.is_training:
            agents[agent_name].save(weights_dir / f"{role.name.lower()}_final.pt")


def log_episode_results(config: DictConfig, episode_rewards: dict, env, episode_count: int, global_step: int):
    """Log episode results to WandB"""
    if config.wandb.mode != "disabled":
        for agent_name in episode_rewards:
            role = env.agent_roles[agent_name]
            wandb.log({
                f"train/{role.name.lower()}_episode_reward": episode_rewards[agent_name],
                "train/episode": episode_count,
                "global_step": global_step
            })


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    
    # Setup training
    device, run_name, timestamp, weights_dir = setup_training(config)
    
    # Load drone configurations from scenario
    drone_configs = create_drone_configs(config.scenario)
    
    # Create environment
    env, observations, agent_names = create_training_environment(config, drone_configs)
    
    # Create agents
    agents, agent_roles = create_training_agents(config, drone_configs, env, agent_names, device, weights_dir)
    
    # Run training loop
    run_training_loop(config, env, agents, agent_names, drone_configs, weights_dir)
    
    # Save final models
    save_final_models(agents, env, drone_configs, weights_dir)
    
    print("Training complete!")
    
    if config.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    train() 