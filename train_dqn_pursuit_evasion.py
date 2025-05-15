import os
import numpy as np
import torch
import random
import datetime
import wandb
from task.lider_drone_base import LidarDroneBaseEnv, DroneRole, DroneConfig
from RL.dqn_agent import DroneDQNAgent

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Hyperparameters
TOTAL_TIMESTEPS = 1000000
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
BUFFER_SIZE = 10000
GAMMA = 0.99  # Discount factor
TAU = 0.005   # For soft update of target network
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 700000
TARGET_UPDATE_INTERVAL = 1000
SAVE_INTERVAL = 5000
EVALUATE_INTERVAL = 2000

def train_agents(drone_configs):
    """
    Train DQN agents for pursuit-evasion based on drone configurations
    
    Args:
        drone_configs (list): List of DroneConfig objects
    """
    # Get current time for wandb run name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which drones are being trained
    training_roles = [config.role for config in drone_configs if config.is_training]
    train_mode = "_".join([role.name.lower() for role in training_roles]) or "none"
    
    model_name = f"DQN_PE_{current_time}_{train_mode}"
    
    # Create weights directory for this run
    weights_dir = f"weights/pursuit_evade/{current_time}"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Track best model performance for saving
    best_models = {}  # Dictionary to track best models by agent name and role
    
    # Starting step for training if resuming
    start_step = 0
    resume_paths = {
        DroneRole.PURSUER: None,
        DroneRole.EVADER: None
    }
    
    # Set starting epsilon based on resume paths
    local_epsilon_start = EPSILON_START
    local_epsilon_end = EPSILON_END
    local_epsilon_decay = EPSILON_DECAY
    
    # Check if we're resuming from previous training
    for config in drone_configs:
        if config.resume_from:
            resume_paths[config.role] = config.resume_from
            print(f"Will resume {config.role.name} training from {config.resume_from}")
    
    # Initialize wandb with metadata in run name
    drone_config_info = []
    for i, config in enumerate(drone_configs):
        drone_config_info.append({
            f"drone_{i}_role": config.role.name,
            f"drone_{i}_training": config.is_training,
            f"drone_{i}_action_length": config.action_length,
            f"drone_{i}_resume_from": config.resume_from
        })
    
    wandb.init(
        project="drone-pursuit-evasion",
        name=model_name,
        config={
            "model_type": "DQN",
            "task": "Pursuit-Evasion",
            "learning_rate": LEARNING_RATE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "gamma": GAMMA,
            "tau": TAU,
            "epsilon_start": local_epsilon_start,
            "epsilon_end": local_epsilon_end,
            "epsilon_decay": local_epsilon_decay,
            "date": current_time,
            "train_mode": train_mode,
            "drone_configs": drone_config_info
        }
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment with drone configurations - no rendering during training
    env = LidarDroneBaseEnv(
        lidar_reach=4.0,
        num_ray=20,
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=False  # Disable rendering during training
    )
    
    # Reset to initialize agent list
    observations, _ = env.reset()
    
    # Get agent names from observations
    agent_names = list(observations.keys())
    
    # Map agent names to their roles
    agent_roles = {}
    for agent_name in agent_names:
        role = env.agent_roles[agent_name]
        agent_roles[agent_name] = role
    
    # Print agent role assignments
    for agent_name, role in agent_roles.items():
        print(f"Agent {agent_name} assigned role: {role.name}")
    
    # Create dictionary of agents by agent name
    agents = {}
    buffers = {}
    
    # Initialize agents for each role that exists in the environment
    for agent_name, role in agent_roles.items():
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        # Get the corresponding drone config
        drone_config = next((config for config in drone_configs if config.role == role), None)
        
        if drone_config:
            # Skip agent creation for HOVER and RANDOM roles
            if role in [DroneRole.HOVER, DroneRole.RANDOM]:
                continue
                
            # Create agent with appropriate parameters
            agent = DroneDQNAgent(
                observation_space=obs_space, 
                action_space=act_space, 
                device=device,
                gamma=GAMMA,
                epsilon_start=local_epsilon_start,
                epsilon_end=local_epsilon_end,
                epsilon_decay=local_epsilon_decay,
                learning_rate=LEARNING_RATE
            )
            
            # Load model if resuming
            if drone_config.resume_from and os.path.exists(drone_config.resume_from):
                print(f"Loading {role.name} model from {drone_config.resume_from} for agent {agent_name}")
                agent.load(drone_config.resume_from)
            
            # Set to train mode if this drone is being trained
            if drone_config.is_training:
                agent.train()
                print(f"Agent {agent_name} ({role.name}) set to training mode")
            else:
                agent.eval()
                print(f"Agent {agent_name} ({role.name}) set to evaluation mode")
            
            # Create replay buffer
            if drone_config.is_training:
                buffer = agent.create_buffer(BUFFER_SIZE)
                
                # Store agent and buffer by agent name
                agents[agent_name] = agent
                buffers[agent_name] = buffer
                
                # Save initial model if not resuming
                if not drone_config.resume_from:
                    agent.save(path=f"{weights_dir}/{role.name.lower()}", step=0)
                    print(f"Saved initial model for agent {agent_name} ({role.name}) at step 0")
    
    # Training loop
    obs = observations.copy()
    episode_rewards = {agent: 0 for agent in agent_names}
    episode_lengths = {agent: 0 for agent in agent_names}
    episode_count = 0
    death_reasons = {'collision': 0, 'out_of_bounds': 0, 'capture': 0, 'max_steps': 0}
    
    print("Starting training...")
    for global_step in range(start_step, TOTAL_TIMESTEPS):
        actions = {}
        
        # Select actions for each agent based on its role
        for agent_name in list(obs.keys()):
            agent_obs = obs[agent_name]
            agent_role = env.agent_roles[agent_name]
            
            # Handle different roles
            if agent_role == DroneRole.RANDOM:
                # Random action
                actions[agent_name] = env.action_space(agent_name).sample()
            elif agent_role == DroneRole.HOVER:
                # No action needed for hovering (handled in environment)
                actions[agent_name] = 0  # Default action
            elif agent_name in agents:
                agent = agents[agent_name]
                drone_config = next((config for config in drone_configs if config.role == agent_role), None)
                
                if drone_config and drone_config.is_training:
                    # Training mode: use epsilon-greedy policy
                    actions[agent_name] = agent.select_action(agent_obs)
                else:
                    # Evaluation mode: use greedy policy
                    actions[agent_name] = int(agent.exploit(agent_obs)[0])
            else:
                # Default fallback
                actions[agent_name] = env.action_space(agent_name).sample()
        
        # Perform action in environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check why the episode ended (for logging)
        for agent_name, info in infos.items():
            if info.get("collision", False):
                death_reasons['collision'] += 1
                print(f"Agent {agent_name} died from collision at step {global_step}")
            elif info.get("out_of_bounds", False):
                death_reasons['out_of_bounds'] += 1
                print(f"Agent {agent_name} died from going out of bounds at step {global_step}")
            elif info.get("capture", False):
                death_reasons['capture'] += 1
                print(f"Pursuer captured evader at step {global_step}")
        
        # Check if episode reached max steps
        if any(truncations.values()):
            death_reasons['max_steps'] += 1
            print(f"Episode reached maximum steps at step {global_step}")
        
        # Update replay buffers for training agents
        if len(next_obs) == len(obs):  # Make sure we have observations for all agents
            for agent_name in agent_names:
                if agent_name in obs and agent_name in next_obs:
                    # Get agent role
                    agent_role = env.agent_roles[agent_name]
                    
                    # Track episode stats
                    episode_rewards[agent_name] += rewards[agent_name]
                    episode_lengths[agent_name] += 1
                    
                    # Only update buffer if this role is being trained
                    if agent_name in agents and agent_name in buffers:
                        drone_config = next((config for config in drone_configs if config.role == agent_role), None)
                        
                        if drone_config and drone_config.is_training:
                            done = terminations[agent_name] or truncations[agent_name]
                            
                            # Add experience to buffer
                            buffers[agent_name].add(
                                obs[agent_name],
                                actions[agent_name],
                                rewards[agent_name],
                                next_obs[agent_name],
                                done
                            )
        
        # Train agents if enough samples are collected
        if global_step > BATCH_SIZE:
            for agent_name, agent in agents.items():
                if agent_name in buffers:
                    buffer = buffers[agent_name]
                    
                    # Check if this role is being trained
                    drone_config = next((config for config in drone_configs if config.role == env.agent_roles[agent_name]), None)
                    
                    if drone_config and drone_config.is_training and len(buffer) > BATCH_SIZE:
                        # Sample batch and update network
                        batch = buffer.sample(BATCH_SIZE)
                        loss = agent.update(batch)
                        
                        # Update exploration rate
                        agent.update_epsilon_cosine()
                        
                        # Log training metrics (prefixed with 'train/')
                        wandb.log({
                            f"train/{agent_name}_loss": loss,
                            f"train/{agent_name}_epsilon": agent.get_epsilon(),
                            "global_step": global_step
                        })
        
        # Periodically update target networks for training agents
        if global_step % TARGET_UPDATE_INTERVAL == 0:
            for agent_name, agent in agents.items():
                drone_config = next((config for config in drone_configs if config.role == env.agent_roles[agent_name]), None)
                
                if drone_config and drone_config.is_training:
                    agent.update_target_network()
            print(f"Updated target networks at step {global_step}")
        
        # Save models periodically
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            for agent_name, agent in agents.items():
                drone_config = next((config for config in drone_configs if config.role == env.agent_roles[agent_name]), None)
                
                if drone_config and drone_config.is_training:
                    agent.save(path=f"{weights_dir}/{env.agent_roles[agent_name].name.lower()}", step=global_step)
            print(f"Saved checkpoint models at step {global_step}")
        
        # Check if episode is done
        done = all(terminations.values()) or all(truncations.values())
        
        if done:
            # Log episode stats with prefix 'train/'
            for agent_name in episode_rewards:
                agent_role = env.agent_roles[agent_name]
                
                wandb.log({
                    f"train/{agent_role.name.lower()}_episode_reward": episode_rewards[agent_name],
                    f"train/{agent_role.name.lower()}_episode_length": episode_lengths[agent_name],
                    "train/episode": episode_count,
                    "train/captures": death_reasons['capture'],
                    "train/collisions": death_reasons['collision'],
                    "train/out_of_bounds": death_reasons['out_of_bounds'],
                    "train/max_steps": death_reasons['max_steps'],
                })
                print(f"Episode {episode_count}, {agent_role.name}: Reward={episode_rewards[agent_name]:.2f}, Length={episode_lengths[agent_name]}")
            
            # Reset episode tracking
            episode_rewards = {agent: 0 for agent in agent_names}
            episode_lengths = {agent: 0 for agent in agent_names}
            episode_count += 1
            
            # Reset environment
            obs, _ = env.reset()
        else:
            # Update observation
            obs = next_obs
            
        # Evaluate periodically
        if global_step % EVALUATE_INTERVAL == 0:
            eval_results = evaluate_agents(agents, agent_roles, global_step, drone_configs)
            
            # Check if this is the best model for each agent
            for agent_name, agent in agents.items():
                if agent_name in eval_results['avg_rewards']:
                    # Get current role of this agent
                    role = env.agent_roles[agent_name]
                    
                    # Initialize best model tracking for this agent if not exists
                    if agent_name not in best_models:
                        best_models[agent_name] = {'step': 0, 'performance': float('-inf'), 'role': role}
                    
                    # Check if this is a better performance
                    if eval_results['avg_rewards'][agent_name] > best_models[agent_name]['performance']:
                        best_models[agent_name]['performance'] = eval_results['avg_rewards'][agent_name]
                        best_models[agent_name]['step'] = global_step
                        
                        # Save best model
                        agent.save(path=f"{weights_dir}/{role.name.lower()}", is_best=True)
                        print(f"New best model for agent {agent_name} ({role.name}) at step {global_step} with reward {eval_results['avg_rewards'][agent_name]:.2f}")
    
    # Save final models
    for agent_name, agent in agents.items():
        agent.save(path=f"{weights_dir}/{env.agent_roles[agent_name].name.lower()}")
    
    # Save information about best models
    with open(f"{weights_dir}/best_models.txt", "w") as f:
        for agent_name, data in best_models.items():
            role_name = data['role'].name if 'role' in data else 'unknown'
            f.write(f"Agent {agent_name} ({role_name}): step {data['step']} with performance {data['performance']:.2f}\n")
    
    print("Training complete!")
    for agent_name, data in best_models.items():
        role_name = data['role'].name if 'role' in data else 'unknown'
        print(f"Best {agent_name} ({role_name}) model: step {data['step']} with reward {data['performance']:.2f}")
    print(f"Death reasons: {death_reasons}")
    
    # Finalize wandb
    wandb.finish()

def evaluate_agents(agents, agent_roles, global_step, drone_configs, num_episodes=3):
    """Evaluate the performance of trained agents"""
    print(f"Evaluating agents at step {global_step}...")
    
    # Create a temporary environment with rendering for evaluation
    eval_env = LidarDroneBaseEnv(
        lidar_reach=4.0,
        num_ray=20,
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=True  # Enable visualization during evaluation
    )
    
    # Reset the environment
    eval_obs, _ = eval_env.reset()
    
    # Keep track of original training modes and switch to eval mode
    original_modes = {}
    for agent_name, agent in agents.items():
        original_modes[agent_name] = agent.q_net.training
        agent.eval()
    
    # Get agent names from evaluation environment
    eval_agent_names = list(eval_obs.keys())
    
    # Map evaluation environment agent names to their roles
    eval_agent_roles = {}
    for agent_name in eval_agent_names:
        eval_agent_roles[agent_name] = eval_env.agent_roles[agent_name]
    
    # Prepare tracking variables
    avg_rewards = {agent: 0 for agent in eval_agent_names}
    avg_lengths = 0
    capture_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_rewards = {agent: 0 for agent in eval_agent_names}
        episode_length = 0
        
        while not done and episode_length < 500:  # Add max episode length
            actions = {}
            
            # Select actions for each agent based on its role
            for agent_name in list(obs.keys()):
                if agent_name not in obs:
                    continue  # Skip agents without observations
                    
                agent_obs = obs[agent_name]
                agent_role = eval_env.agent_roles[agent_name]
                
                # Handle different roles
                if agent_role == DroneRole.RANDOM:
                    # Random action
                    actions[agent_name] = eval_env.action_space(agent_name).sample()
                elif agent_role == DroneRole.HOVER:
                    # No action needed for hovering (handled in environment)
                    actions[agent_name] = 0  # Default action
                elif agent_name in agents:
                    # Use agent's policy if we have a matching agent
                    agent = agents[agent_name]
                    actions[agent_name] = int(agent.exploit(agent_obs)[0])
                else:
                    # Try to find a matching agent by role
                    matching_agents = [name for name, role in agent_roles.items() 
                                    if role == agent_role and name in agents]
                    if matching_agents:
                        # Use the first agent with the matching role
                        agent = agents[matching_agents[0]]
                        actions[agent_name] = int(agent.exploit(agent_obs)[0])
                    else:
                        # Default to random action
                        actions[agent_name] = eval_env.action_space(agent_name).sample()
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = eval_env.step(actions)
            
            # Update rewards
            for agent_name, reward in rewards.items():
                episode_rewards[agent_name] += reward
            
            episode_length += 1
            
            # Track termination reasons
            for agent_name, info in infos.items():
                if info.get("capture", False):
                    capture_count += 1
                if info.get("collision", False):
                    collision_count += 1
                if info.get("out_of_bounds", False):
                    out_of_bounds_count += 1
                
            # Check for episode termination
            done = all(terminations.values()) or all(truncations.values())
            
            # Update observation
            obs = next_obs
        
        # Update average metrics
        for agent_name in eval_agent_names:
            avg_rewards[agent_name] += episode_rewards[agent_name] / num_episodes
        avg_lengths += episode_length / num_episodes
        
        print(f"Eval Episode {episode+1}/{num_episodes}: ", end="")
        for agent_name in eval_agent_names:
            role = eval_env.agent_roles[agent_name]
            print(f"{agent_name} ({role.name}) Reward={episode_rewards[agent_name]:.2f}, ", end="")
        print(f"Length={episode_length}")
    
    # Log evaluation results with prefix 'eval/'
    for agent_name in eval_agent_names:
        role = eval_env.agent_roles[agent_name]
        role_name = role.name.lower()
        wandb.log({
            f"eval/{agent_name}_{role_name}_reward": avg_rewards[agent_name],
            "eval/episode_length": avg_lengths,
            "eval/capture_rate": capture_count / num_episodes,
            "eval/collision_rate": collision_count / num_episodes,
            "eval/out_of_bounds_rate": out_of_bounds_count / num_episodes,
            "global_step": global_step
        })
    
    print(f"Evaluation complete. Avg Length: {avg_lengths:.2f}, Capture Rate: {capture_count / num_episodes:.2f}")
    for agent_name in eval_agent_names:
        role = eval_env.agent_roles[agent_name]
        print(f"Avg {agent_name} ({role.name}) Reward: {avg_rewards[agent_name]:.2f}")
    
    # Restore original training modes
    for agent_name, agent in agents.items():
        if original_modes[agent_name]:
            agent.train()
    
    return {
        'avg_rewards': avg_rewards,
        'avg_length': avg_lengths,
        'capture_rate': capture_count / num_episodes
    }

if __name__ == "__main__":
    # Define drone configurations
    drone_configs = [
        DroneConfig(
            role=DroneRole.PURSUER,
            start_pos=np.array([1, 1, 1]),
            start_orn=np.array([0, 0, 0]),
            action_length=7.0,
            is_training=True,  # Train the pursuer
            resume_from=None   # Start from scratch
        ),
        DroneConfig(
            role=DroneRole.EVADER,
            start_pos=np.array([-1, -1, 1]),
            start_orn=np.array([0, 0, 0]),
            action_length=7.0,
            is_training=True,  # Train the evader
            resume_from=None   # Start from scratch
        )
    ]
    
    # Example usage for different training scenarios:
    
    # # Train only pursuer against random evader
    # drone_configs = [
    #     DroneConfig(
    #         role=DroneRole.PURSUER,
    #         start_pos=np.array([1, 1, 1]),
    #         start_orn=np.array([0, 0, 0]),
    #         action_length=7.0,
    #         is_training=True
    #     ),
    #     DroneConfig(
    #         role=DroneRole.RANDOM,
    #         start_pos=np.array([-1, -1, 1]),
    #         start_orn=np.array([0, 0, 0]),
    #         action_length=5.0
    #     )
    # ]
    
    # # Train only evader against hovering pursuer
    # drone_configs = [
    #     DroneConfig(
    #         role=DroneRole.HOVER,
    #         start_pos=np.array([1, 1, 1]),
    #         start_orn=np.array([0, 0, 0])
    #     ),
    #     DroneConfig(
    #         role=DroneRole.EVADER,
    #         start_pos=np.array([-1, -1, 1]),
    #         start_orn=np.array([0, 0, 0]),
    #         action_length=7.0,
    #         is_training=True
    #     )
    # ]
    
    # # Resume training from saved models
    # drone_configs = [
    #     DroneConfig(
    #         role=DroneRole.PURSUER,
    #         start_pos=np.array([1, 1, 1]),
    #         start_orn=np.array([0, 0, 0]),
    #         action_length=7.0,
    #         is_training=True,
    #         resume_from="weights/pursuit_evade/20230615_123456/pursuer_best.pt"
    #     ),
    #     DroneConfig(
    #         role=DroneRole.EVADER,
    #         start_pos=np.array([-1, -1, 1]),
    #         start_orn=np.array([0, 0, 0]),
    #         action_length=7.0,
    #         is_training=True,
    #         resume_from="weights/pursuit_evade/20230615_123456/evader_best.pt"
    #     )
    # ]
    
    train_agents(drone_configs) 