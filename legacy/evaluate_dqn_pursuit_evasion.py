import os
import numpy as np
import torch
import argparse
import time
import datetime
from task.lider_drone_base import LidarDroneBaseEnv, DroneRole, DroneConfig
from RL.dqn_agent import DroneDQNAgent, QNetwork

def load_models(model_paths, device='cpu'):
    """
    Load trained models from saved weights
    
    Args:
        model_paths (dict): Dictionary mapping DroneRoles to model paths
        device (str): Device to load models on ('cpu' or 'cuda')
    """
    # Check if model paths exist
    agents = {}
    drone_configs = []
    
    for role, path in model_paths.items():
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{role.name} model not found at: {path}")
    
    # Create basic drone configs based on roles
    for i, (role, path) in enumerate(model_paths.items()):
        # Skip roles without models
        if path is None and role not in [DroneRole.HOVER, DroneRole.RANDOM]:
            continue
            
        # Create a position based on role
        if role == DroneRole.PURSUER:
            pos = np.array([1, 1, 1])
        elif role == DroneRole.EVADER:
            pos = np.array([-1, -1, 1])
        else:
            pos = np.array([0, 0, 1])
            
        # Add to drone configurations
        drone_configs.append(
            DroneConfig(
                role=role,
                start_pos=pos,
                start_orn=np.array([0, 0, 0]),
                action_length=7.0  # Default action length
            )
        )
    
    # Create environment to get observation and action spaces
    env = LidarDroneBaseEnv(
        lidar_reach=4.0, 
        num_ray=20, 
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=False
    )
    
    # Reset environment to initialize agent list
    observations, _ = env.reset()
    
    # Get agent names and map to their roles
    agent_names = list(observations.keys())
    agent_roles = {}
    
    for agent_name in agent_names:
        role = env.agent_roles[agent_name]
        agent_roles[agent_name] = role
        print(f"Agent {agent_name} has role: {role.name}")
    
    # Create agents for each drone that needs one
    for agent_name, role in agent_roles.items():
        # Skip roles that don't need agents
        if role in [DroneRole.HOVER, DroneRole.RANDOM]:
            continue
            
        # Check if we have a model for this role
        if role in model_paths and model_paths[role] is not None:
            obs_space = env.observation_space(agent_name)
            act_space = env.action_space(agent_name)
            
            # Create and load agent
            agent = DroneDQNAgent(obs_space, act_space, device=device)
            print(f"Loading {role.name} model from {model_paths[role]} for agent {agent_name}")
            agent.load(model_paths[role])
            agent.eval()  # Set to evaluation mode
            
            # Store agent by its name (unique ID)
            agents[agent_name] = agent
    
    return agents, agent_roles, drone_configs

def find_latest_models(base_dir="weights/pursuit_evade"):
    """Find the latest trained models in the weights directory"""
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} not found.")
        return {}
    
    # Check for models in timestamp subdirectories first
    timestamps = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)) and d not in ["pursuer", "evador"]:  # Skip special directories
            # Make sure directory contains model files
            if any(f.endswith(".pt") for f in os.listdir(os.path.join(base_dir, d))):
                timestamps.append(d)
    
    # If we found timestamp directories with models
    if timestamps:
        # Sort by timestamp (newest first)
        timestamps.sort(reverse=True)
        latest_dir = os.path.join(base_dir, timestamps[0])
        print(f"Found latest model directory: {latest_dir}")
        
        model_paths = {}
        
        for role in [DroneRole.PURSUER, DroneRole.EVADER]:
            role_name = role.name.lower()
            
            # Find models for this role
            role_models = [f for f in os.listdir(latest_dir) 
                          if f.startswith(f"{role_name}_") and f.endswith(".pt")]
            
            if not role_models:
                print(f"No {role.name} models found in {latest_dir}")
                continue
                
            # Prioritize models: best > final > highest step count
            if f"{role_name}_best.pt" in role_models:
                model_paths[role] = os.path.join(latest_dir, f"{role_name}_best.pt")
            elif f"{role_name}_final.pt" in role_models:
                model_paths[role] = os.path.join(latest_dir, f"{role_name}_final.pt")
            else:
                # Get highest step count
                step_models = [m for m in role_models 
                              if m != f"{role_name}_0.pt" 
                              and not m.endswith("best.pt") 
                              and not m.endswith("final.pt")]
                
                if step_models:
                    # Sort by step number
                    def get_step_number(filename):
                        try:
                            if '_' in filename and '.' in filename:
                                return int(filename.split('_')[1].split('.')[0])
                            return 0
                        except (IndexError, ValueError):
                            return 0
                            
                    step_models.sort(key=get_step_number, reverse=True)
                    model_paths[role] = os.path.join(latest_dir, step_models[0])
                elif f"{role_name}_0.pt" in role_models:
                    model_paths[role] = os.path.join(latest_dir, f"{role_name}_0.pt")
        
        if model_paths:
            # Print found models
            for role, path in model_paths.items():
                print(f"Selected {role.name} model: {path}")
            return model_paths
    
    # If no models found in timestamp directories, check root directory
    print("No models found in timestamp directories, checking root directory...")
    model_paths = {}
    
    for role in [DroneRole.PURSUER, DroneRole.EVADER]:
        role_name = role.name.lower()
        role_models = [f for f in os.listdir(base_dir) 
                      if f.startswith(f"{role_name}_") and f.endswith(".pt")]
        
        if role_models:
            model_paths[role] = os.path.join(base_dir, role_models[0])
            print(f"Selected {role.name} model: {model_paths[role]}")
    
    if not model_paths:
        print(f"No suitable models found in {base_dir}")
    
    return model_paths

def evaluate(agents, agent_roles, drone_configs, num_episodes=10, render=True, sleep_time=0.01):
    """
    Evaluate trained agents
    
    Args:
        agents (dict): Dictionary mapping agent names to agent instances
        agent_roles (dict): Dictionary mapping agent names to their roles
        drone_configs (list): List of DroneConfig objects
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
        sleep_time (float): Time to sleep between steps
    """
    # Create environment with drone configurations
    env = LidarDroneBaseEnv(
        lidar_reach=4.0, 
        num_ray=20, 
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=render,  # Enable visualization
        render_mode="human" if render else None  # Set render mode
    )
    
    print(f"Simulation rendering is {'enabled' if render else 'disabled'}")
    if render:
        print(f"Sleep between steps: {sleep_time} seconds (use --sleep 0 for fastest evaluation)")
    
    # Reset to initialize agent list
    obs, _ = env.reset()
    
    # Get agent names
    agent_names = list(obs.keys())
    
    # Track statistics
    episode_rewards = {agent: 0 for agent in agent_names}
    episode_lengths = []
    capture_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = {agent: 0 for agent in agent_names}
        step_count = 0
        
        while not done:
            if render and sleep_time > 0:
                time.sleep(sleep_time)  # Slow down visualization
                
            actions = {}
            
            # Select actions for each agent based on its role
            for agent_name in list(obs.keys()):
                if agent_name not in obs:
                    continue  # Skip agents without observations
                    
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
                    # Use agent's policy
                    agent = agents[agent_name]
                    actions[agent_name] = int(agent.exploit(agent_obs)[0])
                else:
                    # Default fallback
                    actions[agent_name] = env.action_space(agent_name).sample()
                    print(f"Warning: No agent for {agent_name} with role {agent_role.name}, using random action")
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update statistics
            for agent_name, reward in rewards.items():
                episode_reward[agent_name] += reward
            
            step_count += 1
            
            # Check for termination reasons
            for agent_name, info in infos.items():
                if info.get("capture", False):
                    capture_count += 1
                    if render:
                        print(f"Capture occurred at step {step_count}!")
                if info.get("collision", False):
                    collision_count += 1
                    if render:
                        print(f"Collision occurred at step {step_count}!")
                if info.get("out_of_bounds", False):
                    out_of_bounds_count += 1
                    if render:
                        print(f"Out of bounds at step {step_count}!")
            
            # Check for episode termination
            done = all(terminations.values()) or all(truncations.values())
            
            # Update observation
            obs = next_obs
        
        # Store episode statistics
        for agent_name in agent_names:
            episode_rewards[agent_name] += episode_reward[agent_name]
        episode_lengths.append(step_count)
        
        # Print episode results
        print(f"Episode {episode+1}/{num_episodes}: Length={step_count}")
        for agent_name in agent_names:
            role = env.agent_roles[agent_name]
            print(f"  {agent_name} ({role.name}) Reward: {episode_reward[agent_name]:.2f}")
    
    # Compute average statistics
    avg_rewards = {agent: episode_rewards[agent] / num_episodes for agent in agent_names}
    avg_length = sum(episode_lengths) / num_episodes
    capture_rate = capture_count / num_episodes
    collision_rate = collision_count / num_episodes
    out_of_bounds_rate = out_of_bounds_count / num_episodes
    
    print("\nEvaluation Results:")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Capture Rate: {capture_rate:.2f} ({capture_count}/{num_episodes})")
    print(f"Collision Rate: {collision_rate:.2f} ({collision_count}/{num_episodes})")
    print(f"Out of Bounds Rate: {out_of_bounds_rate:.2f} ({out_of_bounds_count}/{num_episodes})")
    
    for agent_name in agent_names:
        role = env.agent_roles[agent_name]
        print(f"Average {agent_name} ({role.name}) Reward: {avg_rewards[agent_name]:.2f}")
    
    return {
        "avg_rewards": avg_rewards,
        "avg_length": avg_length,
        "capture_rate": capture_rate,
        "collision_rate": collision_rate,
        "out_of_bounds_rate": out_of_bounds_rate
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agents for drone pursuit-evasion")
    parser.add_argument("--pursuer", type=str, default=None,
                        help="Path to the pursuer model weights")
    parser.add_argument("--evader", type=str, default=None,
                        help="Path to the evader model weights")
    parser.add_argument("--pursuer-role", type=str, default='pursuer',
                        help="Role for the first drone (pursuer, evader, hover, random)")
    parser.add_argument("--evader-role", type=str, default='evader',
                        help="Role for the second drone (pursuer, evader, hover, random)")
    parser.add_argument("--pursuer-action-length", type=float, default=7.0,
                        help="Action length for the pursuer")
    parser.add_argument("--evader-action-length", type=float, default=7.0,
                        help="Action length for the evader")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (headless mode)")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Sleep time between steps for visualization (set to 0 for fastest evaluation)")
    parser.add_argument("--find-latest", action="store_true",
                        help="Automatically find and use the latest saved models")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate a timestamp for this evaluation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Evaluation started at: {timestamp}")
    
    # Parse roles using DroneRole enum directly
    role_map = {
        'pursuer': DroneRole.PURSUER,
        'evader': DroneRole.EVADER,
        'hover': DroneRole.HOVER,
        'random': DroneRole.RANDOM
    }
    
    pursuer_role = role_map.get(args.pursuer_role.lower(), DroneRole.PURSUER)
    evader_role = role_map.get(args.evader_role.lower(), DroneRole.EVADER)
    
    # Find latest models if requested, or use provided paths
    if args.find_latest:
        model_paths = find_latest_models()
    else:
        model_paths = {}
        if args.pursuer and pursuer_role not in [DroneRole.HOVER, DroneRole.RANDOM]:
            model_paths[pursuer_role] = args.pursuer
        if args.evader and evader_role not in [DroneRole.HOVER, DroneRole.RANDOM]:
            model_paths[evader_role] = args.evader
    
    if not model_paths and all(role in [DroneRole.HOVER, DroneRole.RANDOM] for role in [pursuer_role, evader_role]):
        print("Note: No models needed as both drones are set to HOVER or RANDOM roles")
    elif not model_paths:
        print("Error: No models specified. Use --pursuer and --evader arguments or --find-latest")
        return
    
    try:
        # Load models
        agents, agent_roles, _ = load_models(model_paths, device)
        
        # Create drone configurations
        drone_configs = [
            DroneConfig(
                role=pursuer_role,
                start_pos=np.array([1, 1, 1]),
                start_orn=np.array([0, 0, 0]),
                action_length=args.pursuer_action_length
            ),
            DroneConfig(
                role=evader_role,
                start_pos=np.array([-1, -1, 1]),
                start_orn=np.array([0, 0, 0]),
                action_length=args.evader_action_length
            )
        ]
        
        # Evaluate
        evaluate(
            agents, 
            agent_roles,
            drone_configs=drone_configs,
            num_episodes=args.episodes,
            render=not args.no_render,
            sleep_time=args.sleep
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable model directories:")
        
        # List available model directories
        base_dir = "weights/pursuit_evade"
        if os.path.exists(base_dir):
            for d in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, d)):
                    print(f"  - {d}")
            print("\nTry running with --find-latest flag to use the most recent model.")
        else:
            print(f"  Directory {base_dir} not found.")
            print("  No trained models found. Please train models first using train_dqn_pursuit_evasion.py")

if __name__ == "__main__":
    main()
