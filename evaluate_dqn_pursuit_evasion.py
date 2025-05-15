import os
import numpy as np
import torch
import argparse
import time
import datetime
from task.lider_drone_base import LidarDroneBaseEnv, DroneRole, DroneConfig
from RL.dqn_agent import DroneDQNAgent, QNetwork

def load_models(pursuer_path, evader_path, device='cpu'):
    """Load trained models from saved weights"""
    
    # Check if model paths exist
    if not os.path.exists(pursuer_path):
        raise FileNotFoundError(f"Pursuer model not found at: {pursuer_path}\nTry using the --find-latest flag or specify the correct path.")
    
    if not os.path.exists(evader_path):
        raise FileNotFoundError(f"Evader model not found at: {evader_path}\nTry using the --find-latest flag or specify the correct path.")
    
    # Define drone configurations
    drone_configs = [
        DroneConfig(
            role=DroneRole.PURSUER,
            start_pos=np.array([1, 1, 1]),
            start_orn=np.array([0, 0, 0])
        ),
        DroneConfig(
            role=DroneRole.EVADER,
            start_pos=np.array([-1, -1, 1]),
            start_orn=np.array([0, 0, 0])
        )
    ]
    
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
    
    # Get agent names and identify pursuer and evader
    agent_names = list(observations.keys())
    pursuer_agent_name = None
    evader_agent_name = None
    
    for agent_name in agent_names:
        if env.agent_roles[agent_name] == DroneRole.PURSUER:
            pursuer_agent_name = agent_name
        else:
            evader_agent_name = agent_name
    
    print(f"Agent names: Pursuer = {pursuer_agent_name}, Evader = {evader_agent_name}")
    
    # Get observation and action spaces
    pursuer_obs_space = env.observation_space(pursuer_agent_name)
    pursuer_act_space = env.action_space(pursuer_agent_name)
    
    evader_obs_space = env.observation_space(evader_agent_name)
    evader_act_space = env.action_space(evader_agent_name)
    
    # Initialize agents
    pursuer = DroneDQNAgent(pursuer_obs_space, pursuer_act_space, device=device)
    evader = DroneDQNAgent(evader_obs_space, evader_act_space, device=device)
    
    # Load saved weights
    print(f"Loading pursuer model from {pursuer_path}")
    pursuer.q_net.load_state_dict(torch.load(pursuer_path, map_location=device))
    
    print(f"Loading evader model from {evader_path}")
    evader.q_net.load_state_dict(torch.load(evader_path, map_location=device))
    
    # Set to evaluation mode
    pursuer.q_net.eval()
    evader.q_net.eval()
    
    return pursuer, evader, (pursuer_agent_name, evader_agent_name)

def find_latest_models(base_dir="weights/pursuit_evade"):
    """Find the latest trained models in the weights directory"""
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} not found.")
        return None, None
    
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
        
        # Find best pursuer model
        pursuer_models = [f for f in os.listdir(latest_dir) if f.startswith("pursuer_") and f.endswith(".pt")]
        
        # Find best evader model
        evader_models = [f for f in os.listdir(latest_dir) if f.startswith("evader_") and f.endswith(".pt")]
        
        if not pursuer_models or not evader_models:
            print(f"Could not find both pursuer and evader models in {latest_dir}")
        else:
            # Prioritize final models, then highest step count
            def get_best_model(models):
                if "final.pt" in models:
                    return "final.pt"
                elif "best.pt" in models:
                    return "best.pt"
                
                # Extract step numbers and find the highest
                step_models = [m for m in models if m != "0.pt" and not m.endswith("best.pt") and not m.endswith("final.pt")]  # Exclude initial and special models
                if not step_models:
                    return "0.pt" if "0.pt" in models else models[0]  # Return initial if no others exist, or first model
                
                # Sort by step number
                def get_step_number(filename):
                    try:
                        if '_' in filename and '.' in filename:
                            return int(filename.split('_')[1].split('.')[0])
                        return 0
                    except (IndexError, ValueError):
                        return 0
                        
                step_models.sort(key=get_step_number, reverse=True)
                return step_models[0]
            
            # Get best models
            best_pursuer = get_best_model(pursuer_models)
            best_evader = get_best_model(evader_models)
            
            pursuer_model = os.path.join(latest_dir, best_pursuer)
            evader_model = os.path.join(latest_dir, best_evader)
            
            print(f"Selected pursuer model: {pursuer_model}")
            print(f"Selected evader model: {evader_model}")
            
            return pursuer_model, evader_model
    
    # If no models found in subdirectories, check root directory
    print("No models found in timestamp directories, checking root directory...")
    
    # Look for models directly in the base directory
    pursuer_models = [f for f in os.listdir(base_dir) if f.startswith("pursuer_") and f.endswith(".pt")]
    evader_models = [f for f in os.listdir(base_dir) if f.startswith("evader_") and f.endswith(".pt")]
    
    if pursuer_models and evader_models:
        print(f"Found models in root directory: {base_dir}")
        
        pursuer_model = os.path.join(base_dir, pursuer_models[0])
        evader_model = os.path.join(base_dir, evader_models[0])
        
        print(f"Selected pursuer model: {pursuer_model}")
        print(f"Selected evader model: {evader_model}")
        
        return pursuer_model, evader_model
    
    # No models found anywhere
    print(f"No suitable models found in {base_dir}")
    return None, None

def evaluate(pursuer, evader, agent_names, num_episodes=10, render=True, sleep_time=0.01, swap_roles=False):
    """Evaluate trained agents"""
    
    pursuer_agent_name, evader_agent_name = agent_names
    
    # Create drone configurations based on whether roles are swapped
    if swap_roles:
        print("Swapping pursuer and evader roles!")
        drone_configs = [
            DroneConfig(
                role=DroneRole.EVADER,  # Swapped: first drone is evader
                start_pos=np.array([1, 1, 1]),
                start_orn=np.array([0, 0, 0])
            ),
            DroneConfig(
                role=DroneRole.PURSUER,  # Swapped: second drone is pursuer
                start_pos=np.array([-1, -1, 1]),
                start_orn=np.array([0, 0, 0])
            )
        ]
    else:
        drone_configs = [
            DroneConfig(
                role=DroneRole.PURSUER,
                start_pos=np.array([1, 1, 1]),
                start_orn=np.array([0, 0, 0])
            ),
            DroneConfig(
                role=DroneRole.EVADER,
                start_pos=np.array([-1, -1, 1]),
                start_orn=np.array([0, 0, 0])
            )
        ]
    
    # Create environment with drone configurations
    env = LidarDroneBaseEnv(
        lidar_reach=4.0, 
        num_ray=20, 
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=render,  # 시각화 활성화
        render_mode="human" if render else None  # 렌더링 모드 설정
    )
    
    print(f"Simulation rendering is {'enabled' if render else 'disabled'}")
    if render:
        print(f"Sleep between steps: {sleep_time} seconds (use --sleep 0 for fastest evaluation)")
    
    # Reset to initialize agent list
    obs, _ = env.reset()
    
    # Get agent names in the evaluation environment based on their roles
    eval_agent_names = list(obs.keys())
    eval_pursuer_name = None
    eval_evader_name = None
    
    for name in eval_agent_names:
        if env.agent_roles[name] == DroneRole.PURSUER:
            eval_pursuer_name = name
        else:
            eval_evader_name = name
    
    print(f"Evaluation environment agent roles: Pursuer = {eval_pursuer_name}, Evader = {eval_evader_name}")
    
    # Track statistics
    episode_rewards = {agent: [] for agent in agent_names}
    episode_lengths = []
    capture_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = {agent: 0 for agent in agent_names}
        step_count = 0
        
        while not done:
            if render and sleep_time > 0:
                time.sleep(sleep_time)  # Slow down visualization
                
            actions = {}
            
            # Get actions for each agent
            for env_agent_name in list(obs.keys()):
                agent_obs = obs[env_agent_name]
                
                # Determine which agent should control this drone based on role
                if env.agent_roles[env_agent_name] == DroneRole.PURSUER:
                    agent = pursuer
                    original_name = pursuer_agent_name
                else:
                    agent = evader
                    original_name = evader_agent_name
                
                actions[env_agent_name] = int(agent.exploit(agent_obs)[0])
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update statistics - match environment agent to original agent based on role
            for env_agent_name, reward in rewards.items():
                if env.agent_roles[env_agent_name] == DroneRole.PURSUER:
                    episode_reward[pursuer_agent_name] += reward
                else:
                    episode_reward[evader_agent_name] += reward
            
            step_count += 1
            
            # Check for capture
            capture_occurred = any(info.get("capture", False) for info in infos.values())
            if capture_occurred:
                capture_count += 1
                if render:
                    print(f"Capture occurred at step {step_count}!")
            
            # Check for episode termination
            done = all(terminations.values()) or all(truncations.values())
        
        # Store episode statistics
        for agent_name in agent_names:
            episode_rewards[agent_name].append(episode_reward[agent_name])
        episode_lengths.append(step_count)
        
        print(f"Episode {episode+1}/{num_episodes}: " +
              f"Pursuer Reward={episode_reward[pursuer_agent_name]:.2f}, " +
              f"Evader Reward={episode_reward[evader_agent_name]:.2f}, " +
              f"Length={step_count}")
    
    # Compute average statistics
    avg_rewards = {agent: np.mean(rewards) for agent, rewards in episode_rewards.items()}
    avg_length = np.mean(episode_lengths)
    capture_rate = capture_count / num_episodes
    
    print("\nEvaluation Results:")
    print(f"Average Pursuer Reward: {avg_rewards[pursuer_agent_name]:.2f}")
    print(f"Average Evader Reward: {avg_rewards[evader_agent_name]:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Capture Rate: {capture_rate:.2f} ({capture_count}/{num_episodes})")
    
    return {
        "avg_rewards": avg_rewards,
        "avg_length": avg_length,
        "capture_rate": capture_rate
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agents for drone pursuit-evasion")
    parser.add_argument("--pursuer", type=str, default="weights/pursuit_evade/pursuer_final.pt",
                        help="Path to the pursuer model weights")
    parser.add_argument("--evader", type=str, default="weights/pursuit_evade/evader_final.pt",
                        help="Path to the evader model weights")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (headless mode)")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Sleep time between steps for visualization (set to 0 for fastest evaluation)")
    parser.add_argument("--swap-roles", action="store_true",
                        help="Swap the pursuer and evader roles for evaluation")
    parser.add_argument("--find-latest", action="store_true",
                        help="Automatically find and use the latest saved models")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate a timestamp for this evaluation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Evaluation started at: {timestamp}")
    
    # Find the latest models if requested
    if args.find_latest:
        pursuer_path, evader_path = find_latest_models()
        if pursuer_path is None or evader_path is None:
            print("Could not find latest models. Please specify model paths manually.")
            return
        args.pursuer = pursuer_path
        args.evader = evader_path
    
    model_info = os.path.basename(os.path.dirname(args.pursuer))
    print(f"Evaluating model: {model_info}")
    
    try:
        # Load models
        pursuer, evader, agent_names = load_models(args.pursuer, args.evader, device)
        
        # Evaluate
        evaluate(
            pursuer, 
            evader, 
            agent_names=agent_names, 
            num_episodes=args.episodes,
            render=not args.no_render,
            sleep_time=args.sleep,
            swap_roles=args.swap_roles
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
