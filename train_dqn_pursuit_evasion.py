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

def train_agents(resume_from=None, epsilon_start=None, train_mode="both"):
    """
    Train DQN agents for pursuit-evasion
    
    Args:
        resume_from (str, optional): Path to directory containing models to resume from
        epsilon_start (float, optional): Override the starting epsilon value when resuming
        train_mode (str): Which agents to train - "both", "pursuer", or "evader"
    """
    # Get current time for wandb run name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"DQN_PE_{current_time}_{train_mode}"
    
    # Create weights directory for this run
    weights_dir = f"weights/pursuit_evade/{current_time}"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Track best model performance for saving
    best_models = {
        'pursuer': {'step': 0, 'performance': float('-inf')},  # Higher reward is better
        'evader': {'step': 0, 'performance': float('-inf')}    # Higher reward is better
    }
    
    # Starting step for training if resuming
    start_step = 0
    
    # Set up epsilon parameters - copy from globals to local variables
    local_epsilon_start = EPSILON_START
    local_epsilon_end = EPSILON_END
    local_epsilon_decay = EPSILON_DECAY
    
    # Determine if we're resuming from previous training
    if resume_from:
        print(f"Resuming training from {resume_from}")
        # Get the highest step number from the directory
        model_files = os.listdir(resume_from)
        step_numbers = []
        for f in model_files:
            if f.startswith(("pursuer_", "evader_")) and f.endswith(".pt") and not f.endswith("final.pt") and not f.endswith("best.pt"):
                try:
                    step = int(f.split('_')[1].split('.')[0])
                    step_numbers.append(step)
                except (IndexError, ValueError):
                    continue
        
        if step_numbers:
            start_step = max(step_numbers)
            print(f"Resuming from step {start_step}")
    
    # Set starting epsilon based on resume point or use default
    if epsilon_start is not None:
        local_epsilon_start = epsilon_start
        print(f"Using provided epsilon: {local_epsilon_start}")
    elif resume_from and start_step > 0:
        # Calculate what epsilon would be at this point in training
        local_epsilon_start = max(
            local_epsilon_end,
            local_epsilon_start - (local_epsilon_start - local_epsilon_end) * (start_step / local_epsilon_decay)
        )
        print(f"Starting with epsilon = {local_epsilon_start} based on resume point")
    
    # Initialize wandb with metadata in run name
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
            "resumed_from": resume_from,
            "train_mode": train_mode
        }
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    # print(observations)
    # print(env.observation_space(pursuer_agent_name) )
    # print(env.action_space(pursuer_agent_name))

    # Get agent names from observations
    agent_names = list(observations.keys())
    
    # Identify pursuer and evader based on the environment's roles
    pursuer_agent_name = None
    evader_agent_name = None
    for agent_name in agent_names:
        if env.agent_roles[agent_name] == DroneRole.PURSUER:
            pursuer_agent_name = agent_name
        else:
            evader_agent_name = agent_name
    
    print(f"Agent names: Pursuer = {pursuer_agent_name}, Evader = {evader_agent_name}")
    
    # Get action and observation spaces for each agent
    pursuer_obs_space = env.observation_space(pursuer_agent_name)
    pursuer_act_space = env.action_space(pursuer_agent_name)
    
    evader_obs_space = env.observation_space(evader_agent_name)
    evader_act_space = env.action_space(evader_agent_name)
    
    # Initialize agents with the DroneDQNAgent class
    pursuer = DroneDQNAgent(
        observation_space=pursuer_obs_space, 
        action_space=pursuer_act_space, 
        device=device,
        gamma=GAMMA,
        epsilon_start=local_epsilon_start,
        epsilon_end=local_epsilon_end,
        epsilon_decay=local_epsilon_decay,
        learning_rate=LEARNING_RATE
    )
    
    evader = DroneDQNAgent(
        observation_space=evader_obs_space, 
        action_space=evader_act_space, 
        device=device,
        gamma=GAMMA,
        epsilon_start=local_epsilon_start,
        epsilon_end=local_epsilon_end,
        epsilon_decay=local_epsilon_decay,
        learning_rate=LEARNING_RATE
    )
    
    # Log training mode
    print(f"Training mode: {train_mode}")
    
    # If resuming, load the models
    if resume_from and start_step > 0:
        pursuer_path = os.path.join(resume_from, f"pursuer_{start_step}.pt")
        evader_path = os.path.join(resume_from, f"evader_{start_step}.pt")
        
        if os.path.exists(pursuer_path):
            print(f"Loading pursuer model from {pursuer_path}")
            pursuer.load(pursuer_path)
        else:
            print(f"Warning: Could not find pursuer model at {pursuer_path}")
        
        if os.path.exists(evader_path):
            print(f"Loading evader model from {evader_path}")
            evader.load(evader_path)
        else:
            print(f"Warning: Could not find evader model at {evader_path}")
    
    # Set training mode for agents
    if train_mode in ["both", "pursuer"]:
        pursuer.train()
    else:
        pursuer.eval()
        
    if train_mode in ["both", "evader"]:
        evader.train()
    else:
        evader.eval()
    
    # Initialize replay buffers using the DroneDQNAgent's buffer class
    pursuer_buffer = pursuer.create_buffer(BUFFER_SIZE)
    evader_buffer = evader.create_buffer(BUFFER_SIZE)
    
    # Save initial models if not resuming
    if not resume_from:
        pursuer.save(path=f"{weights_dir}/pursuer", step=0)
        evader.save(path=f"{weights_dir}/evader", step=0)
        print(f"Saved initial models at step 0")
    
    # Training loop
    obs = observations.copy()
    episode_rewards = {agent: 0 for agent in agent_names}
    episode_lengths = {agent: 0 for agent in agent_names}
    episode_count = 0
    death_reasons = {'collision': 0, 'out_of_bounds': 0, 'capture': 0, 'max_steps': 0}
    
    print("Starting training...")
    for global_step in range(start_step, TOTAL_TIMESTEPS):
        actions = {}
        # Select actions for each agent
        for agent_name in list(obs.keys()):
            agent_obs = obs[agent_name]
            
            # Select action based on training mode
            if agent_name == pursuer_agent_name:
                if train_mode in ["both", "pursuer"]:
                    # Use epsilon-greedy selection from the agent
                    actions[agent_name] = pursuer.select_action(agent_obs)
                else:
                    # Use exploitation only (no exploration) when not training
                    actions[agent_name] = int(pursuer.exploit(agent_obs)[0])
            else:  # evader
                if train_mode in ["both", "evader"]:
                    # Use epsilon-greedy selection from the agent
                    actions[agent_name] = evader.select_action(agent_obs)
                else:
                    # Use exploitation only (no exploration) when not training
                    actions[agent_name] = int(evader.exploit(agent_obs)[0])
        
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
        
        # Update replay buffers
        if len(next_obs) == len(obs):  # Make sure we have observations for all agents
            for agent_name in agent_names:
                if agent_name in obs and agent_name in next_obs:
                    # Track episode stats
                    episode_rewards[agent_name] += rewards[agent_name]
                    episode_lengths[agent_name] += 1
                    
                    # Update replay buffer based on training mode
                    if (agent_name == pursuer_agent_name and train_mode in ["both", "pursuer"]) or \
                       (agent_name == evader_agent_name and train_mode in ["both", "evader"]):
                        done = terminations[agent_name] or truncations[agent_name]
                        
                        # Add experience to appropriate buffer
                        if agent_name == pursuer_agent_name:
                            pursuer_buffer.add(
                                obs[agent_name],
                                actions[agent_name],
                                rewards[agent_name],
                                next_obs[agent_name],
                                done
                            )
                        else:  # evader
                            evader_buffer.add(
                                obs[agent_name],
                                actions[agent_name],
                                rewards[agent_name],
                                next_obs[agent_name],
                                done
                            )
        
        # Train agents if enough samples are collected
        if global_step > BATCH_SIZE:
            # Train pursuer (if enabled)
            if train_mode in ["both", "pursuer"] and len(pursuer_buffer) > BATCH_SIZE:
                # Sample batch and update network
                batch = pursuer_buffer.sample(BATCH_SIZE)
                loss = pursuer.update(batch)
                
                # Update exploration rate
                pursuer.update_epsilon_cosine()
                
                # Log training metrics (prefixed with 'train/')
                wandb.log({
                    "train/pursuer_loss": loss,
                    "train/pursuer_epsilon": pursuer.get_epsilon(),
                    "global_step": global_step
                })
            
            # Train evader (if enabled)
            if train_mode in ["both", "evader"] and len(evader_buffer) > BATCH_SIZE:
                # Sample batch and update network
                batch = evader_buffer.sample(BATCH_SIZE)
                loss = evader.update(batch)
                
                # Update exploration rate
                evader.update_epsilon_cosine()
                
                # Log training metrics (prefixed with 'train/')
                wandb.log({
                    "train/evader_loss": loss,
                    "train/evader_epsilon": evader.get_epsilon(),
                    "global_step": global_step
                })
        
        # Periodically update target networks
        if global_step % TARGET_UPDATE_INTERVAL == 0:
            if train_mode in ["both", "pursuer"]:
                pursuer.update_target_network()
            if train_mode in ["both", "evader"]:
                evader.update_target_network()
            print(f"Updated target networks at step {global_step}")
        
        # Save models periodically
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            pursuer.save(path=f"{weights_dir}/pursuer", step=global_step)
            evader.save(path=f"{weights_dir}/evader", step=global_step)
            print(f"Saved checkpoint models at step {global_step}")
        
        # Check if episode is done
        done = all(terminations.values()) or all(truncations.values())
        
        if done:
            # Log episode stats with prefix 'train/'
            for agent_name in episode_rewards:
                agent_role = "pursuer" if agent_name == pursuer_agent_name else "evader"
                wandb.log({
                    f"train/{agent_role}_episode_reward": episode_rewards[agent_name],
                    f"train/{agent_role}_episode_length": episode_lengths[agent_name],
                    "train/episode": episode_count,
                    "train/captures": death_reasons['capture'],
                    "train/collisions": death_reasons['collision'],
                    "train/out_of_bounds": death_reasons['out_of_bounds'],
                    "train/max_steps": death_reasons['max_steps'],
                })
                print(f"Episode {episode_count}, {agent_role.capitalize()}: Reward={episode_rewards[agent_name]:.2f}, Length={episode_lengths[agent_name]}")
            
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
            eval_results = evaluate_agents(pursuer, evader, (pursuer_agent_name, evader_agent_name), global_step, train_mode)
            
            # Check if this is the best model for pursuer
            if eval_results['avg_rewards'][pursuer_agent_name] > best_models['pursuer']['performance']:
                best_models['pursuer']['performance'] = eval_results['avg_rewards'][pursuer_agent_name]
                best_models['pursuer']['step'] = global_step
                
                # Save best pursuer model
                pursuer.save(path=f"{weights_dir}/pursuer", is_best=True)
                print(f"New best pursuer model at step {global_step} with reward {eval_results['avg_rewards'][pursuer_agent_name]:.2f}")
            
            # Check if this is the best model for evader
            if eval_results['avg_rewards'][evader_agent_name] > best_models['evader']['performance']:
                best_models['evader']['performance'] = eval_results['avg_rewards'][evader_agent_name]
                best_models['evader']['step'] = global_step
                
                # Save best evader model
                evader.save(path=f"{weights_dir}/evader", is_best=True)
                print(f"New best evader model at step {global_step} with reward {eval_results['avg_rewards'][evader_agent_name]:.2f}")
    
    # Save final models
    pursuer.save(path=f"{weights_dir}/pursuer")
    evader.save(path=f"{weights_dir}/evader")
    
    # Save information about best models
    with open(f"{weights_dir}/best_models.txt", "w") as f:
        f.write(f"Best pursuer model: step {best_models['pursuer']['step']} with performance {best_models['pursuer']['performance']:.2f}\n")
        f.write(f"Best evader model: step {best_models['evader']['step']} with performance {best_models['evader']['performance']:.2f}\n")
    
    print("Training complete!")
    print(f"Best pursuer model: step {best_models['pursuer']['step']} with reward {best_models['pursuer']['performance']:.2f}")
    print(f"Best evader model: step {best_models['evader']['step']} with reward {best_models['evader']['performance']:.2f}")
    print(f"Death reasons: {death_reasons}")
    
    # Finalize wandb
    wandb.finish()

def evaluate_agents(pursuer, evader, agent_names, global_step, train_mode="both", num_episodes=3):
    """Evaluate the performance of trained agents"""
    print(f"Evaluating agents at step {global_step}...")
    
    pursuer_agent_name, evader_agent_name = agent_names
    
    # Define drone configurations for evaluation
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
    
    # Create environment with rendering for evaluation
    eval_env = LidarDroneBaseEnv(
        lidar_reach=4.0,
        num_ray=20,
        flight_mode=7,
        drone_configs=drone_configs,
        render_simulation=True  # Enable visualization during evaluation
    )
    
    # Initialize with reset
    eval_obs, _ = eval_env.reset()
    
    # Get agent names in the evaluation environment based on their roles
    eval_agent_names = list(eval_obs.keys())
    eval_pursuer_name = None
    eval_evader_name = None
    
    for name in eval_agent_names:
        if eval_env.agent_roles[name] == DroneRole.PURSUER:
            eval_pursuer_name = name
        else:
            eval_evader_name = name
    
    # Map original agent names to evaluation environment's agent names
    name_mapping = {
        pursuer_agent_name: eval_pursuer_name,
        evader_agent_name: eval_evader_name
    }
    
    # Set agents to evaluation mode
    pursuer.eval()
    evader.eval()
    
    avg_rewards = {agent: 0 for agent in agent_names}
    avg_lengths = 0
    capture_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_rewards = {agent: 0 for agent in agent_names}
        episode_length = 0
        
        while not done and episode_length < 500:  # Add max episode length
            actions = {}
            
            # Deterministic actions for evaluation
            for env_agent_name in list(obs.keys()):
                if env_agent_name not in obs:
                    continue  # Skip agents without observations
                    
                agent_obs = obs[env_agent_name]
                
                # Determine which agent should control this drone
                if env_agent_name == eval_pursuer_name:
                    action = int(pursuer.exploit(agent_obs)[0])
                else:
                    action = int(evader.exploit(agent_obs)[0])
                
                actions[env_agent_name] = action
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = eval_env.step(actions)
            
            # Update rewards - map environment agent names to original agent names
            for env_agent_name, reward in rewards.items():
                if env_agent_name == eval_pursuer_name:
                    episode_rewards[pursuer_agent_name] += reward
                else:
                    episode_rewards[evader_agent_name] += reward
            
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
        for agent_name in agent_names:
            avg_rewards[agent_name] += episode_rewards[agent_name] / num_episodes
        avg_lengths += episode_length / num_episodes
        
        print(f"Eval Episode {episode+1}/{num_episodes}: " + 
              f"Pursuer Reward={episode_rewards[pursuer_agent_name]:.2f}, " +
              f"Evader Reward={episode_rewards[evader_agent_name]:.2f}, " +
              f"Length={episode_length}")
    
    # Log evaluation results with prefix 'eval/'
    wandb.log({
        "eval/pursuer_reward": avg_rewards[pursuer_agent_name],
        "eval/evader_reward": avg_rewards[evader_agent_name],
        "eval/episode_length": avg_lengths,
        "eval/capture_rate": capture_count / num_episodes,
        "eval/collision_rate": collision_count / num_episodes,
        "eval/out_of_bounds_rate": out_of_bounds_count / num_episodes,
        "global_step": global_step
    })
    
    print(f"Evaluation complete. " +
          f"Avg Pursuer Reward: {avg_rewards[pursuer_agent_name]:.2f}, " +
          f"Avg Evader Reward: {avg_rewards[evader_agent_name]:.2f}, " +
          f"Avg Length: {avg_lengths:.2f}, " +
          f"Capture Rate: {capture_count / num_episodes:.2f}")
    
    # Set agents back to their previous modes
    if train_mode in ["both", "pursuer"]:
        pursuer.train()
    if train_mode in ["both", "evader"]:
        evader.train()
    
    return {
        'avg_rewards': avg_rewards,
        'avg_length': avg_lengths,
        'capture_rate': capture_count / num_episodes
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN agents for drone pursuit-evasion")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to directory containing models to resume from")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Starting epsilon value (overrides default when resuming)")
    parser.add_argument("--train-mode", type=str, choices=["both", "pursuer", "evader"], default="both",
                        help="Select which agent to train (both, pursuer, or evader)")
    
    args = parser.parse_args()
    
    train_agents(args.resume, args.epsilon, args.train_mode) 