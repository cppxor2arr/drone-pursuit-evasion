# Drone Pursuit-Evasion Training Guide

This repository implements a multi-agent reinforcement learning environment for drone pursuit-evasion scenarios using PyBullet simulation. The system supports multiple RL algorithms (DQN, PPO, SAC) and provides flexible configuration through Hydra.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pybullet; print('PyBullet installed successfully')"
```

### Basic Training

```bash
# Train DQN vs DQN (default)
python train_hydra.py

# Train with specific scenario
python train_hydra.py scenario=pursuit_evasion

# Train with different algorithms
python train_hydra.py scenario=sac_vs_sac
python train_hydra.py scenario=ppo_vs_random
python train_hydra.py scenario=mixed_agents
```

## 🎯 Available Algorithms

### 1. Deep Q-Network (DQN)
- **Best for**: Discrete action spaces, sample efficiency
- **Characteristics**: Off-policy, experience replay, target networks
- **Configuration**: `conf/agent/dqn.yaml`

### 2. Proximal Policy Optimization (PPO)
- **Best for**: Stable training, continuous control
- **Characteristics**: On-policy, policy gradients, clip objective
- **Configuration**: `conf/agent/ppo.yaml`

### 3. Soft Actor-Critic (SAC)
- **Best for**: Continuous control, sample efficiency, robustness
- **Characteristics**: Off-policy, maximum entropy, twin critics
- **Configuration**: `conf/agent/sac.yaml`

### 4. Special Agents
- **Random Agent**: Random action selection (baseline)
- **Hovering Agent**: Always selects hover action

## 📋 Training Scenarios

### Available Scenarios

| Scenario | Description | Pursuer | Evader | Use Case |
|----------|-------------|---------|--------|----------|
| `pursuit_evasion` | Classic DQN vs DQN | DQN | DQN | Baseline comparison |
| `sac_vs_sac` | SAC vs SAC | SAC | SAC | Advanced continuous control |
| `ppo_vs_random` | PPO vs Random | PPO | Random | PPO evaluation |
| `sac_vs_random` | SAC vs Random | SAC | Random | SAC evaluation |
| `mixed_agents` | Different algorithms | PPO | DQN | Cross-algorithm comparison |

### New Pursuer Algorithm Testing Scenarios

| Scenario | Description | Pursuer | Evader | Use Case |
|----------|-------------|---------|--------|----------|
| `ppo_pursuer_vs_hovering_evader` | PPO vs Stationary | PPO | Hovering | PPO basic testing |
| `dqn_pursuer_vs_hovering_evader` | DQN vs Stationary | DQN | Hovering | DQN basic testing |
| `sac_pursuer_vs_hovering_evader` | SAC vs Stationary | SAC | Hovering | SAC basic testing |
| `ppo_pursuer_vs_random_evader` | PPO vs Random | PPO | Random | PPO robustness testing |
| `dqn_pursuer_vs_random_evader` | DQN vs Random | DQN | Random | DQN robustness testing |

### Cross-Algorithm Competition Scenarios

| Scenario | Description | Pursuer | Evader | Use Case |
|----------|-------------|---------|--------|----------|
| `ppo_pursuer_vs_dqn_evader` | PPO vs DQN | PPO | DQN | Policy vs Value methods |
| `dqn_pursuer_vs_ppo_evader` | DQN vs PPO | DQN | PPO | Value vs Policy methods |
| `sac_pursuer_vs_dqn_evader` | SAC vs DQN | SAC | DQN | Continuous vs Discrete |
| `sac_pursuer_vs_ppo_evader` | SAC vs PPO | SAC | PPO | Off-policy vs On-policy |

### Create Custom Scenarios

Create a new scenario file in `conf/scenario/`:

```yaml
# conf/scenario/my_scenario.yaml
drones:
  - role: PURSUER
    agent_type: sac  # dqn, ppo, sac, random, hovering
    start_pos: [1.0, 1.0, 1.0]
    start_orn: [0.0, 0.0, 0.0]
    action_length: 7.0
    is_training: true
    resume_from: null
    name: "my_pursuer"
  
  - role: EVADER
    agent_type: dqn
    start_pos: [-1.0, -1.0, 1.0]
    start_orn: [0.0, 0.0, 0.0]
    action_length: 5.0
    is_training: true
    resume_from: "path/to/pretrained/model.pt"
    name: "my_evader"
```

## 🔧 Configuration System

### Hydra Configuration Structure

```
conf/
├── config.yaml              # Main configuration
├── agent/
│   ├── dqn.yaml             # DQN hyperparameters
│   ├── ppo.yaml             # PPO hyperparameters
│   └── sac.yaml             # SAC hyperparameters
└── scenario/
    ├── pursuit_evasion.yaml  # DQN vs DQN
    ├── sac_vs_sac.yaml      # SAC vs SAC
    ├── ppo_vs_random.yaml   # PPO vs Random
    ├── sac_vs_random.yaml   # SAC vs Random
    └── mixed_agents.yaml    # Mixed algorithms
```

### Command Line Overrides

```bash
# Override hyperparameters
python train_hydra.py agent.learning_rate=0.001 agent.gamma=0.95

# Change training duration
python train_hydra.py training.total_timesteps=100000

# Modify environment
python train_hydra.py environment.lidar_reach=5.0 environment.num_ray=32

# Enable/disable WandB
python train_hydra.py wandb.mode=online
python train_hydra.py wandb.mode=disabled

# Change device
python train_hydra.py device=cuda
python train_hydra.py device=cpu
```

### Key Configuration Options

#### Agent Parameters (DQN)
```yaml
gamma: 0.99                    # Discount factor
learning_rate: 0.0005          # Learning rate
epsilon_start: 1.0             # Initial exploration
epsilon_end: 0.05              # Final exploration
epsilon_decay: 7000000         # Exploration decay steps
batch_size: 32                 # Training batch size
buffer_size: 10000             # Replay buffer size
network:
  hidden_dims: [120, 84, 84]   # Network architecture
```

#### Agent Parameters (PPO)
```yaml
gamma: 0.99                    # Discount factor
learning_rate: 0.0003          # Learning rate
clip_range: 0.2                # PPO clip parameter
value_coef: 0.5                # Value function coefficient
entropy_coef: 0.01             # Entropy coefficient
gae_lambda: 0.95               # GAE lambda
ppo_epochs: 10                 # Epochs per update
batch_size: 256                # Batch size
buffer_size: 2048              # Rollout buffer size
```

#### Agent Parameters (SAC)
```yaml
gamma: 0.99                    # Discount factor
learning_rate: 0.0003          # Learning rate
tau: 0.005                     # Soft update coefficient
alpha: 0.2                     # Temperature parameter
automatic_entropy_tuning: true # Auto-tune temperature
batch_size: 256                # Training batch size
buffer_size: 100000            # Replay buffer size
warmup_steps: 1000             # Random action steps
```

#### Training Parameters
```yaml
total_timesteps: 50000         # Total training steps
save_interval: 10000           # Model save frequency
evaluate_interval: 5000        # Evaluation frequency
eval_episodes: 5               # Episodes per evaluation
log_interval: 100              # Logging frequency
```

## 🏃‍♂️ Training Examples

### 1. Basic DQN Training
```bash
# Default DQN vs DQN training
python train_hydra.py

# With custom hyperparameters
python train_hydra.py \
  agent.learning_rate=0.001 \
  agent.epsilon_decay=1000000 \
  training.total_timesteps=100000
```

### 2. SAC Training
```bash
# SAC vs SAC with default settings
python train_hydra.py scenario=sac_vs_sac

# SAC with custom parameters
python train_hydra.py scenario=sac_vs_sac \
  agent.learning_rate=0.0001 \
  agent.alpha=0.1 \
  agent.batch_size=512
```

### 3. PPO Training
```bash
# PPO vs Random
python train_hydra.py scenario=ppo_vs_random

# PPO with larger networks
python train_hydra.py scenario=ppo_vs_random \
  agent.network.hidden_dims=[512,512,256]
```

### 4. Mixed Algorithm Training
```bash
# PPO pursuer vs DQN evader
python train_hydra.py scenario=mixed_agents

# With WandB logging
python train_hydra.py scenario=mixed_agents \
  wandb.mode=online \
  wandb.project=drone-pursuit \
  experiment_name=mixed_training
```

### 5. Resume Training
```bash
# Resume from checkpoint
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/20241201_143022/pursuer_50000.pt"
```

## 📊 Monitoring and Logging

### WandB Integration
```bash
# Enable WandB logging
python train_hydra.py \
  wandb.mode=online \
  wandb.project=my-drone-project \
  wandb.entity=my-username \
  experiment_name=my_experiment

# Disable WandB
python train_hydra.py wandb.mode=disabled
```

### Local Logging
- Models saved to: `weights/pursuit_evade/YYYYMMDD_HHMMSS/`
- Hydra outputs: `outputs/YYYY-MM-DD/HH-MM-SS/`
- Configuration: Automatically saved with each run

## 🎮 Evaluation

### Evaluate Trained Models
```bash
# Evaluate with GUI (requires X11)
python evaluate_hydra.py

# Evaluate without GUI
python evaluate_trained_models.py

# Evaluate specific models
python evaluate_trained_models.py \
  --weights-dir weights/pursuit_evade/20241201_143022 \
  --episodes 10

# Evaluate with custom scenario
python evaluate_hydra.py scenario=sac_vs_sac
```

### Evaluation Options
```bash
# More episodes for better statistics
python evaluate_trained_models.py --episodes 20

# Quiet mode (less verbose output)
python evaluate_trained_models.py --quiet

# Specific device
python evaluate_trained_models.py --device cuda
```

## 🐛 Troubleshooting

### Common Issues

#### 1. PyBullet/X11 Issues
```bash
# For headless servers
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &

# Or disable rendering
python train_hydra.py environment.render_simulation=false
```

#### 2. CUDA Issues
```bash
# Force CPU
python train_hydra.py device=cpu

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Configuration Errors
```bash
# Validate configuration
python train_hydra.py --cfg job

# Check available configurations
python train_hydra.py --help
```

#### 4. Memory Issues
```bash
# Reduce batch size
python train_hydra.py agent.batch_size=16

# Reduce buffer size
python train_hydra.py agent.buffer_size=5000
```

### Performance Tips

1. **Use GPU**: Set `device=cuda` for faster training
2. **Tune batch size**: Larger batches for stability, smaller for memory
3. **Adjust buffer size**: Larger buffers for sample diversity
4. **Monitor metrics**: Use WandB for real-time monitoring

## 📈 Advanced Usage

### Hyperparameter Sweeps
```bash
# Manual sweep
for lr in 0.001 0.0005 0.0001; do
  python train_hydra.py agent.learning_rate=$lr experiment_name=sweep_lr_$lr
done

# With WandB sweeps (create sweep config first)
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

### Custom Reward Functions
Modify `task/lider_drone_base.py`:
```python
def compute_pursuer_reward(self, agent_id, other_agent_id, collision, out_of_bounds):
    # Custom pursuer reward logic
    pass

def compute_evader_reward(self, agent_id, other_agent_id, collision, out_of_bounds):
    # Custom evader reward logic
    pass
```

### Multi-GPU Training
```bash
# Specify GPU
CUDA_VISIBLE_DEVICES=0 python train_hydra.py device=cuda

# Multiple parallel runs
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python train_hydra.py device=cuda &
done
```

## 🤝 Contributing

1. **Add new algorithms**: Create agent in `RL/` directory
2. **Add scenarios**: Create YAML in `conf/scenario/`
3. **Extend environment**: Modify `task/lider_drone_base.py`
4. **Add configurations**: Create YAML in `conf/agent/`

## 📚 References

- [Hydra Documentation](https://hydra.cc/)
- [WandB Documentation](https://docs.wandb.ai/)
- [PyBullet Documentation](https://pybullet.org/)
- [PyFlyt Documentation](https://pyflyt.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License.

## 🔧 Improved Training System

### Modular Training Architecture

The training system has been refactored into smaller, focused functions for better maintainability and debugging:

- **`setup_training()`**: Handles seed, device, directories, and WandB initialization
- **`create_training_environment()`**: Sets up the drone environment and agent roles
- **`create_training_agents()`**: Creates and configures all agents based on scenario
- **`run_training_loop()`**: Executes the main training loop with experience processing
- **`save_periodic_models()`**: Handles model checkpointing during training
- **`save_final_models()`**: Saves final trained models
- **`log_episode_results()`**: Manages episode logging to WandB

### Benefits of Modular Design

1. **Easier Debugging**: Each function has a single responsibility
2. **Better Testability**: Individual components can be tested in isolation
3. **Improved Readability**: Code is organized into logical sections
4. **Enhanced Maintainability**: Changes to one aspect don't affect others
5. **Flexible Extension**: Easy to add new functionality or modify existing behavior

## 🎯 Available Algorithms 