# 🚁 Drone Pursuit-Evasion Training Examples

This guide provides practical examples for training different agent combinations in the drone pursuit-evasion environment.

## 🚀 Quick Training Commands

### 1. Algorithm-Specific Training

```bash
# DQN vs DQN (Classic Deep Q-Learning)
python train_hydra.py scenario=pursuit_evasion

# PPO vs Random (Policy Gradient vs Baseline)
python train_hydra.py scenario=ppo_vs_random

# SAC vs Random (Soft Actor-Critic vs Baseline)
python train_hydra.py scenario=sac_vs_random

# SAC vs SAC (Advanced Continuous Control)
python train_hydra.py scenario=sac_vs_sac

# Mixed Algorithms (PPO vs DQN)
python train_hydra.py scenario=mixed_agents
```

### 2. Pursuer Algorithm Testing (vs Static Targets)

```bash
# Test PPO Pursuer vs Hovering Evader
python train_hydra.py scenario=ppo_pursuer_vs_hovering_evader

# Test DQN Pursuer vs Hovering Evader  
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader

# Test SAC Pursuer vs Hovering Evader
python train_hydra.py scenario=sac_pursuer_vs_hovering_evader

# Test Pursuers vs Random Evader
python train_hydra.py scenario=ppo_pursuer_vs_random_evader
python train_hydra.py scenario=dqn_pursuer_vs_random_evader
python train_hydra.py scenario=sac_vs_random  # SAC vs Random already exists
```

### 3. Cross-Algorithm Competition

```bash
# PPO Pursuer vs Intelligent Evaders
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader

# SAC Pursuer vs Intelligent Evaders
python train_hydra.py scenario=sac_pursuer_vs_dqn_evader
python train_hydra.py scenario=sac_pursuer_vs_ppo_evader

# DQN Pursuer vs Intelligent Evaders  
python train_hydra.py scenario=dqn_pursuer_vs_ppo_evader
```

### 4. Algorithm Comparison Matrix

| Pursuer | Evader | Scenario Command |
|---------|--------|------------------|
| PPO | Hovering | `scenario=ppo_pursuer_vs_hovering_evader` |
| DQN | Hovering | `scenario=dqn_pursuer_vs_hovering_evader` |
| SAC | Hovering | `scenario=sac_pursuer_vs_hovering_evader` |
| PPO | Random | `scenario=ppo_pursuer_vs_random_evader` |
| DQN | Random | `scenario=dqn_pursuer_vs_random_evader` |
| SAC | Random | `scenario=sac_vs_random` |
| PPO | DQN | `scenario=ppo_pursuer_vs_dqn_evader` |
| DQN | PPO | `scenario=dqn_pursuer_vs_ppo_evader` |
| SAC | DQN | `scenario=sac_pursuer_vs_dqn_evader` |
| SAC | PPO | `scenario=sac_pursuer_vs_ppo_evader` |
| DQN | DQN | `scenario=pursuit_evasion` |
| PPO | PPO | `scenario=ppo_vs_random` (modify config) |
| SAC | SAC | `scenario=sac_vs_sac` |

### 5. Training Duration Examples

```bash
# Quick test (1K steps)
python train_hydra.py training.total_timesteps=1000

# Short training (10K steps)
python train_hydra.py training.total_timesteps=10000

# Medium training (50K steps)
python train_hydra.py training.total_timesteps=50000

# Long training (100K steps)
python train_hydra.py training.total_timesteps=100000
```

### 6. Hyperparameter Tuning Examples

```bash
# DQN with different learning rates
python train_hydra.py agent.learning_rate=0.001
python train_hydra.py agent.learning_rate=0.0005
python train_hydra.py agent.learning_rate=0.0001

# DQN with different exploration
python train_hydra.py agent.epsilon_decay=1000000  # Faster decay
python train_hydra.py agent.epsilon_decay=10000000 # Slower decay

# SAC with different temperatures
python train_hydra.py scenario=sac_vs_sac agent.alpha=0.1
python train_hydra.py scenario=sac_vs_sac agent.alpha=0.2
python train_hydra.py scenario=sac_vs_sac agent.alpha=0.5

# PPO with different clip ranges
python train_hydra.py scenario=ppo_vs_random agent.clip_range=0.1
python train_hydra.py scenario=ppo_vs_random agent.clip_range=0.2
python train_hydra.py scenario=ppo_vs_random agent.clip_range=0.3
```

### 7. Network Architecture Examples

```bash
# Small networks (faster training)
python train_hydra.py agent.network.hidden_dims=[64,64]

# Medium networks (balanced)
python train_hydra.py agent.network.hidden_dims=[128,128]

# Large networks (better capacity)
python train_hydra.py agent.network.hidden_dims=[256,256,128]

# Deep networks
python train_hydra.py agent.network.hidden_dims=[512,256,128,64]
```

### 8. Logging and Monitoring Examples

```bash
# Enable WandB logging
python train_hydra.py wandb.mode=online wandb.project=my-drone-project

# Custom experiment name
python train_hydra.py experiment_name=dqn_baseline_v1

# Combine with hyperparameters
python train_hydra.py \
  scenario=sac_vs_sac \
  experiment_name=sac_lr_sweep_001 \
  agent.learning_rate=0.001 \
  wandb.mode=online

# Disable logging for quick tests
python train_hydra.py wandb.mode=disabled
```

## 📊 Evaluation Examples

### 1. Basic Evaluation

```bash
# Evaluate latest models with GUI
python evaluate_hydra.py

# Evaluate without GUI (headless)
python evaluate_trained_models.py

# Evaluate specific scenario
python evaluate_hydra.py scenario=sac_vs_sac
```

### 2. Evaluation Options

```bash
# More episodes for better statistics
python evaluate_trained_models.py --episodes 20

# Specific model directory
python evaluate_trained_models.py --weights-dir weights/pursuit_evade/20241201_143022

# Force CPU evaluation
python evaluate_trained_models.py --device cpu

# Quiet mode (less output)
python evaluate_trained_models.py --quiet
```

## 🎯 Algorithm Comparison Study

### Systematic Comparison Setup

```bash
# 1. Train baseline DQN
python train_hydra.py \
  scenario=pursuit_evasion \
  experiment_name=comparison_dqn \
  training.total_timesteps=50000 \
  wandb.mode=online

# 2. Train PPO
python train_hydra.py \
  scenario=ppo_vs_random \
  experiment_name=comparison_ppo \
  training.total_timesteps=50000 \
  wandb.mode=online

# 3. Train SAC
python train_hydra.py \
  scenario=sac_vs_random \
  experiment_name=comparison_sac \
  training.total_timesteps=50000 \
  wandb.mode=online

# 4. Evaluate all
python evaluate_trained_models.py --episodes 50
```

## 🔬 Hyperparameter Sweep Examples

### Learning Rate Sweep
```bash
#!/bin/bash
for lr in 0.001 0.0005 0.0001; do
  python train_hydra.py \
    agent.learning_rate=$lr \
    experiment_name=lr_sweep_$lr \
    training.total_timesteps=20000 \
    wandb.mode=online &
done
wait
```

### SAC Temperature Sweep
```bash
#!/bin/bash
for alpha in 0.1 0.2 0.5; do
  python train_hydra.py \
    scenario=sac_vs_sac \
    agent.alpha=$alpha \
    experiment_name=sac_alpha_$alpha \
    training.total_timesteps=20000 \
    wandb.mode=online &
done
wait
```

### Network Size Sweep
```bash
#!/bin/bash
declare -a sizes=("[64,64]" "[128,128]" "[256,256]")
for size in "${sizes[@]}"; do
  python train_hydra.py \
    agent.network.hidden_dims=$size \
    experiment_name=network_$size \
    training.total_timesteps=20000 \
    wandb.mode=online &
done
wait
```

## 🎮 Custom Scenario Examples

### Create Custom Training Scenarios

1. **Asymmetric Action Lengths**
```yaml
# conf/scenario/asymmetric.yaml
drones:
  - role: PURSUER
    agent_type: sac
    action_length: 10.0  # Fast pursuer
    is_training: true
  - role: EVADER
    agent_type: dqn
    action_length: 5.0   # Slower evader
    is_training: true
```

2. **Pre-trained vs Fresh**
```yaml
# conf/scenario/pretrained_vs_fresh.yaml
drones:
  - role: PURSUER
    agent_type: dqn
    resume_from: "weights/pursuit_evade/20241201_143022/pursuer_final.pt"
    is_training: false  # Frozen pre-trained
  - role: EVADER
    agent_type: sac
    is_training: true   # Fresh training
```

3. **Multi-algorithm Comparison**
```yaml
# conf/scenario/algorithm_battle.yaml
drones:
  - role: PURSUER
    agent_type: ppo
    is_training: true
  - role: EVADER
    agent_type: sac
    is_training: true
```

### Train Custom Scenarios
```bash
# Train asymmetric scenario
python train_hydra.py scenario=asymmetric

# Train pre-trained vs fresh
python train_hydra.py scenario=pretrained_vs_fresh

# Train algorithm battle
python train_hydra.py scenario=algorithm_battle
```

## 📈 Performance Optimization

### GPU Training
```bash
# Ensure GPU usage
python train_hydra.py device=cuda

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Memory Optimization
```bash
# Reduce memory usage
python train_hydra.py \
  agent.batch_size=16 \
  agent.buffer_size=5000

# For low-memory systems
python train_hydra.py \
  agent.batch_size=8 \
  agent.buffer_size=1000 \
  agent.network.hidden_dims=[64,64]
```

### Parallel Training
```bash
# Run multiple experiments in parallel
python train_hydra.py experiment_name=exp1 &
python train_hydra.py experiment_name=exp2 &
python train_hydra.py experiment_name=exp3 &
wait
```

## 🐛 Troubleshooting Examples

### Common Issues and Solutions

1. **X11/GUI Issues**
```bash
# Disable rendering for headless systems
python train_hydra.py environment.render_simulation=false

# Set up virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

2. **Memory Issues**
```bash
# Reduce batch size
python train_hydra.py agent.batch_size=8

# Reduce buffer size
python train_hydra.py agent.buffer_size=1000
```

3. **Slow Training**
```bash
# Use GPU
python train_hydra.py device=cuda

# Reduce episode length
python train_hydra.py environment.max_episode_steps=200

# Increase logging interval
python train_hydra.py training.log_interval=1000
```

4. **Configuration Errors**
```bash
# Validate configuration
python train_hydra.py --cfg job

# Show available options
python train_hydra.py --help
```

## 📚 Advanced Examples

### Resume Training
```bash
# Resume from specific checkpoint
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/20241201_143022/pursuer_50000.pt"

# Continue training with different hyperparameters
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/20241201_143022/pursuer_50000.pt" \
  agent.learning_rate=0.0001
```

### Multi-Stage Training
```bash
# Stage 1: Basic training
python train_hydra.py \
  experiment_name=stage1_basic \
  training.total_timesteps=25000

# Stage 2: Fine-tuning with lower learning rate
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/latest/pursuer_final.pt" \
  agent.learning_rate=0.0001 \
  experiment_name=stage2_finetune \
  training.total_timesteps=25000
```

### Curriculum Learning
```bash
# Easy environment (larger action length)
python train_hydra.py \
  scenario.drones[0].action_length=10.0 \
  experiment_name=curriculum_easy \
  training.total_timesteps=20000

# Medium difficulty
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/latest/pursuer_final.pt" \
  scenario.drones[0].action_length=7.0 \
  experiment_name=curriculum_medium \
  training.total_timesteps=20000

# Hard environment (smaller action length)
python train_hydra.py \
  scenario.drones[0].resume_from="weights/pursuit_evade/latest/pursuer_final.pt" \
  scenario.drones[0].action_length=5.0 \
  experiment_name=curriculum_hard \
  training.total_timesteps=20000
```

This guide provides comprehensive examples for all aspects of training and evaluation in the drone pursuit-evasion environment. Start with the basic examples and gradually move to more advanced configurations as needed. 