# 🚁 Training Scenarios Quick Reference

## Environment Complexity Stages

### 🎯 Recommended: Use Stage Parameter Override

```bash
# 🌌 Stage 1: Open Space (Easy) - No obstacles
python train_hydra.py scenario=sac_vs_sac environment.stage=open

# 🏗️ Stage 2: Single Obstacle (Medium) - One cylinder in center
python train_hydra.py scenario=sac_vs_sac environment.stage=single

# 🌆 Stage 3: Multiple Obstacles (Hard) - Grid of cylinders  
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple
```

### Examples with Different Scenarios
```bash
# SAC vs SAC across all stages
python train_hydra.py scenario=sac_vs_sac environment.stage=open
python train_hydra.py scenario=sac_vs_sac environment.stage=single
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple

# Cross-algorithm training across stages
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=open
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=single
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=multiple
```

### Alternative: Predefined Environment Configs
```bash
# These also work, but stage override is more flexible
python train_hydra.py scenario=sac_vs_sac environment=open_stage
python train_hydra.py scenario=sac_vs_sac environment=single_obstacle
python train_hydra.py scenario=sac_vs_sac environment=pursuit_evasion
```

---

## Basic Algorithm Testing

### Pursuer vs Static/Simple Targets
```bash
# PPO Pursuer Testing
python train_hydra.py scenario=ppo_pursuer_vs_hovering_evader  # vs Stationary
python train_hydra.py scenario=ppo_pursuer_vs_random_evader   # vs Random

# DQN Pursuer Testing  
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader # vs Stationary
python train_hydra.py scenario=dqn_pursuer_vs_random_evader  # vs Random

# SAC Pursuer Testing
python train_hydra.py scenario=sac_pursuer_vs_hovering_evader # vs Stationary
python train_hydra.py scenario=sac_vs_random                 # vs Random (existing)
```

## Cross-Algorithm Competition

### Pursuers vs Intelligent Evaders
```bash
# PPO Pursuer vs Learning Evaders
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader

# DQN Pursuer vs Learning Evaders  
python train_hydra.py scenario=dqn_pursuer_vs_ppo_evader

# SAC Pursuer vs Learning Evaders
python train_hydra.py scenario=sac_pursuer_vs_dqn_evader
python train_hydra.py scenario=sac_pursuer_vs_ppo_evader
```

## Same Algorithm Training

### Symmetric Training (Both Agents Learning)
```bash
# DQN vs DQN (Classic)
python train_hydra.py scenario=pursuit_evasion

# PPO vs PPO 
python train_hydra.py scenario=ppo_vs_random  # Note: modify config for PPO vs PPO

# SAC vs SAC (Advanced)
python train_hydra.py scenario=sac_vs_sac
```

## Mixed Algorithm Scenarios

### Existing Mixed Scenarios
```bash
# PPO vs DQN (from original)
python train_hydra.py scenario=mixed_agents

# PPO vs Random (from original)
python train_hydra.py scenario=ppo_vs_random

# SAC vs Random (from original)  
python train_hydra.py scenario=sac_vs_random
```

## Complete Scenario Matrix

| Pursuer → <br> Evader ↓ | PPO | DQN | SAC |
|-------------------------|-----|-----|-----|
| **Hovering** | `ppo_pursuer_vs_hovering_evader` | `dqn_pursuer_vs_hovering_evader` | `sac_pursuer_vs_hovering_evader` |
| **Random** | `ppo_pursuer_vs_random_evader` | `dqn_pursuer_vs_random_evader` | `sac_vs_random` |
| **PPO** | `ppo_vs_random`* | `dqn_pursuer_vs_ppo_evader` | `sac_pursuer_vs_ppo_evader` |
| **DQN** | `ppo_pursuer_vs_dqn_evader` | `pursuit_evasion` | `sac_pursuer_vs_dqn_evader` |
| **SAC** | N/A | N/A | `sac_vs_sac` |

*Note: `ppo_vs_random` needs config modification for PPO vs PPO

## Quick Training Commands

### Short Tests (1K steps)
```bash
python train_hydra.py scenario=SCENARIO_NAME training.total_timesteps=1000 wandb.mode=disabled
```

### Medium Training (10K steps)
```bash
python train_hydra.py scenario=SCENARIO_NAME training.total_timesteps=10000 wandb.mode=online
```

### Long Training (50K steps)
```bash
python train_hydra.py scenario=SCENARIO_NAME training.total_timesteps=50000 wandb.mode=online
```

## Evaluation Commands

### Evaluate Latest Models
```bash
# With GUI (requires X11)
python evaluate_hydra.py scenario=SCENARIO_NAME

# Without GUI (headless)
python evaluate_trained_models.py --episodes 10
```

### Evaluate Specific Models
```bash
python evaluate_trained_models.py \
  --weights-dir weights/pursuit_evade/20241201_143022 \
  --episodes 20
```

## Custom Training Examples

### Algorithm Comparison Study
```bash
# Train all pursuers against hovering evader
python train_hydra.py scenario=ppo_pursuer_vs_hovering_evader experiment_name=ppo_baseline
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader experiment_name=dqn_baseline  
python train_hydra.py scenario=sac_pursuer_vs_hovering_evader experiment_name=sac_baseline

# Compare results
python evaluate_trained_models.py --episodes 50
```

### Cross-Algorithm Tournament
```bash
# Round 1: PPO vs others
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader experiment_name=ppo_vs_dqn

# Round 2: SAC vs others  
python train_hydra.py scenario=sac_pursuer_vs_dqn_evader experiment_name=sac_vs_dqn
python train_hydra.py scenario=sac_pursuer_vs_ppo_evader experiment_name=sac_vs_ppo

# Round 3: DQN vs others
python train_hydra.py scenario=dqn_pursuer_vs_ppo_evader experiment_name=dqn_vs_ppo
``` 