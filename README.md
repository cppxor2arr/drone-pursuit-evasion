# 🚁 Drone Pursuit-Evasion Multi-Agent RL System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive multi-agent reinforcement learning environment for drone pursuit-evasion scenarios using PyBullet simulation. Features multiple RL algorithms (DQN, PPO, SAC), modular training system, and flexible configuration through Hydra.

![Drone Pursuit-Evasion](https://img.shields.io/badge/Environment-3D_Simulation-green.svg)
![Algorithms](https://img.shields.io/badge/Algorithms-DQN_|_PPO_|_SAC-blue.svg)

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd drone-pursuit-evasion

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pybullet; print('PyBullet installed successfully')"
```

### Basic Training
```bash
# Train DQN vs DQN (default)
python train_hydra.py

# Train SAC vs SAC (advanced)
python train_hydra.py scenario=sac_vs_sac

# Train PPO vs Random (quick test)
python train_hydra.py scenario=ppo_pursuer_vs_random_evader training.total_timesteps=10000
```

### Environment Complexity Stages
```bash
# Stage 1: Open space (no obstacles) - Easy
python train_hydra.py scenario=sac_vs_sac environment.stage=open

# Stage 2: Single obstacle - Medium
python train_hydra.py scenario=sac_vs_sac environment.stage=single

# Stage 3: Multiple obstacles - Hard (default)
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple
```

### Evaluation
```bash
# Evaluate latest models with GUI
python evaluate_hydra.py

# Evaluate without GUI (headless)
python evaluate_trained_models.py --episodes 10
```

### Visualization
```bash
# Visualize pretrained models with 3D simulation
python visualize_hydra.py

# Visualize specific scenario and stage
python visualize_hydra.py scenario=sac_vs_sac environment.stage=single

# Load specific model weights
python visualize_hydra.py visualization.weights_dir=weights/my_experiment/20241201_143022
```

## 🎯 Features

### **Multi-Algorithm Support**
- **DQN**: Deep Q-Network with experience replay
- **PPO**: Proximal Policy Optimization with GAE
- **SAC**: Soft Actor-Critic with twin critics
- **Special Agents**: Random and Hovering agents for baselines

### **Environment Complexity Stages**
- **Stage 1 (Open)**: Pure pursuit-evasion in open space - ideal for algorithm development
- **Stage 2 (Single)**: Basic obstacle avoidance with one central cylinder
- **Stage 3 (Multiple)**: Complex navigation through grid of obstacles
- **Curriculum Learning**: Progressive training from simple to complex scenarios

### **Modular Training System**
- **Refactored Architecture**: Clean, maintainable code structure
- **Config-Based Environment**: Pass configuration objects instead of individual parameters
- **Unified Agent API**: Consistent interface across all algorithms
- **Flexible Scenarios**: Easy configuration of agent combinations
- **Comprehensive Logging**: WandB integration with detailed metrics

### **Rich Environment**
- **3D Physics Simulation**: PyBullet-based realistic drone dynamics
- **LiDAR Sensors**: Configurable ray-casting for obstacle detection
- **Dynamic Scenarios**: Multiple starting positions and configurations
- **Reward Engineering**: Sophisticated reward functions for both roles

## 📊 Available Training Scenarios

### **Algorithm Comparison Matrix**

| Pursuer → <br> Evader ↓ | PPO | DQN | SAC |
|-------------------------|-----|-----|-----|
| **Hovering** | `ppo_pursuer_vs_hovering_evader` | `dqn_pursuer_vs_hovering_evader` | `sac_pursuer_vs_hovering_evader` |
| **Random** | `ppo_pursuer_vs_random_evader` | `dqn_pursuer_vs_random_evader` | `sac_vs_random` |
| **PPO** | *Configure manually* | `dqn_pursuer_vs_ppo_evader` | `sac_pursuer_vs_ppo_evader` |
| **DQN** | `ppo_pursuer_vs_dqn_evader` | `pursuit_evasion` | `sac_pursuer_vs_dqn_evader` |
| **SAC** | *Create new scenario* | *Create new scenario* | `sac_vs_sac` |

### **Quick Training Commands**
```bash
# Basic Algorithm Testing
python train_hydra.py scenario=ppo_pursuer_vs_hovering_evader
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader  
python train_hydra.py scenario=sac_pursuer_vs_hovering_evader

# Cross-Algorithm Competition
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader
python train_hydra.py scenario=sac_pursuer_vs_ppo_evader

# Symmetric Training (Both agents learning)
python train_hydra.py scenario=pursuit_evasion  # DQN vs DQN
python train_hydra.py scenario=sac_vs_sac       # SAC vs SAC
```

## 🔧 Configuration System

### **Hydra-Based Configuration**
```bash
# Override hyperparameters
python train_hydra.py agent.learning_rate=0.001 agent.gamma=0.95

# Change training duration  
python train_hydra.py training.total_timesteps=50000

# Enable/disable logging
python train_hydra.py wandb.mode=online
python train_hydra.py wandb.mode=disabled
```

### **Configuration Structure**
```
conf/
├── config.yaml              # Main configuration
├── agent/
│   ├── dqn.yaml             # DQN hyperparameters
│   ├── ppo.yaml             # PPO hyperparameters
│   └── sac.yaml             # SAC hyperparameters
└── scenario/
    ├── pursuit_evasion.yaml      # DQN vs DQN
    ├── sac_vs_sac.yaml          # SAC vs SAC
    ├── ppo_pursuer_vs_dqn_evader.yaml
    └── ... (many more scenarios)
```

## 📈 Performance Monitoring

### **WandB Integration**
- **Real-time Metrics**: Training loss, episode rewards, capture rates
- **Detailed Evaluation**: Episode tables, outcome statistics
- **Hyperparameter Tracking**: Automatic configuration logging
- **Model Checkpointing**: Best and periodic model saving

### **Evaluation Metrics**
- **Success Rates**: Capture, collision, timeout, out-of-bounds
- **Episode Statistics**: Length, rewards, outcome timing
- **Agent Performance**: Individual and comparative analysis

## 📚 Documentation

### **Current Documentation (Latest)**
- **[📖 README.md](README.md)** - This main overview and quick start guide
- **[📖 Complete Training Guide](README_TRAINING.md)** - Comprehensive training documentation
- **[🎯 Training Examples](TRAINING_EXAMPLES.md)** - Practical examples and use cases  
- **[⚡ Quick Reference](SCENARIO_QUICK_REFERENCE.md)** - Fast scenario lookup
- **[🎓 Curriculum Learning Guide](CURRICULUM_LEARNING_GUIDE.md)** - Progressive complexity training (RECOMMENDED)
- **[🏗️ Environment Stages Guide](ENVIRONMENT_STAGES_EXAMPLES.md)** - Detailed stage explanations

### **Legacy Files**
- **[🗂️ legacy/](legacy/)** - Contains obsolete scripts (do not use)
  - Old training scripts replaced by modular system
  - See `legacy/README.md` for migration guide

## 🛠 System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended (optional)
- **Memory**: 4GB+ RAM recommended
- **Display**: X11 for GUI evaluation (optional)

## 🎮 Usage Examples

### **Algorithm Comparison Study**
```bash
# Train all algorithms against static target
python train_hydra.py scenario=ppo_pursuer_vs_hovering_evader experiment_name=ppo_baseline
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader experiment_name=dqn_baseline  
python train_hydra.py scenario=sac_pursuer_vs_hovering_evader experiment_name=sac_baseline

# Evaluate and compare
python evaluate_trained_models.py --episodes 50
```

### **Hyperparameter Sweeps**
```bash
# Learning rate comparison
for lr in 0.001 0.0005 0.0001; do
  python train_hydra.py agent.learning_rate=$lr experiment_name=lr_sweep_$lr &
done
```

### **Custom Scenarios**
Create new scenarios in `conf/scenario/` for specific research needs:
```yaml
# conf/scenario/my_custom_scenario.yaml
drones:
  - role: PURSUER
    agent_type: sac
    start_pos: [2.0, 2.0, 1.0]
    action_length: 10.0
    is_training: true
  - role: EVADER
    agent_type: ppo
    start_pos: [-2.0, -2.0, 1.0]
    action_length: 5.0
    is_training: true
```

## 🐛 Troubleshooting

### **Common Issues**
```bash
# X11/Display issues (headless systems)
python train_hydra.py environment.render_simulation=false

# Memory issues
python train_hydra.py agent.batch_size=16 agent.buffer_size=5000

# CUDA issues
python train_hydra.py device=cpu

# Configuration validation
python train_hydra.py --cfg job
```

## 🤝 Contributing

1. **Algorithm Extensions**: Add new RL algorithms in `RL/` directory
2. **Environment Features**: Extend `task/lider_drone_base.py`
3. **Scenario Creation**: Add new scenarios in `conf/scenario/`
4. **Documentation**: Update guides and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyBullet**: Physics simulation engine
- **PyFlyt**: Drone simulation framework  
- **Hydra**: Configuration management
- **WandB**: Experiment tracking and visualization

---

**🚀 Ready to start? Check out the [Training Guide](README_TRAINING.md) for detailed instructions!**

## 🎓 Curriculum Learning with Environment Stages

### **Progressive Training Strategy**

Instead of using separate environment config files, simply override the `stage` parameter for curriculum learning:

```bash
# 🌌 Stage 1: Start with open space (easiest)
python train_hydra.py scenario=pursuit_evasion environment.stage=open training.total_timesteps=25000 experiment_name=curriculum_stage1

# 🏗️ Stage 2: Add single obstacle (medium difficulty)  
python train_hydra.py scenario=pursuit_evasion environment.stage=single training.total_timesteps=25000 experiment_name=curriculum_stage2

# 🌆 Stage 3: Full complexity with multiple obstacles (hardest)
python train_hydra.py scenario=pursuit_evasion environment.stage=multiple training.total_timesteps=50000 experiment_name=curriculum_stage3
```

### **Stage Parameter Options**
- `environment.stage=open` - No obstacles (pure pursuit-evasion)
- `environment.stage=single` - One central cylinder obstacle
- `environment.stage=multiple` - Grid of obstacles (default)

### **Transfer Learning Between Stages**
```bash
# Train base model on open stage
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=30000 experiment_name=base_model

# Transfer to single obstacle stage (resume from base model)
python train_hydra.py scenario=sac_vs_sac environment.stage=single \
  scenario.drones[0].resume_from=weights/base_model/pursuer_final.pt \
  scenario.drones[1].resume_from=weights/base_model/evader_final.pt \
  training.total_timesteps=30000 experiment_name=transferred_model

# Final training on complex stage
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple \
  scenario.drones[0].resume_from=weights/transferred_model/pursuer_final.pt \
  scenario.drones[1].resume_from=weights/transferred_model/evader_final.pt \
  training.total_timesteps=40000 experiment_name=final_model
```

### **Quick Stage Testing**
```bash
# Test any scenario across all stages quickly
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=5000
python train_hydra.py scenario=sac_vs_sac environment.stage=single training.total_timesteps=5000  
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple training.total_timesteps=5000
```

**💡 Tip**: The stage parameter works with any scenario - just change `environment.stage=X` to switch complexity levels!

## ✅ Latest Updates (2024)

### 🛠️ Major Improvements Completed
- **✅ Fixed PPO Training**: Resolved buffer interface issues, now fully functional
- **✅ Fixed Environment Errors**: Eliminated agent key errors and tensor/list type mismatches  
- **✅ Added Visualization Script**: New `visualize_hydra.py` for 3D simulation demos
- **✅ Simplified Configuration**: Removed redundant YAML files, use `environment.stage=X` instead
- **✅ Enhanced Documentation**: Comprehensive guides for all features and use cases

### 🚀 Ready-to-Use Features
- **Multi-Algorithm Training**: PPO, DQN, SAC all working correctly
- **Environment Complexity Stages**: Easy parameter switching between complexity levels
- **Visualization & Evaluation**: Professional tools for model demonstration
- **Curriculum Learning**: Progressive training from simple to complex environments
- **Cross-Algorithm Competition**: Train any algorithm against any other

---
