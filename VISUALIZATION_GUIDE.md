# 🎬 Visualization Guide

## Overview

The `visualize_hydra.py` script allows you to visualize trained drone pursuit-evasion scenarios with 3D simulation. Unlike evaluation scripts, this focuses on visual demonstration of behaviors rather than performance metrics.

## 🚀 Quick Start

### **🐌 Slow Motion Visualization (Recommended for Watching)**
```bash
# Slow motion with pauses (great for watching behavior)
python visualize_hydra.py experiment_name=test_compatible_weights

# Custom slow motion settings
python visualize_hydra.py experiment_name=test_compatible_weights \
  visualization.sleep_time=0.1 \
  visualization.episodes=2 \
  visualization.max_episode_steps=50
```

### **⚡ Fast Visualization (For Quick Testing)**
```bash
# Fast visualization without pauses
python visualize_hydra.py experiment_name=test_compatible_weights \
  visualization.slow_motion=false \
  visualization.pause_between_episodes=false \
  visualization.episodes=5
```

### **🎯 Specific Scenarios**
```bash
# Open stage (no obstacles)
python visualize_hydra.py experiment_name=test_compatible_weights \
  environment.stage=open \
  visualization.episodes=3

# Single obstacle
python visualize_hydra.py experiment_name=test_compatible_weights \
  environment.stage=single \
  visualization.episodes=3
```

## ⚙️ Visualization Settings

### Speed Control
- `visualization.slow_motion=true/false` - Enable/disable slow motion
- `visualization.sleep_time=0.05` - Seconds between steps (0.01=fast, 0.1=slow)
- `visualization.pause_between_episodes=true/false` - Wait for Enter between episodes

### Episode Control  
- `visualization.episodes=3` - Number of episodes to run
- `visualization.max_episode_steps=100` - Maximum steps per episode

### Rendering Control
- `environment.render_simulation=true/false` - Enable/disable 3D GUI

## 🎮 Interactive Controls

When `pause_between_episodes=true`:
- **Press Enter** to continue to next episode
- **Press Ctrl+C** to stop visualization

## 📊 Visual Output

During visualization you'll see:
- 🎯 **Capture events** when pursuer catches evader  
- 💥 **Collision events** when drones hit obstacles
- 🚫 **Out of bounds** when drones leave the area
- 📊 **Episode summaries** with rewards and outcomes

## 🔧 Troubleshooting

### No GUI Appearing
```bash
# Check display connection
echo $DISPLAY

# Test PyBullet GUI
python -c "import pybullet as p; p.connect(p.GUI); input('Press Enter to close')"
```

### Too Fast to See
```bash
# Use slower settings
python visualize_hydra.py experiment_name=YOUR_EXPERIMENT \
  visualization.slow_motion=true \
  visualization.sleep_time=0.15 \
  visualization.max_episode_steps=30
```

### No Trained Models
```bash
# Train a quick model first
python train_hydra.py scenario=pursuit_evasion \
  training.total_timesteps=2000 \
  experiment_name=quick_demo

# Then visualize
python visualize_hydra.py experiment_name=quick_demo
```

## 🎯 Configuration Options

### Core Visualization Settings
```bash
# Number of episodes to visualize
python visualize_hydra.py visualization.episodes=10

# Episode length limit
python visualize_hydra.py visualization.max_episode_steps=1000

# Slow motion for better observation
python visualize_hydra.py visualization.slow_motion=true

# Pause between episodes for manual control
python visualize_hydra.py visualization.pause_between_episodes=true
```

### Environment Stages
```bash
# Stage 1: Open space (no obstacles)
python visualize_hydra.py environment.stage=open

# Stage 2: Single obstacle
python visualize_hydra.py environment.stage=single

# Stage 3: Multiple obstacles (default)
python visualize_hydra.py environment.stage=multiple
```

### Scenario Selection
```bash
# Algorithm comparison scenarios
python visualize_hydra.py scenario=pursuit_evasion        # DQN vs DQN
python visualize_hydra.py scenario=ppo_pursuer_vs_dqn_evader
python visualize_hydra.py scenario=sac_pursuer_vs_ppo_evader

# Testing against baselines
python visualize_hydra.py scenario=ppo_pursuer_vs_hovering_evader
python visualize_hydra.py scenario=sac_pursuer_vs_random_evader
```

## 📁 Model Loading

### Auto-Detection (Recommended)
The script automatically looks for `pursuer_final.pt` and `evader_final.pt` in the specified weights directory:

```bash
# Auto-detect from latest training
python visualize_hydra.py visualization.weights_dir=weights/pursuit_evade/latest

# Auto-detect from specific experiment
python visualize_hydra.py visualization.weights_dir=weights/pursuit_evade/20241201_143022
```

### Manual Model Paths
Edit `conf/visualization.yaml` to specify exact paths:

```yaml
visualization:
  model_paths:
    uav_0: "path/to/my_pursuer_model.pt"
    uav_1: "path/to/my_evader_model.pt"
```

## 🎮 Usage Examples

### Compare Algorithm Performance
```bash
# Visualize DQN vs DQN baseline
python visualize_hydra.py scenario=pursuit_evasion \
  visualization.weights_dir=weights/dqn_experiment \
  visualization.episodes=3

# Visualize SAC vs SAC advanced
python visualize_hydra.py scenario=sac_vs_sac \
  visualization.weights_dir=weights/sac_experiment \
  visualization.episodes=3

# Compare cross-algorithm
python visualize_hydra.py scenario=ppo_pursuer_vs_dqn_evader \
  visualization.weights_dir=weights/mixed_experiment \
  visualization.episodes=3
```

### Curriculum Learning Demonstration
```bash
# Show progression through complexity stages
python visualize_hydra.py scenario=sac_vs_sac environment.stage=open \
  visualization.weights_dir=weights/curriculum_stage1 \
  visualization.episodes=2

python visualize_hydra.py scenario=sac_vs_sac environment.stage=single \
  visualization.weights_dir=weights/curriculum_stage2 \
  visualization.episodes=2

python visualize_hydra.py scenario=sac_vs_sac environment.stage=multiple \
  visualization.weights_dir=weights/curriculum_stage3 \
  visualization.episodes=2
```

### Interactive Visualization
```bash
# Pause between episodes for analysis
python visualize_hydra.py visualization.pause_between_episodes=true \
  visualization.episodes=5

# Slow motion for detailed observation
python visualize_hydra.py visualization.slow_motion=true \
  visualization.episodes=3

# Long episodes to see full behaviors
python visualize_hydra.py visualization.max_episode_steps=1000 \
  visualization.episodes=2
```

## 🛠 Advanced Configuration

### Custom Visualization Config
Create a custom config file `conf/my_visualization.yaml`:

```yaml
defaults:
  - agent: sac
  - environment: pursuit_evasion
  - scenario: sac_vs_sac
  - _self_

visualization:
  name: "My Custom Demo"
  episodes: 10
  max_episode_steps: 800
  slow_motion: true
  pause_between_episodes: false
  weights_dir: "weights/my_best_experiment/20241201_143022"

environment:
  render_simulation: true
  stage: "multiple"

device: "cuda"
seed: 42
```

Use it with:
```bash
python visualize_hydra.py --config-name=my_visualization
```

### Override Multiple Parameters
```bash
python visualize_hydra.py \
  scenario=pursuit_evasion \
  environment.stage=single \
  visualization.episodes=5 \
  visualization.slow_motion=true \
  visualization.weights_dir=weights/my_experiment \
  device=cuda
```

## 🎥 Output and Controls

### What You'll See
- **3D Drone Environment**: PyBullet physics simulation
- **LiDAR Rays**: Green lines showing sensor rays
- **Obstacles**: Cylinders based on environment stage
- **Flight Dome**: Transparent sphere showing boundaries
- **Real-time Physics**: Realistic drone dynamics

### Console Output
```
🚁 Drone Pursuit-Evasion Visualization
==================================================
🎬 Starting Visualization: Drone Pursuit-Evasion Demo
Environment Stage: multiple
Episodes to run: 5
Episode length limit: 500
Device: cuda

🤖 Loading pretrained agents...
  ✅ Loaded SAC PURSUER from: weights/latest/pursuer_final.pt
  ✅ Loaded SAC EVADER from: weights/latest/evader_final.pt

📺 Episode 1/5
  📊 Episode 1 Results:
     Length: 127 steps
     Outcome: capture
     uav_0 (PURSUER): 15.32
     uav_1 (EVADER): -8.45
```

### Episode Outcomes
- **Capture**: Pursuer successfully catches evader
- **Collision**: Agent hits obstacle
- **Out of bounds**: Agent exits flight dome
- **Timeout**: Episode reaches step limit

## 🐛 Troubleshooting

### Common Issues

**No GUI Display (Headless System)**:
```bash
# Set up virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
python visualize_hydra.py
```

**Model Loading Errors**:
```bash
# Check if model files exist
ls -la weights/pursuit_evade/latest/

# Use absolute paths if needed
python visualize_hydra.py visualization.weights_dir=/full/path/to/weights
```

**Performance Issues**:
```bash
# Use CPU if GPU memory is low
python visualize_hydra.py device=cpu

# Reduce episode count
python visualize_hydra.py visualization.episodes=2
```

**Configuration Validation**:
```bash
# Check configuration without running
python visualize_hydra.py --cfg job
```

## 💡 Tips

1. **Start Simple**: Begin with 2-3 episodes to test setup
2. **Use Slow Motion**: Enable for detailed behavior analysis
3. **Try Different Stages**: Compare complexity levels
4. **Interactive Mode**: Use pause between episodes for screenshots
5. **Model Comparison**: Visualize different training checkpoints

## 🔗 Related Scripts

- **`train_hydra.py`**: Train new models for visualization
- **`evaluate_hydra.py`**: Evaluate model performance (no GUI)
- **`evaluate_trained_models.py`**: Headless evaluation with metrics

---

**Ready to visualize? Start with:**
```bash
python visualize_hydra.py scenario=sac_vs_sac visualization.episodes=3
``` 