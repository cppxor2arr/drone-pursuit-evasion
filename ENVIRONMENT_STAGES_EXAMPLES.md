# 🏗️ Environment Complexity Stages Guide

This guide explains the three-stage environment complexity system that allows progressive training from simple to complex scenarios.

## 🌟 Overview

The environment stages provide a curriculum learning approach:

1. **Stage 1 (Open)**: Pure pursuit-evasion dynamics without obstacles
2. **Stage 2 (Single)**: Basic obstacle avoidance with one central cylinder
3. **Stage 3 (Multiple)**: Complex navigation through a grid of obstacles

## 📊 Stage Comparison

| Stage | Complexity | Obstacles | Use Case | Learning Focus |
|-------|------------|-----------|----------|----------------|
| **Open** | Easy | None | Algorithm development | Pure strategies |
| **Single** | Medium | 1 cylinder | Obstacle introduction | Basic avoidance |
| **Multiple** | Hard | Grid pattern | Full challenge | Complex navigation |

## 🚀 Usage Examples

### Method 1: Using Predefined Environment Configs

```bash
# Stage 1: Open Space Training
python train_hydra.py scenario=sac_vs_sac environment=open_stage

# Stage 2: Single Obstacle Training  
python train_hydra.py scenario=sac_vs_sac environment=single_obstacle

# Stage 3: Multiple Obstacles Training
python train_hydra.py scenario=sac_vs_sac environment=pursuit_evasion
```

### Method 2: Override Stage Parameter

```bash
# Override stage while keeping other environment settings
python train_hydra.py scenario=pursuit_evasion environment.stage=open
python train_hydra.py scenario=pursuit_evasion environment.stage=single  
python train_hydra.py scenario=pursuit_evasion environment.stage=multiple
```

### Method 3: Combined with Different Scenarios

```bash
# Test different algorithms across stages
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=open
python train_hydra.py scenario=dqn_pursuer_vs_hovering_evader environment.stage=single
python train_hydra.py scenario=sac_pursuer_vs_ppo_evader environment.stage=multiple
```

## 🎯 Curriculum Learning Strategy

### Progressive Training Approach

```bash
# Step 1: Master basics in open space
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=50000 experiment_name=sac_stage1

# Step 2: Add simple obstacle avoidance
python train_hydra.py scenario=sac_vs_sac environment.stage=single training.total_timesteps=50000 experiment_name=sac_stage2

# Step 3: Handle complex navigation
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple training.total_timesteps=100000 experiment_name=sac_stage3
```

### Transfer Learning Between Stages

```bash
# Train on Stage 1, then transfer to Stage 2
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=25000 experiment_name=transfer_base

# Resume from Stage 1 model for Stage 2 training
python train_hydra.py scenario=sac_vs_sac environment.stage=single training.resume_from=weights/transfer_base/pursuer_final.pt training.total_timesteps=25000 experiment_name=transfer_stage2
```

## 🔧 Configuration Details

### Environment Stage Parameters

Each stage can be configured in `conf/environment/`:

```yaml
# Stage Configuration
environment:
  stage: "open"    # Options: "open", "single", "multiple"
  
  # Other parameters remain the same
  lidar_reach: 4.0
  num_ray: 20
  # ... reward parameters
```

### Stage-Specific Characteristics

#### Stage 1: Open Space
- **Obstacles**: None
- **Focus**: Pure pursuit/evasion dynamics
- **Difficulty**: Low
- **Good for**: Algorithm development, baseline testing

#### Stage 2: Single Obstacle
- **Obstacles**: One cylinder at center (0, 0)
- **Focus**: Basic obstacle avoidance + pursuit/evasion
- **Difficulty**: Medium
- **Good for**: Teaching basic navigation skills

#### Stage 3: Multiple Obstacles  
- **Obstacles**: 6x6 grid of cylinders (2m spacing)
- **Focus**: Complex navigation + pursuit/evasion
- **Difficulty**: High
- **Good for**: Final evaluation, complex behavior emergence

## 📈 Performance Analysis

### Expected Learning Curves

**Stage 1 (Open)**: 
- Fast convergence
- High capture rates for pursuer
- Simple evasion patterns

**Stage 2 (Single)**:
- Moderate convergence
- Obstacle-aware behaviors
- Strategic positioning around obstacle

**Stage 3 (Multiple)**:
- Slower convergence
- Complex path planning
- Sophisticated hiding/chasing strategies

### Evaluation Across Stages

```bash
# Evaluate same model across all stages
python evaluate_hydra.py --model weights/sac_trained/pursuer_final.pt --stages open,single,multiple

# Compare performance degradation
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=10000 --eval-all-stages
```

## 🎮 Interactive Examples

### Quick Stage Comparison

```bash
# Quick training on all stages (for testing)
python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=5000 experiment_name=quick_open
python train_hydra.py scenario=sac_vs_sac environment.stage=single training.total_timesteps=5000 experiment_name=quick_single  
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple training.total_timesteps=5000 experiment_name=quick_multiple
```

### Visual Comparison (with rendering)

```bash
# Enable rendering to see the differences
python train_hydra.py scenario=sac_vs_sac environment.stage=open environment.render_simulation=true training.total_timesteps=1000
python train_hydra.py scenario=sac_vs_sac environment.stage=single environment.render_simulation=true training.total_timesteps=1000
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple environment.render_simulation=true training.total_timesteps=1000
```

## 📊 Monitoring and Analysis

### WandB Logging

Each stage automatically logs:
- Environment complexity level
- Collision rates (higher in complex stages)
- Episode lengths (typically longer in complex stages)
- Success rates per stage

### Key Metrics to Track

- **Capture Rate**: Success in reaching the target
- **Collision Rate**: Frequency of obstacle collisions
- **Episode Length**: Time to episode completion
- **Reward Progression**: Learning efficiency per stage

## 🔬 Research Applications

### Curriculum Learning Studies
```bash
# Compare curriculum vs direct training
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple training.total_timesteps=100000 experiment_name=direct_hard

# vs

python train_hydra.py scenario=sac_vs_sac environment.stage=open training.total_timesteps=30000 experiment_name=curriculum_1
python train_hydra.py scenario=sac_vs_sac environment.stage=single training.total_timesteps=30000 experiment_name=curriculum_2  
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple training.total_timesteps=40000 experiment_name=curriculum_3
```

### Algorithm Robustness Testing
```bash
# Test algorithm performance across complexity levels
for stage in open single multiple; do
  for algo in ppo dqn sac; do
    python train_hydra.py scenario=${algo}_vs_${algo} environment.stage=$stage experiment_name=${algo}_${stage}
  done
done
```

## 🛠️ Customization

### Creating Custom Stages

You can modify `task/lider_drone_base.py` to add new obstacle patterns:

```python
elif self.environment_stage == EnvironmentStage.CUSTOM:
    # Your custom obstacle configuration
    pass
```

### Adjusting Obstacle Parameters

Override obstacle properties:
```bash
# Modify through configuration (future feature)
python train_hydra.py scenario=sac_vs_sac environment.stage=single environment.obstacle_radius=1.0
```

---

**💡 Tip**: Start with the open stage for algorithm development, then progress through stages to build robust behaviors! 