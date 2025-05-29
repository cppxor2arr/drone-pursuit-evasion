# 🎓 Curriculum Learning Guide

## 🌟 Simple Stage Parameter Approach

**You don't need separate environment YAML files!** The only difference between environment stages is the `stage` parameter. Simply override it for curriculum learning.

**✅ Simplified**: We removed redundant `open_stage.yaml` and `single_obstacle.yaml` files. Just use `environment.stage=X` with any base environment configuration.

## 🚀 Three-Stage Curriculum

### Stage 1: Open Space (Easy)
```bash
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=open
```
- **Environment**: No obstacles
- **Focus**: Pure pursuit-evasion dynamics
- **Good for**: Algorithm development, baseline testing

### Stage 2: Single Obstacle (Medium)
```bash
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=single
```
- **Environment**: One central cylinder
- **Focus**: Basic obstacle avoidance + pursuit-evasion
- **Good for**: Teaching navigation skills

### Stage 3: Multiple Obstacles (Hard)
```bash
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=multiple
```
- **Environment**: Grid of cylinders
- **Focus**: Complex navigation + strategic planning
- **Good for**: Final evaluation, advanced behaviors

## 📚 Complete Curriculum Examples

### Example 1: DQN vs DQN Curriculum
```bash
# Stage 1: Master basics (25K steps)
python train_hydra.py scenario=pursuit_evasion environment.stage=open \
  training.total_timesteps=25000 experiment_name=dqn_curriculum_stage1

# Stage 2: Add complexity (25K steps)
python train_hydra.py scenario=pursuit_evasion environment.stage=single \
  training.total_timesteps=25000 experiment_name=dqn_curriculum_stage2

# Stage 3: Full challenge (50K steps)
python train_hydra.py scenario=pursuit_evasion environment.stage=multiple \
  training.total_timesteps=50000 experiment_name=dqn_curriculum_stage3
```

### Example 2: SAC vs SAC with Transfer Learning
```bash
# Stage 1: Build foundation
python train_hydra.py scenario=sac_vs_sac environment.stage=open \
  training.total_timesteps=30000 experiment_name=sac_base

# Stage 2: Transfer and adapt  
python train_hydra.py scenario=sac_vs_sac environment.stage=single \
  scenario.drones[0].resume_from=weights/sac_base/pursuer_final.pt \
  scenario.drones[1].resume_from=weights/sac_base/evader_final.pt \
  training.total_timesteps=30000 experiment_name=sac_transfer

# Stage 3: Final mastery
python train_hydra.py scenario=sac_vs_sac environment.stage=multiple \
  scenario.drones[0].resume_from=weights/sac_transfer/pursuer_final.pt \
  scenario.drones[1].resume_from=weights/sac_transfer/evader_final.pt \
  training.total_timesteps=40000 experiment_name=sac_final
```

### Example 3: Cross-Algorithm Curriculum
```bash
# PPO Pursuer vs DQN Evader across all stages
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=open training.total_timesteps=20000
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=single training.total_timesteps=20000
python train_hydra.py scenario=ppo_pursuer_vs_dqn_evader environment.stage=multiple training.total_timesteps=30000
```

## 🔄 Quick Stage Testing

Test any scenario across all stages quickly:
```bash
# Replace YOUR_SCENARIO with any available scenario
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=open training.total_timesteps=5000
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=single training.total_timesteps=5000
python train_hydra.py scenario=YOUR_SCENARIO environment.stage=multiple training.total_timesteps=5000
```

## 📊 Monitoring Progress

### Expected Learning Patterns

**Stage 1 (Open)**:
- Fast convergence
- High success rates
- Simple strategies emerge

**Stage 2 (Single)**:
- Moderate convergence  
- Obstacle-aware behaviors
- Strategic positioning around obstacle

**Stage 3 (Multiple)**:
- Slower convergence
- Complex path planning
- Sophisticated hiding/chasing strategies

### Key Metrics to Track
- **Capture Rate**: Success frequency
- **Episode Length**: Time to completion
- **Collision Rate**: Obstacle interactions
- **Reward Progression**: Learning efficiency

## 🎯 Best Practices

1. **Start Simple**: Always begin with `environment.stage=open`
2. **Progressive Steps**: Don't skip stages
3. **Transfer Learning**: Use `resume_from` for continuity
4. **Monitor Metrics**: Track performance across stages
5. **Adjust Timesteps**: Allocate more time to harder stages

## 💡 Advanced Tips

- Use different experiment names for each stage
- Save models at each stage for later analysis  
- Compare direct training vs curriculum learning
- Experiment with different timestep allocations
- Test robustness by evaluating stage 3 models on stage 1

---

**Remember**: The `environment.stage` parameter works with ANY scenario - just change the stage to switch complexity levels! 