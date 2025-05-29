# Legacy Files

This folder contains obsolete files that have been replaced by the new modular system.

## Obsolete Scripts

### `train_dqn_pursuit_evasion.py` (OBSOLETE)
- **Replaced by**: `train_hydra.py`
- **Issues**: 
  - Hardcoded hyperparameters
  - Only supports DQN algorithm
  - Monolithic structure
  - No unified agent API

### `evaluate_dqn_pursuit_evasion.py` (OBSOLETE)
- **Replaced by**: `evaluate_hydra.py` and `evaluate_trained_models.py`
- **Issues**:
  - Only works with DQN models
  - Limited evaluation metrics
  - No cross-algorithm support

## Use the New System Instead

**✅ For Training:**
```bash
python train_hydra.py scenario=pursuit_evasion  # DQN vs DQN (like old script)
python train_hydra.py scenario=sac_vs_sac       # SAC vs SAC (new)
python train_hydra.py scenario=ppo_pursuer_vs_random_evader  # PPO vs Random (new)
```

**✅ For Evaluation:**
```bash
python evaluate_hydra.py              # With GUI
python evaluate_trained_models.py     # Headless evaluation
```

## Why These Files Are Obsolete

1. **Limited Algorithm Support**: Only DQN, no PPO or SAC
2. **Hardcoded Configuration**: No flexibility for different scenarios
3. **Poor Code Structure**: Monolithic functions, hard to maintain
4. **No Cross-Algorithm Support**: Can't compare different algorithms
5. **Limited Evaluation**: Basic metrics only

The new system provides:
- ✅ Multiple algorithms (DQN, PPO, SAC)
- ✅ Flexible Hydra configuration
- ✅ Modular, maintainable code
- ✅ Cross-algorithm evaluation
- ✅ Comprehensive metrics and logging

**Recommendation**: Do not use these files. Use the new training system instead. 