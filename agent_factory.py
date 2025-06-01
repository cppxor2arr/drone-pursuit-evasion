#!/usr/bin/env python3
"""
Unified agent factory for creating agents from scenario configurations.
This module provides a single source of truth for agent creation logic.
Uses explicit configuration loading - no defaults, errors when configs are missing.
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from task.lider_drone_base import AgentType
from RL import DroneDQNAgent, DronePPOAgent, DroneSACAgent, RandomAgent, HoveringAgent


def load_agent_config(agent_type: AgentType) -> DictConfig:
    """
    Load agent configuration from dedicated config files.
    NO DEFAULTS - raises explicit errors if config is missing or incomplete.
    
    Args:
        agent_type: Type of agent to load config for
        
    Returns:
        Agent configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        AttributeError: If required config parameters are missing
    """
    
    # Get the directory where this script is located (not cwd which Hydra changes)
    script_dir = Path(__file__).parent
    
    if agent_type == AgentType.DQN:
        config_file = script_dir / "conf" / "agent" / "dqn.yaml"
    elif agent_type == AgentType.PPO:
        config_file = script_dir / "conf" / "agent" / "ppo.yaml"  
    elif agent_type == AgentType.SAC:
        config_file = script_dir / "conf" / "agent" / "sac.yaml"
    elif agent_type in [AgentType.RANDOM, AgentType.HOVERING]:
        # These agents don't need configuration
        return OmegaConf.create({})
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Check if config file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_file}")
    
    # Load config directly from YAML file
    try:
        agent_config = OmegaConf.load(config_file)
        
        # Extract agent config based on structure
        if agent_type == AgentType.DQN:
            # DQN config has "agent:" section due to @package _global_
            if 'agent' not in agent_config:
                raise AttributeError(f"No 'agent' section found in {config_file}")
            final_config = agent_config.agent
        else:
            # PPO and SAC configs are at root level
            final_config = agent_config
        
        # Remove _target_ field if present (not needed for our use)
        if '_target_' in final_config:
            del final_config['_target_']
        
        # Validate required parameters exist
        validate_agent_config(final_config, agent_type)
        
        print(f"📋 Loaded {agent_type.value} config from {config_file}")
        return final_config
            
    except Exception as e:
        raise RuntimeError(f"Failed to load {agent_type.value} config from {config_file}: {e}")


def validate_agent_config(config: DictConfig, agent_type: AgentType):
    """
    Validate that all required configuration parameters are present.
    Raises explicit errors for missing parameters.
    
    Args:
        config: Agent configuration to validate
        agent_type: Type of agent
        
    Raises:
        AttributeError: If required parameters are missing
    """
    
    if agent_type == AgentType.DQN:
        required_params = [
            'gamma', 'learning_rate', 'epsilon_start', 'epsilon_end', 
            'epsilon_decay', 'batch_size', 'buffer_size', 
            'target_update_interval', 'soft_update_tau', 'network'
        ]
    elif agent_type == AgentType.PPO:
        required_params = [
            'gamma', 'learning_rate', 'clip_range', 'value_coef',
            'entropy_coef', 'gae_lambda', 'max_grad_norm', 'ppo_epochs',
            'batch_size', 'buffer_size', 'network'
        ]
    elif agent_type == AgentType.SAC:
        required_params = [
            'gamma', 'learning_rate', 'tau', 'alpha',
            'automatic_entropy_tuning', 'target_update_interval',
            'batch_size', 'buffer_size', 'warmup_steps', 'network'
        ]
    else:
        return  # Random and Hovering don't need validation
    
    missing_params = []
    for param in required_params:
        if param not in config:
            missing_params.append(param)
    
    if missing_params:
        raise AttributeError(
            f"Missing required {agent_type.value} config parameters: {missing_params}. "
            f"Please check conf/agent/{agent_type.value}.yaml"
        )
    
    # Validate network structure
    if 'network' in required_params:
        if 'hidden_dims' not in config.network:
            raise AttributeError(
                f"Missing 'hidden_dims' in network config for {agent_type.value}. "
                f"Please check conf/agent/{agent_type.value}.yaml"
            )


def create_agent(agent_type: str, config: DictConfig, obs_space, act_space, device: str, for_visualization: bool = False):
    """
    Create agent using explicit configuration from dedicated config files.
    NO DEFAULTS - all configuration must be explicitly provided.
    
    Args:
        agent_type: Agent type (string or AgentType enum)
        config: Full configuration (not used - we load from dedicated files)
        obs_space: Observation space
        act_space: Action space  
        device: Device for agent
        for_visualization: If True, disable exploration for DQN agents
    
    Returns:
        Created agent instance
        
    Raises:
        ValueError: If agent type is unknown
        FileNotFoundError: If agent config file is missing
        AttributeError: If required config parameters are missing
    """
    # Convert string to enum if needed
    if isinstance(agent_type, str):
        try:
            agent_type = AgentType(agent_type)
        except ValueError:
            raise ValueError(f"Invalid agent type: {agent_type}. Valid types: {[t.value for t in AgentType]}")
    
    # Load explicit configuration from dedicated files
    agent_config = load_agent_config(agent_type)
    
    if agent_type == AgentType.DQN:
        # For visualization, disable exploration (explicit override)
        if for_visualization:
            print(f"🎭 Disabling exploration for DQN visualization")
            agent_config = agent_config.copy()
            agent_config.epsilon_start = 0.0
            agent_config.epsilon_end = 0.0
            agent_config.epsilon_decay = 1
        
        print(f"✅ Creating DQN agent with explicit config from dqn.yaml")
        return DroneDQNAgent(obs_space, act_space, agent_config, device)
    
    elif agent_type == AgentType.PPO:
        print(f"✅ Creating PPO agent with explicit config from ppo.yaml")
        return DronePPOAgent(obs_space, act_space, agent_config, device)
    
    elif agent_type == AgentType.SAC:
        print(f"✅ Creating SAC agent with explicit config from sac.yaml")
        return DroneSACAgent(obs_space, act_space, agent_config, device)
    
    elif agent_type == AgentType.RANDOM:
        print(f"✅ Creating Random agent (no config needed)")
        return RandomAgent(obs_space, act_space, device)
    
    elif agent_type == AgentType.HOVERING:
        print(f"✅ Creating Hovering agent (no config needed)")
        return HoveringAgent(obs_space, act_space, device)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_agent_from_scenario(scenario_config, agent_name: str, config: DictConfig, obs_space, act_space, device: str, for_visualization: bool = False):
    """
    Create agent directly from scenario configuration.
    
    Args:
        scenario_config: Scenario configuration containing drone definitions
        agent_name: Name of the agent to create
        config: Full configuration  
        obs_space: Observation space
        act_space: Action space
        device: Device for agent
        for_visualization: If True, disable exploration for DQN agents
        
    Returns:
        Created agent instance
        
    Raises:
        ValueError: If agent not found in scenario
    """
    # Find the drone config for this agent
    drone_config = None
    for drone in scenario_config.drones:
        if drone.name == agent_name:
            drone_config = drone
            break
    
    if drone_config is None:
        raise ValueError(f"Agent '{agent_name}' not found in scenario configuration")
    
    # Create agent using the drone's specified type
    return create_agent(
        agent_type=drone_config.agent_type,
        config=config,
        obs_space=obs_space,
        act_space=act_space,
        device=device,
        for_visualization=for_visualization
    ) 