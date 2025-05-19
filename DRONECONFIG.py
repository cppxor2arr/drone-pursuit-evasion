from task.lider_drone_base import DroneConfig, DroneRole
import numpy as np
drone_configs = [
    DroneConfig(
        role=DroneRole.PURSUER,
        name="pursuer_drone",
        start_pos=np.array([1, 1, 1]),
        start_orn=np.array([0, 0, 0]),
        action_length=7.0,
        is_training=True,  # Train the pursuer
        resume_from=None   # Start from scratch
    ),
    DroneConfig(
        role=DroneRole.EVADER,
        name="evader_drone",
        start_pos=np.array([-1, -1, 1]),
        start_orn=np.array([0, 0, 0]),
        action_length=7.0,
        is_training=True,  # Train the evader
        resume_from=None   # Start from scratch
    )
]
