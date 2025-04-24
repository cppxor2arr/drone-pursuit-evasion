from task.lider_drone_base import LidarDroneBaseEnv
import pybullet as p
from PyFlyt.pz_envs import MAQuadXHoverEnvV2
import numpy as np
from supersuit import frame_stack_v1

env = LidarDroneBaseEnv(
    lidar_reach=4.0,
    num_ray=3,
    render_mode="human",
    flight_mode=7,
    start_pos=np.array([[1, 1, 1], [-1, -1, 1]]),
    start_orn=np.array([[0, 0, 0], [0, 0, 0]]),
)  # MAQuadXHoverEnvV2(render_mode="human")
env = frame_stack_v1(env, stack_size=2)
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # actions = {agent: [0,0,0,0] for agent in env.agents}

    # Hack
    env.unwrapped.prev_observations = {
        agent: ob[:12] for agent, ob in observations.items()
    }

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    print(actions)
    # for agent in env.agents:
    #     print(env.action_space(agent).sample(), env.observation_space(agent))
env.close()
