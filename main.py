from task.lider_drone_base import LidarDroneBaseEnv
import pybullet as p
from PyFlyt.pz_envs import MAQuadXHoverEnvV2

env = LidarDroneBaseEnv(render_mode="human")#MAQuadXHoverEnvV2(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
env.close()
