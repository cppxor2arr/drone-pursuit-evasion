from task.lider_drone_base import LidarDroneBaseEnv
import pybullet as p
from PyFlyt.pz_envs import MAQuadXHoverEnvV2
import numpy as np


def render_flight_dome(env):
    concaveSphereVisualId = env.aviary.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName="hi_res_sphere.obj",
        meshScale=[-env.flight_dome_size] * 3,
        rgbaColor=[1.0, 0.0, 0.0, 0.8],
        specularColor=[0.4, 0.4, 0.4],
    )
    concaveSphereId = env.aviary.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=concaveSphereVisualId,
        basePosition=[0.0, 0.0, 0.0],
        useMaximalCoordinates=True,
        flags=0,
    )
    convexSphereVisualId = env.aviary.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName="hi_res_sphere.obj",
        meshScale=[env.flight_dome_size] * 3,
        rgbaColor=[1.0, 0.0, 0.0, 0.8],
        specularColor=[0.4, 0.4, 0.4],
    )
    convexSphereId = env.aviary.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=convexSphereVisualId,
        basePosition=[0.0, 0.0, 0.0],
        useMaximalCoordinates=True,
        flags=0,
    )
    env.aviary.register_all_new_bodies()


env = LidarDroneBaseEnv(
    lidar_reach=4.0,
    num_ray=3,
    render_mode="human",
    flight_mode=7,
    start_pos=np.array([[1, 1, 1], [-1, -1, 1]]),
    start_orn=np.array([[0, 0, 0], [0, 0, 0]]),
)  # MAQuadXHoverEnvV2(render_mode="human")

observations, infos = env.reset()
print(observations)

render_flight_dome(env)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # actions = {agent: [0,0,0,0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    print(actions)
    print(f"Reward: {rewards}")
    # for agent in env.agents:
    #     print(env.action_space(agent).sample(), env.observation_space(agent))
env.close()
