def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
import random

if __name__ == "__main__":
    from task.lider_drone_base import LidarDroneBaseEnv
    # from supersuit import frame_stack_v1, concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
    # from stable_baselines3.common.type_aliases import ReplayBufferSamples
    from stable_baselines3.common.buffers import ReplayBuffer
    from RL.dqn_agent import DroneDQNAgent
    import numpy as np
    import gymnasium 

    import multiprocessing
    multiprocessing.set_start_method("fork")

    env = LidarDroneBaseEnv(lidar_reach=4., num_ray=3, render_mode="human", flight_mode=7, start_pos=np.array([[1,1,1], [-1,-1,1]]),start_orn=np.array([[0,0,0], [0,0,0]]))

    print(env.reset())
    actions_spaces = [env.action_space(agent) for agent in env.agents]
    observation_spaces = [env.observation_space(agent) for agent in env.agents]
    #observations, rewards, terminations, truncations, infos = env.step(actions)
    drone_idx = 0
    print(observation_spaces[drone_idx],actions_spaces[drone_idx] )
    drone1 = DroneDQNAgent(observation_spaces[drone_idx], actions_spaces[drone_idx],device='cpu')
    drone_idx = 1
    drone2 = DroneDQNAgent(observation_spaces[drone_idx], actions_spaces[drone_idx],device='cpu')
    rb = ReplayBuffer(
        buffer_size=1000,
        observation_space=observation_spaces[drone_idx],
        action_space=actions_spaces[drone_idx],
        handle_timeout_termination=False,
    )
    total_timesteps = 10000
    start_e = 1
    end_e = 0
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_e, end_e, 10000, global_step)
        if random.random() < epsilon:
            drone1.exploit()
            drone2.exploit()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

