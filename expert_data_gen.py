import metaworld
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from metaworld.policies import ENV_POLICY_MAP
import skvideo.io
import numpy as np

def obs_processor(obs, overwrite_goal=None):
    oc = np.concatenate([obs[:3], obs[-3:]])
    if overwrite_goal is not None:
        oc[-3:] = overwrite_goal
    return oc

def get_expert_trajectory(env, env_name, goal_pos=None, max_traj_len=200, overwrite_goal=None, random_actions=False):
    """
    Get an expert (or noisy expert)trajectory from the environment.
    """
    p = ENV_POLICY_MAP[env_name]()
    obs, info = env.reset()
    if goal_pos is not None:
        env.env.env.env.env.env.env.env._target_pos = goal_pos

    obs = env.env.env.env.env.env.env.env._get_obs()

    done = False
    count = 0
    imgs = []
    traj = []
    while count < max_traj_len:
        count += 1
        if random_actions:
            a = env.action_space.sample()
        else:
            a = p.get_action(obs)
        next_obs, rewards, trunc, termn, info = env.step(a)
        done = trunc or termn
        step = {
            "obs": obs_processor(obs, overwrite_goal),
            "action": a,
            "next_obs": obs_processor(next_obs, overwrite_goal),
            "rewards": rewards,
            "dones": done,
        }
        traj.append(step)
        
        obs = next_obs

    return traj

def execute_policy(env, policy, goal_pos=None, max_traj_len=200):
    """
    Execute a policy in the environment and return the trajectory.
    """
    obs, info = env.reset()
    if goal_pos is not None:
        env.env.env.env.env.env.env.env._target_pos = goal_pos

    obs = env.env.env.env.env.env.env.env._get_obs()

    done = False
    count = 0
    imgs = []
    traj = {
        "obs": [],
        "actions": [],
        "next_obs": [],
        "rewards": [],
        "dones": [],
    }
    while count < max_traj_len:
        count += 1
        a = policy.get_action(obs_processor(obs))
        next_obs, rewards, trunc, termn, info = env.step(a)
        done = trunc or termn

        traj["obs"].append(obs_processor(obs))
        traj["action"].append(a)
        traj["next_obs"].append(obs_processor(next_obs))
        traj["rewards"].append(rewards)
        traj["dones"].append(done)
        
        obs = next_obs
        # print(int(info["success"]))

    return traj

if __name__ == "__main__":
    env_name = "reach-v3"
    max_traj_len = 50
    goal_pos = np.array([-0.02520992,  0.82678295,  0.16976698])
    overwrite_goal = np.array([-0.42520992,  0.82678295,  0.46976698])
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")
    traj = get_expert_trajectory(env, env_name, goal_pos=goal_pos, overwrite_goal=overwrite_goal, max_traj_len=max_traj_len)
        
    import IPython; IPython.embed()