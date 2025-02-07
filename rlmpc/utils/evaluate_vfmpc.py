from typing import Callable

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from casadi import *
import do_mpc
import matplotlib.pyplot as plt
import wandb

def evaluate(
    mpc,
    estimator,
    envs,
    rb,
    device: torch.device = torch.device("cpu"),
    eval_episodes: int = 1,
    seed: int = 1,
    add_true_data: bool = True
):
    
    envs.seed(seed=seed)
    obs = envs.reset()
    
    mpc.x0 = np.reshape(obs["observation"], envs.observation_space["observation"].shape[::-1])
    mpc.set_initial_guess()  

    episodic_returns = []
    t = 0
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = {}
            for key, value in obs.items():
                obs_tensor[key] = torch.Tensor(value).to(device)
           
            # MPC action
            x0 = estimator.make_step(obs["observation"])
            actions = mpc.make_step(x0).flatten()
            actions = np.expand_dims(np.reshape(actions, envs.action_space.shape), axis=0) # I'm assuming we're only using one environment instantiation

        next_obs, rewards, dones, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, inf in enumerate(infos):
            if inf["TimeLimit.truncated"]:
                real_next_obs = inf["terminal_observation"].copy()

        if add_true_data:
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        t += 1

        if "episode" in infos[0]:
            for info in infos:
                if "episode" not in info:
                    continue

                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]

        obs = next_obs

    return episodic_returns