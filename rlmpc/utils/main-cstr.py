"""Run a bunch of experiments in a loop"""
import os
import runpy
import sys
import time
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

import wandb
from datetime import date

project_name = "CSTR-RL-MPC-"+f"{date.today()}"

# n_goals = [4, 0]
# rewards = ["default"]
trials = 4
# model_horizon = [1, 4]
# grad_steps = [1]
# target_network_frequency = [1, 2]
# batch_size = [64]
# policy_lr = [1e-4, 1e-3]
# q_lr = [1e-3]
n_sampled_goal = [1]
goal_selection_strategy = ["future"]
# linear_solver = ["pardiso"] 
# eps = [1e-03]
error_vs_goal = ["error"]
optimizer = ["AdamW"]
uncertain_params = ["missing_truth", "nominal"]
add_true_data = [0]
n_horizon = [1]

experiment_num = 0
for seed in range(trials):
    for n in n_sampled_goal:
        for g in goal_selection_strategy:
            for eg in error_vs_goal:
                for opt in optimizer:
                    for params in uncertain_params:
                        for adddata in add_true_data:
                            for horiz in n_horizon:

                                experiment_num += 1

                                print("Experiment: ", experiment_num)
                                # the empty string in the first spot is actually important!
                                print(seed, n, g, eg)
                                # the empty string in the first spot is actually important!
                                sys.argv = ["",  f"--seed={int(seed)}", f"--optimizer={str(opt)}", f"--uncertain_params={str(params)}", f"--add_true_data={adddata}",
                                            f"--n_sampled_goal={int(n)}", f"--goal_selection_strategy={str(g)}", f"--error_vs_goal={str(eg)}", f"--n_horizon={int(horiz)}",
                                            f"--wandb_project_name={project_name}"]

                                experiment = runpy.run_path(path_name="cstr_sac_mpc.py", run_name="__main__")

