## Evaluate a goal-conditioned MPC scheme for double inverted pendulum. We sweep over several prediction horizons and variances in the MPC cost.

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from casadi import *
import do_mpc
import pandas as pd

import sys
sys.path.append('')
from rlmpc.utils.networks import SoftQNetwork, ValueNetwork, Actor, ActorValueNetwork
from rlmpc.envs.examples.DIP.template_model import template_model
from rlmpc.envs.examples.DIP.template_mpc import template_mpc
from rlmpc.envs.examples.DIP.template_simulator import template_simulator


control_strategy = "all" # important to label the samples 'VF+MPC' or 'MPC' for processing later
exp_path = "src/envs/examples/DIP/results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize sampling planner
sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite=True)

num_steps = 100
goal = np.array([0.0, 0.0])
uncertain_params = "include_truth"
model = template_model()
# eval_same_goal = np.array([goal])
eval_same_goal = goal
solver = "MA27"


estimator = do_mpc.estimator.StateFeedback(model)

# Add sampling variable including the corresponding evaluation function
n_horizon = [75]
var = [0.5]
strategies = ["goal-conditioned"]

# n_horizon = [100]
# std = [1.0]

sp.set_sampling_var('control_strategy')
sp.set_sampling_var('n_horizon')
sp.set_sampling_var('var')


plan = sp.product(n_horizon=n_horizon, var=var, control_strategy=strategies)
print(plan)

# plan = sp.gen_sampling_plan(n_samples=n_samples)
pd.DataFrame(plan).head()

def run_closed_loop(control_strategy, n_horizon, var):

    mpc = template_mpc(model, mpc_mode=control_strategy, uncertain_params=uncertain_params, solver=solver, store_full_solution=False, goal = eval_same_goal, var = var, n_horizon=n_horizon, ts=0.04, input=5.0, pos=5.0, silence_solver = True)
    
    simulator = template_simulator(model, uncertain_params=uncertain_params, goal = eval_same_goal, ts=0.04, reltol=1e-4)

    """
    Set initial state
    """
    if True:
        simulator.x0['theta'] = .9*np.pi
        simulator.x0['pos'] = 0.0
        
    x0 = simulator.x0.cat.full()

    mpc.reset_history()
    simulator.reset_history()
    estimator.reset_history()

    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()
    z0 = simulator.init_algebraic_variables()

    for _ in range(num_steps):

        u0 = mpc.make_step(x0)

        x0 = simulator.make_step(u0)
        x0 = estimator.make_step(x0)
        # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

    print("done")

    return simulator.data

# Initialize sampler with generated plan
sampler = do_mpc.sampling.Sampler(plan)
# Set directory to store the results:
sampler_dir = exp_path  + "/gcmpc/"
sampler.data_dir = sampler_dir
sampler.set_param(overwrite=True)

# Set the sampling function
sampler.set_sample_function(run_closed_loop)

# Generate the data
sampler.sample_data()

# Initialize DataHandler
dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = sampler_dir

dh.set_post_processing('default', lambda data: data)
dh.set_post_processing('L1', lambda data: np.linalg.norm(data['_aux', 'error_cos_theta1'], ord=1) +  np.linalg.norm(data['_aux', 'error_cos_theta2'], ord=1))
dh.set_post_processing('L2', lambda data: np.linalg.norm(data['_aux', 'error_cos_theta1'], ord=2) +  np.linalg.norm(data['_aux', 'error_cos_theta2'], ord=2))
dh.set_post_processing('tv_input', lambda data: np.linalg.norm(np.diff(data['_u'], axis=0)))
dh.set_post_processing('time_near_goal', lambda data: np.sum(np.exp(-0.5*(np.linalg.norm(data['_aux', 'track_error_squared'],ord=2, axis=1) / 0.1)**2)))
dh.set_post_processing('fuzzy_pass_fail', lambda data: np.mean(np.exp(-0.5*(np.linalg.norm(data['_aux', 'track_error_squared'][num_steps//2::],ord=2, axis=1) / 0.1)**2)))

df = pd.DataFrame(dh[:])
df.to_pickle(sampler_dir+"smallvar_"+f"{n_horizon[0]}_"+f"{uncertain_params}_"+"sample_data.pkl")
df = pd.read_pickle(sampler_dir+"smallvar_"+f"{n_horizon[0]}_"+f"{uncertain_params}_"+"sample_data.pkl")
print(df.head())
# print(df["experiment_id"])

from rlmpc.eval.dip_plot import episode_plot
fig = episode_plot(df["default"].iloc[0])
fig.savefig(sampler_dir+"testfig")
