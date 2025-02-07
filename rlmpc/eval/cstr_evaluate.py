import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from casadi import *
import do_mpc
import pandas as pd

import sys
sys.path.append('')
from rlmpc.utils.networks import SoftQNetwork, ValueNetwork, Actor, ActorValueNetwork
from rlmpc.envs.examples.CSTR.template_model import template_model
from rlmpc.envs.examples.CSTR.template_mpc import template_mpc
from rlmpc.envs.examples.CSTR.template_simulator import template_simulator

robust_nominal_rl = "robust" # for rl, robust or nominal training (just use "" for mpc)
control_strategy = 'rl' # important to label the samples 'VF+MPC' or 'MPC' for processing later
n_horizon = 1 # not used is just implementing the vanilla RL (no MPC) policy
robustness_experiment = False # whether to change simulator parameters each rollout
if not robustness_experiment:
     tag_experiments = "nominalexp_"
else:
     tag_experiments = ""
exp_path = "runs/cstr__cstr_sac_mpc__1__1738630412/" # robust rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vf = ValueNetwork(input_size=7).to(device)
actor = Actor(input_size=5).to(device)

vf.load_state_dict(torch.load(exp_path+"/vf.pth", weights_only=True))
vf.eval()
actor.load_state_dict(torch.load(exp_path+"/actor.pth", weights_only=True))
actor.eval()
vf_actor = ActorValueNetwork(vf,actor)

# Initialize sampling planner
sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite=True)

num_steps = 50
n_samples = 200
goal = 0.65
uncertain_params = "include_truth"
model = template_model()
eval_same_goal = goal
mpc = template_mpc(model, vf=vf_actor, goal=eval_same_goal, n_horizon=n_horizon, silence_solver=True, 
                            mpc_mode="baseline", uncertain_params=uncertain_params)
estimator = do_mpc.estimator.StateFeedback(model)
simulator = template_simulator(model, uncertain_params=uncertain_params, eval_uncertain_env=robustness_experiment, goal=eval_same_goal)

# Sample random feasible initial states
state_low = [0.1, 0.1, 50.0, 50.0]
state_high = [2.0, 2.0, 140.0, 140.0]
def gen_initial_states():
    # x0 = np.array([0.8, 0.4, 134.14, 130.0])
    x0 = np.random.uniform(low=state_low, high=state_high)
    return x0

def prox(x, low, high):
     return np.clip(x, a_min=low, a_max=high)
     
# Add sampling variable and the corresponding evaluation function
sp.set_sampling_var('X0', gen_initial_states)
plan = sp.gen_sampling_plan(n_samples=n_samples)
pd.DataFrame(plan).head()

def run_closed_loop(X0):
        x0 = X0
        mpc.reset_history()
        simulator.reset_history()
        estimator.reset_history()

        mpc.x0 = x0
        simulator.x0 = x0
        estimator.x0 = x0

        mpc.set_initial_guess()

        for _ in range(num_steps):
            if control_strategy == 'rl':
                x = torch.tensor([np.transpose(np.concatenate([x0,[goal - x0[1]]]))], dtype=torch.float32)
                u0 = actor.get_deterministic_action(x)
                u0 = u0.detach().cpu().numpy()
                u0 = np.reshape(u0, (2, 1))
            else:
                u0 = mpc.make_step(x0)

            x0 = simulator.make_step(u0)
            x0 = estimator.make_step(x0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        return simulator.data

# Initialize sampler with generated plan
sampler = do_mpc.sampling.Sampler(plan)
# Set directory to store the results:
sampler_dir = exp_path  + "/sampling_closed_loop/"
sampler.data_dir = sampler_dir
sampler.set_param(overwrite=True)

# Set the sampling function
sampler.set_sample_function(run_closed_loop)

# Generate the data
sampler.sample_data()

# Initialize DataHandler
dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = sampler_dir

# A bunch of things to log
dh.set_post_processing('default', lambda data: data)
dh.set_post_processing('num_steps', lambda data: num_steps)
dh.set_post_processing('L1', lambda data: np.linalg.norm(data['_aux', 'track_error'], ord=1))
dh.set_post_processing('L1_tail', lambda data: np.linalg.norm(data['_aux', 'track_error'][num_steps//2::], ord=1))
dh.set_post_processing('absolute_percent_error', lambda data: np.mean(100*np.abs(data['_aux', 'track_error'][num_steps//2::])/goal))
dh.set_post_processing('L2', lambda data: np.linalg.norm(data['_aux', 'track_error'], ord=2))
dh.set_post_processing('tv_input', lambda data: np.linalg.norm(np.diff(data['_u'], axis=0)))
dh.set_post_processing('constraint_violations', lambda data: np.sum(np.any(state_low > data['_x'], axis=1) + np.any(state_high < data['_x'], axis=1))) 
dh.set_post_processing('time_near_goal', lambda data: np.sum(np.exp(-0.5*(np.linalg.norm(data['_aux', 'track_error'],ord=2, axis=1) / 0.1)**2)))
# dh.set_post_processing('data_minus_prox', lambda data: data['_x'] - prox(data['_x'], low=state_low, high=state_high))
dh.set_post_processing('time_outside_constraints', lambda data: np.sum(np.exp(-0.5*(np.linalg.norm(data['_x'] - prox(data['_x'], low=state_low, high=state_high),ord=2, axis=1) / 0.1)**2) - 1))
dh.set_post_processing('smallvar_fuzzy_pass_fail', lambda data: np.mean(np.exp(-0.5*(np.linalg.norm(data['_aux', 'track_error'][num_steps//2::],ord=2, axis=1) / 0.01)**2)))
dh.set_post_processing('fuzzy_pass_fail', lambda data: np.mean(np.exp(-0.5*(np.linalg.norm(data['_aux', 'track_error'][num_steps//2::],ord=2, axis=1) / 0.1)**2)))
dh.set_post_processing('error', lambda data: np.mean(data['_aux', 'track_error']))
dh.set_post_processing('error_tail', lambda data: np.mean(data['_aux', 'track_error'][num_steps//2::]))


# dh.set_post_processing('success')
dh.set_post_processing('experiment_id', lambda data: exp_path)
dh.set_post_processing('control_strategy', lambda data: robust_nominal_rl + "_" + control_strategy + '_' + uncertain_params)

df = pd.DataFrame(dh[:])
df['overall'] = df['time_near_goal'] + df['time_outside_constraints']
df.to_pickle(sampler_dir + robust_nominal_rl + "" + "_" + f"{control_strategy}_" + tag_experiments + "_sample_data.pkl")
df = pd.read_pickle(sampler_dir + robust_nominal_rl + "" + "_" + f"{control_strategy}_" + tag_experiments + "_sample_data.pkl")
print(df.head())
print(df["experiment_id"])

from cstr_plot import episode_plot
fig = episode_plot(df['default'].iloc[0])
fig.savefig(sampler_dir+"testfig")
