import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from casadi import *
import do_mpc
import pandas as pd
import copy

import sys
sys.path.append('')
from do_mpc.data import save_results, load_results
import matplotlib.pyplot as plt
import seaborn as sns

colors=['#8da0cb','#8E5572','#fc8d62']
palette = sns.color_palette(colors, len(colors))
sns.set(palette=palette, style='ticks')
constraint_color = '#66c2a5'
alpha = 0.25

def get_trajectories(controllers:dict, simulator, estimator, state_init=np.array([0.8, 0.4, 134.14, 130.0]), goal=0.65, num_steps=100, save_data=True):

    simulator_data = {}

    for key, controller in controllers.items():

        x0 = state_init

        if 'rl' not in key: ## assume dictionary only contains RL policies or MPCs
            controller.reset_history()
            controller.x0 = x0
            controller.set_initial_guess()

        simulator.reset_history()
        estimator.reset_history()
        
        simulator.x0 = x0
        estimator.x0 = x0

        for _ in range(num_steps):
            if 'rl' in key:
                x = torch.tensor([np.transpose(np.concatenate([x0,[goal - x0[1]]]))], dtype=torch.float32)
                u0 = controller.get_deterministic_action(x)
                u0 = u0.detach().cpu().numpy()
                u0 = np.reshape(u0, (2, 1))
            else:
                u0 = controller.make_step(x0)

            x0 = simulator.make_step(u0)
            x0 = estimator.make_step(x0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        simulator_data[key] = copy.deepcopy(simulator.data)
        if save_data:
            do_mpc.data.save_results([simulator], result_name=key)

    return simulator_data


def plot_trajectories(sim_data:dict, show_actions=False, labels=["RL", "MPC", "RL + MPC"], low=[0.1, 0.1, 50.0, 50.0] , high=[2.0, 2.0, 140.0, 140.0], num_steps=50, path=''):
    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=3)
    params = {
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    # Initialize graphic:
    if show_actions:
        num_plots = 6
    else:
        num_plots = 4

    fig, ax = plt.subplots(num_plots, sharex=True, layout='constrained')
    plt.xlim((0.0, 0.005*num_steps))

    # Configure plot:
    for key, data in sim_data.items():
        graphics = do_mpc.graphics.Graphics(data)
        # graphics.clear()

        graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
        graphics.add_line(var_type='_x', var_name='C_b', axis=ax[1])
        graphics.add_line(var_type='_aux', var_name='goal', axis=ax[1], linestyle='dotted', color='grey')
        graphics.add_line(var_type='_x', var_name='T_R', axis=ax[2])
        graphics.add_line(var_type='_x', var_name='T_K', axis=ax[3])
    #     graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
    #     graphics.add_line(var_type='_aux', var_name='track_error', axis=ax[2])
        if show_actions:
            graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[4])
            graphics.add_line(var_type='_u', var_name='F', axis=ax[5])
        ax[0].set_ylabel(r'$c_A$')
        ax[1].set_ylabel(r'$c_B$')
        ax[2].set_ylabel(r'$T_R$')
        ax[3].set_ylabel(r'$T_K$')
        
        if show_actions:
            ax[4].set_ylabel(r'$\dot{Q}$ [mW]')
            ax[5].set_ylabel('Flow [dal/h]')
            ax[5].set_xlabel('Time [h]')
        else:
            ax[3].set_xlabel('Time [h]')

        fig.align_ylabels()
        graphics.plot_results()

    ax[0].legend(labels)
    sns.move_legend(ax[0], "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=3)

    ymin, ymax = ax[0].get_ylim()
    ax[0].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[0],ymin]), np.min([high[0], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[1].get_ylim()
    ax[1].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[1],ymin]), np.min([high[1], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[2].get_ylim()
    ax[2].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[2],ymin]), np.min([high[2], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[3].get_ylim()
    ax[3].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[3],ymin]), np.min([high[3], ymax]), color=constraint_color, alpha=alpha)

    fig.savefig("plotall")
    fig.savefig("plotall.pdf")

    return fig


if __name__ == '__main__':

    from rlmpc.utils.networks import SoftQNetwork, ValueNetwork, Actor, ActorValueNetwork
    from rlmpc.envs.examples.CSTR.template_model import template_model
    from rlmpc.envs.examples.CSTR.template_mpc import template_mpc
    from rlmpc.envs.examples.CSTR.template_simulator import template_simulator

    exp_path_robust = "runs/cstr__cstr_sac_mpc__1__1738630412/" # robust rl

    state_init = np.array([0.8, 0.3, 100.14, 100.0]) # MPC gets stuck -- run with eval_uncertain_env=True to get situations where both RL and MPC fail but together they succeed
    # state_init = np.array([0.2, 0.3, 120.14, 120.0])

    goal = 0.6
    n_horizon = 5 # for nstep RL+MPC
    num_steps = 100

    ## Get actor-critic networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vf_robust = ValueNetwork(input_size=7).to(device)
    actor_robust = Actor(input_size=5).to(device)

    vf_robust.load_state_dict(torch.load(exp_path_robust+"/vf.pth", weights_only=True))
    vf_robust.eval()

    actor_robust.load_state_dict(torch.load(exp_path_robust+"/actor.pth", weights_only=True))
    actor_robust.eval()

    vf_actor_robust = ActorValueNetwork(vf_robust,actor_robust)

    ## Setup MPC
    model = template_model()
    mpc = template_mpc(model, vf=None, goal=goal, silence_solver=True, 
                                mpc_mode="baseline", uncertain_params="include_truth")
    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = template_simulator(model, uncertain_params="include_truth", eval_uncertain_env=False, goal=goal)

    ## Setup nstep RL+MPC
    mpc_rlmpc = template_mpc(model, vf=vf_actor_robust, goal=goal, n_horizon=n_horizon, silence_solver=True, 
                                mpc_mode="nstep_vfmpc", uncertain_params="include_truth")
    
    controllers = {"rl": actor_robust,
                       "mpc": mpc,
                       "nstep": mpc_rlmpc}

    sim_data = get_trajectories(controllers, simulator, estimator, state_init=state_init, goal=goal, num_steps=num_steps)
    # do_mpc.data.save_results([sim_data['rl']], result_name='rl')
    # do_mpc.data.save_results(sim_data['mpc'], result_name='mpc')
    # do_mpc.data.save_results(sim_data['nstep'], result_name='nstep')


    plot_trajectories(sim_data, num_steps=num_steps)