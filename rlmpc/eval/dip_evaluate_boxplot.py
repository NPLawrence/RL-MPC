"""
Produces boxplot showing performance of various mpc schemes as the prediction horizon varies
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from casadi import *
import do_mpc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(palette="tab10", style='whitegrid')

def create_boxplot(df, dir = 'src/envs/examples/DIP/results/gcmpc', name="dip_boxplot", x="n_horizon", y="fuzzy_pass_fail", hue="control_strategy", ylabel="Average steady-state reward", legend=True):
    SMALL_SIZE = 15
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=3)

    params = {
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(layout='constrained')
    boxplot = sns.barplot(data=df, x=x, y=y, hue=hue, legend=legend, saturation=.75, gap=0.05)

    sns.move_legend(ax, "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=3, labels=["Expert", "Quadratic", "Goal-conditioned"])
     
    plt.xlabel("Prediction horizon")
    plt.ylabel(ylabel)
    # ax.set_xticklabels(xlabels)

    fig.savefig(dir + name)
    fig.savefig(dir + name + ".pdf")

    return

if __name__ == '__main__':

    dir = 'src/envs/examples/DIP/results/gcmpc/'

    # run dip_evaluate_gcmpc.py several times to generate a list of dfs below
    N = [25, 75, 300]

    dfs = [pd.read_pickle(dir+"smallvar_75_include_truth_sample_data.pkl")]
    df_new = pd.concat(dfs, ignore_index=True)
    df_new['fuzzy_pass_fail'].iloc[0] = 0.01
    df_new['time_near_goal'].iloc[0] = 0.5

    df_new.to_pickle(dir+"testing.pkl")
    print(df_new.head())

    create_boxplot(df_new, dir=dir, y="time_near_goal", ylabel="Time near goal")

