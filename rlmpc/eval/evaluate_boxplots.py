"""
Takes in a data frame and produces a boxplot. Combine several datasets into one data frame to compare them in the same boxplot.
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

sns.set(palette='tab10', style='whitegrid')

def create_boxplot(df, dir_ext="boxplot", x="control_strategy", y="Performance", hue="Breakdown", xlabels=["Nominal MPC", "Robust MPC", "GC MPC", "RL", "RL+MPC"], legend=True):
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
    boxplot = sns.boxenplot(data=df, x=x, y=y, hue=hue, legend=legend)
    sns.move_legend(ax, "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=2)

    plt.xlabel("")
    ax.set_xticklabels(xlabels)

    # Add underbraces to group xlabels together
    # ax.annotate('Robust RL + MPC', xy=(4.5/6, -0.1), xytext=(4.5/6, -0.2), xycoords='axes fraction', 
    #             fontsize=SMALL_SIZE, ha='center', va='bottom',
    #             arrowprops=dict(arrowstyle='-[, widthB=5.9, lengthB=0.5', lw=1.5, color='k'))
    
    # ax.annotate('Robust', xy=(1/3, -0.1), xytext=(1/3, -0.2), xycoords='axes fraction', 
    #             fontsize=SMALL_SIZE, ha='center', va='bottom',
    #             arrowprops=dict(arrowstyle='-[, widthB=3.450, lengthB=0.5', lw=1.5, color='k'))

    fig.savefig(dir_ext)
    fig.savefig(dir_ext + ".pdf")
    
    return

def combine_df(dfs: list, x="control_strategy", y="Performance", hue="Breakdown", values=["time_near_goal", "time_outside_constraints"], labels=["Time near goal", "Time outside constraints"], dfs_independent=True):
    """
    Takes list of dataframes and constructs a single one with new features and categories.
    x: the x label, should already exist as a categorical variable in the dataframes
    y: the y label, column name in the new dataframe of interest
    hue: how classes on the x axis are broken up into subcategories; idea is to have the x axis show MPC vs RL and each one of those  breaks down the performance in terms of reward and constraint violations
    values: the columns in the old dataframes that we want to be the new subcategories for the hue value
    labels: what to display in the legend; if there is an extra term in the labels, it means we add the 'values' to get an overall score
    """

    data_all = {x: [], y: [], hue: []}
    for (j, df) in enumerate(dfs):
        if dfs_independent: # if the x value is the same across 2+ dfs, this makes sure each df produces an independent section of the box plot, otherwise they will be combined
            df[x] = df[x] + "_" + str(j)
        for (i, val) in enumerate(values):
            data_all[x] += df[x].tolist()
            data_all[y] += df[val].tolist()
            data_all[hue] += [labels[i] for _ in range(len(df[val]))]
    df = pd.DataFrame(data_all)
    df.rename(columns={})
    return df

def collect_dfs(dir, tag="nominal_rl", features=["time_near_goal", "fuzzy_pass_fail", "L1", "L1_tail", "error_tail", "smallvar_fuzzy_pass_fail", "error", "absolute_percent_error"]):

    # df_nominal = pd.concat([pd.read_pickle(dir+f"nominalexp_rl_"+f"{i}"+"_sample_data.pkl") for i in range(1,7)], ignore_index=True)
    # df_robust = pd.concat([pd.read_pickle(dir+f"rl_"+f"{i}"+"_sample_data.pkl") for i in range(1,7)], ignore_index=True)

    df_nominal = pd.read_pickle(dir+"_rl_nominalexp__sample_data.pkl")
    df_robust = pd.read_pickle(dir+"_rl__sample_data.pkl")
     
    df_nominal['tag'] = tag

    for ft in features:

        df_nominal[ft] = df_robust[ft]

    df_nominal.to_pickle(dir + tag + "_all.pkl")

    return df_nominal


if __name__ == '__main__':

    # This is a simple use case of generating the boxen plots like the ones in the paper.
    # It uses the sampled closed loop data under `runs` from the trained robust RL agent.
    # One experiment corresponds to robust evaluation, while the other is nominal evaluation; the boxen plot shows the performance of the same agent under these different conditions.
    # By running more evaluation experiments with cstr_evaluate.py, more elaborate boxen plots are possible

    ## CSTR stuff
    robust_rl_dir = "runs/cstr__cstr_sac_mpc__1__1738630412/sampling_closed_loop/robust_rl__sample_data.pkl"
    robust_rl_nominalexp_dir = "runs/cstr__cstr_sac_mpc__1__1738630412/sampling_closed_loop/robust_rl_nominalexp__sample_data.pkl"
    dfs = [pd.read_pickle(robust_rl_dir), pd.read_pickle(robust_rl_nominalexp_dir)]

    # xlabels = [r"MPC\newline (Nominal)", "MPC", "RL", "1-step", "2-step", "5-step"]
    xlabels = ["RL", "RL (nominal exp.)"]

    df_all = combine_df(dfs)
    create_boxplot(df_all, xlabels=xlabels, legend=True)
