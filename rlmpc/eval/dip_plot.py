import numpy as np
import do_mpc
from do_mpc.data import save_results, load_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# sns.set(palette='deep', style='white')
sns.set_palette("deep")

def episode_plot(sim_data, path=''):
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
            "axes.grid" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    # Initialize graphic:
    graphics = do_mpc.graphics.Graphics(sim_data)
    graphics.clear()

    fig, ax = plt.subplots(4, sharex=True, layout='constrained')
    # Configure plot:
    graphics.add_line(var_type='_aux', var_name='cos_theta', axis=ax[0])
    graphics.add_line(var_type='_aux', var_name='cos_goal_theta1', axis=ax[0], linestyle='dotted', color='grey')
    graphics.add_line(var_type='_aux', var_name='cos_goal_theta2', axis=ax[0], linestyle='dotted', color='grey')
    graphics.add_line(var_type='_x', var_name='pos', axis=ax[1])
    graphics.add_line(var_type='_aux', var_name='E_kin', axis=ax[2])
    graphics.add_line(var_type='_aux', var_name='E_pot', axis=ax[2])
    graphics.add_line(var_type='_u', var_name='force', axis=ax[3])
    ax[0].set_ylabel(r'$\cos(\theta)$')
    ax[1].set_ylabel(r'Position [m]')
    ax[2].set_ylabel('Energy [J]')
    ax[3].set_ylabel('Force [N]')
    ax[3].set_xlabel('Time [s]')

    label_lines = graphics.result_lines['_aux', 'cos_theta']
    ax[0].legend(label_lines, [r'$\theta_1$', r'$\theta_2$'], loc='upper left', ncol=1)
    label_lines = graphics.result_lines['_aux', 'E_kin']+graphics.result_lines['_aux', 'E_pot']
    ax[2].legend(label_lines, ['E_kin', 'E_pot'], loc='upper left', ncol=1)

    fig.align_ylabels()

    graphics.plot_results()
    graphics.reset_axes()

    return fig

def simple_plot(sim_data, path=''):
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
            "axes.grid" : False,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(2, sharex=True, layout='constrained')
    plt.xlim((sim_data['_time'][0], sim_data['_time'][-1]))

    # Configure plot:
    ax[0].plot(sim_data['_time'], sim_data['_aux', 'cos_theta', 0], color=sns.color_palette()[0])
    ax[1].plot(sim_data['_time'], sim_data['_aux', 'cos_theta', 1], color=sns.color_palette()[1])
    # ax[2].plot(sim_data['_time'], sim_data['_u', 'force'], color='grey', drawstyle='steps')

    ax[0].annotate('Up', xy=(4.5, 1), xytext=(4, np.sin(np.pi/8)), arrowprops=dict(arrowstyle='->'))
    ax[1].annotate('Down', xy=(5.44, -1), xytext=(4.5, -np.sin(np.pi/8)), arrowprops=dict(arrowstyle='->'))

    ax[0].set_ylabel(r'$\cos(\theta_1)$')
    ax[1].set_ylabel(r'$\cos(\theta_2)$')
    ax[1].set_xlabel('Time [s]')
    ax[0].set_yticks(np.array([-1.0, 0.0, 1.0]))
    ax[1].set_yticks(np.array([-1.0, 0.0, 1.0]))

    fig.align_ylabels()

    return fig