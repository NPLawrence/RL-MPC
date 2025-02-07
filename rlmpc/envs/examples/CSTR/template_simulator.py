#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import copy


def template_simulator(model, RL_env=False, uncertain_params = "include_truth", eval_uncertain_env=False, goal=0.60):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.005
    }

    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['goal'] = goal
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    p_num = simulator.get_p_template()
    num = 10
    if RL_env:
        if uncertain_params == "include_truth": # the default option from do-mpc's examples
            alpha_var, beta_var = [np.array([1.0, 1.05, 0.95]), np.array([1.0, 1.1, 0.9])]
        elif uncertain_params == "nominal":
            alpha_var, beta_var = [np.array([1.0]), np.array([1.0])]
        elif uncertain_params == "missing_truth": # this is typically used for RL training, whereas "include_truth" is used for MPC deployment due to reduced number of scenarios
            alpha_var, beta_var = [np.linspace(1.05, 0.95, num), np.linspace(1.1, 0.9, num)]
        elif uncertain_params == "include_truth_expanded":
            alpha_var, beta_var = [np.array([1.0, 1.1, 0.90]), np.array([1.0, 1.2, 0.8])]

        def p_fun(t_now):
            alpha = np.random.choice(alpha_var)
            beta = np.random.choice(beta_var)
            p_num['alpha'], p_num['beta'] = alpha, beta   
            return p_num 
    else:
        if uncertain_params == "include_truth_expanded":
            p_num['alpha'], p_num['beta'] = 1.10, 1.15
        else:
            p_num['alpha'], p_num['beta'] = 1, 1

        if eval_uncertain_env:
            alpha_var, beta_var = [np.linspace(1.05, 0.95, 7), np.linspace(1.1, 0.9, 8)] # evaluate on values not exactly seen during training
            def p_fun(t_now):
                if t_now==0.0:
                    alpha = np.random.choice(alpha_var)
                    beta = np.random.choice(beta_var)
                    p_num['alpha'], p_num['beta'] = alpha, beta
                return p_num

        else:
            def p_fun(t_now):
                return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator