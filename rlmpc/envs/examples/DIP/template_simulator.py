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


def template_simulator(model, RL_env=False, uncertain_params = "include_truth", eval_uncertain_env=False, goal = np.array([0.0, 0.0]), reltol=1e-1, ts=0.04):

    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        # reltol smaller than 1e-1 leads to errors like 'IDA_LINESEARCH_FAIL' when running for along time -- opting for reliability over accuracy for training. Use smaller value for testing.
        'integration_tool': 'idas',
        'abstol': 1e-8,
        'reltol': reltol,
        't_step': ts
    }

    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['goal_theta1'] = goal.flatten()[0]
        tvp_template['goal_theta2'] = goal.flatten()[1]
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    p_num = simulator.get_p_template()
    num = 10
    if RL_env:
        # Values for the masses (for robust MPC)
        if uncertain_params == "include_truth":
            m0_var = 0.6*np.array([1, 0.95, 1.05])
            m1_var = 0.2*np.array([1, 0.95, 1.05])
            m2_var = 0.2*np.array([1, 0.95, 1.05])
        elif uncertain_params == "missing_truth":
            m0_var = 0.6*np.linspace(1.05, 0.95, num)
            m1_var = 0.2*np.linspace(1.05, 0.95, num)
            m2_var = 0.2*np.linspace(1.05, 0.95, num)
        elif uncertain_params == "nominal":
            m0_var = 0.6*np.array([1.0])
            m1_var = 0.2*np.array([1.0])
            m2_var = 0.2*np.array([1.0])

        def p_fun(t_now):
            m0 = np.random.choice(m0_var)
            m1 = np.random.choice(m1_var)
            m2 = np.random.choice(m2_var)
            p_num['m0'], p_num['m1'], p_num['m2'] = m0, m1, m2                
            return p_num
    else:
        if uncertain_params == "out_of_distribution":
            p_num['m0'], p_num['m1'], p_num['m2'] = 0.5, 0.3, 0.3 
        else:
            p_num['m0'], p_num['m1'], p_num['m2'] = 0.6, 0.2, 0.2

        if eval_uncertain_env:
            m0_var, m1_var, m2_var = [0.6*np.linspace(1.05, 0.95, num), 0.2*np.linspace(1.05, 0.95, num), 0.2*np.linspace(1.05, 0.95, num)]
            def p_fun(t_now):
                if t_now==0.0:
                    m0 = np.random.choice(m0_var)
                    m1 = np.random.choice(m1_var)
                    m2 = np.random.choice(m2_var)
                    p_num['m0'], p_num['m1'], p_num['m2'] = m0, m1, m2
                return p_num
        else:
            def p_fun(t_now):
                return p_num
            
    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator