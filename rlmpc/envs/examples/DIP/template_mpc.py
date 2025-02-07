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

def template_mpc(model, vf = None, goal = np.array([0.0, 0.0]), mpc_mode = "vfmpc", n_horizon = 1, n_robust = 0, silence_solver = True, solver="pardiso", store_full_solution = True,
                RL_env=False, uncertain_params = "include_truth", var = 1.0, input=10.0, ts=0.04, pos=10.0):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.open_loop =  False
    mpc.settings.t_step =  ts
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  3
    mpc.settings.collocation_ni =  1
    mpc.settings.store_full_solution =  store_full_solution
    mpc.use_terminal_bounds = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()
    # see https://coin-or.github.io/Ipopt/OPTIONS.html and https://arxiv.org/pdf/1909.08104
    mpc.nlpsol_opts = {'ipopt.linear_solver': solver} # pardiso, MA27, MA57, spral, HSL_MA86, mumps

    mpc.bounds['lower','_u','force'] = -input
    mpc.bounds['upper','_u','force'] = input

    mpc.bounds['lower', '_x', 'pos'] = -pos
    mpc.bounds['upper', '_x', 'pos'] = pos

    # Values for the masses (for robust MPC)
    if uncertain_params == "include_truth":
        m0_var = 0.6*np.array([1, 0.95, 1.05])
        m1_var = 0.2*np.array([1, 0.95, 1.05])
        m2_var = 0.2*np.array([1, 0.95, 1.05])
    elif uncertain_params == "missing_truth":
        m0_var = 0.6*np.array([0.95, 1.05])
        m1_var = 0.2*np.array([0.95, 1.05])
        m2_var = 0.2*np.array([0.95, 1.05])
    elif uncertain_params == "nominal":
        m0_var = 0.6*np.array([1.0])
        m1_var = 0.2*np.array([1.0])
        m2_var = 0.2*np.array([1.0])

    mpc.set_uncertainty_values(m0=m0_var, m1=m1_var, m2=m2_var)

    if mpc_mode == "baseline":

        mpc.settings.n_horizon =  n_horizon
        mpc.settings.n_robust =  n_robust

        mterm = model.aux['E_kin'] - model.aux['E_pot'] # terminal cost
        lterm = model.aux['E_kin'] - model.aux['E_pot'] # stage cost


        mpc.set_objective(mterm=mterm, lterm=lterm)
        # Input force is implicitly restricted through the objective.
        mpc.set_rterm(force=0.1)

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp', :, 'goal_theta1'] = goal.flatten()[0]
            tvp_template['_tvp', :, 'goal_theta2'] = goal.flatten()[1]
            # tvp_template['_tvp', :, 'goal_height'] = goal.flatten()
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()
        
    elif mpc_mode == "goal-conditioned":

        mpc.settings.n_horizon =  n_horizon
        mpc.settings.n_robust =  n_robust

        mterm = 1-np.exp(-0.5*(model.aux['error_cos_theta1']**2 + model.aux['error_cos_theta2']**2) /  var) # terminal cost
        lterm = 1-np.exp(-0.5*(model.aux['error_cos_theta1']**2 + model.aux['error_cos_theta2']**2) / var) # stage cost

        mpc.set_objective(mterm=mterm, lterm=lterm)

        tvp_template = mpc.get_tvp_template()
        
        def tvp_fun(t_ind):
            tvp_template['_tvp', :, 'goal_theta1'] = goal.flatten()[0]
            tvp_template['_tvp', :, 'goal_theta2'] = goal.flatten()[1]

            # uncomment below to switch goals online
            # ind = t_ind // mpc.settings.t_step
            # if ind <= 6.0 // mpc.settings.t_step:
            #     tvp_template['_tvp', :, 'goal_theta1'] = goal.flatten()[0]
            #     tvp_template['_tvp', :, 'goal_theta2'] = goal.flatten()[1]
            # elif ind <= 12.0 // mpc.settings.t_step:
            # # else:
            #     tvp_template['_tvp', :, 'goal_theta1'] = np.pi
            #     tvp_template['_tvp', :, 'goal_theta2'] = 0.0
            # else:
            #     tvp_template['_tvp', :, 'goal_theta1'] = 0.0
            #     tvp_template['_tvp', :, 'goal_theta2'] = 0.0

            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

    elif mpc_mode == "quadratic":

        mpc.settings.n_horizon =  n_horizon
        mpc.settings.n_robust =  n_robust

        mterm = 0.5*(model.aux['error_cos_theta1']**2 + model.aux['error_cos_theta2']**2) # terminal cost
        lterm = 0.5*(model.aux['error_cos_theta1']**2 + model.aux['error_cos_theta2']**2) # stage cost

        mpc.set_objective(mterm=mterm, lterm=lterm)

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_ind):
            # ind = t_ind // mpc.settings.t_step
            ind = t_ind // mpc.settings.t_step
            if ind <= 6.0 // mpc.settings.t_step:
                tvp_template['_tvp', :, 'goal_theta1'] = goal.flatten()[0]
                tvp_template['_tvp', :, 'goal_theta2'] = goal.flatten()[1]
            elif ind <= 12.0 // mpc.settings.t_step:
            # else:
                tvp_template['_tvp', :, 'goal_theta1'] = np.pi
                tvp_template['_tvp', :, 'goal_theta2'] = 0.0
            else:
                tvp_template['_tvp', :, 'goal_theta1'] = 0.0
                tvp_template['_tvp', :, 'goal_theta2'] = 0.0
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc