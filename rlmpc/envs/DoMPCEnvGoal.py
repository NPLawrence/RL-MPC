"""
A Gym wrapper that serves as the bridge between SB3 and do-mpc
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import do_mpc

class DoMPCEnvGoal(gym.Env):
    """
    Gym environment that uses do-mpc for carrying out simulations
    """
    # def __init__(self, simulator:do_mpc.simulator.Simulator, 
    def __init__(self, template_simulator:callable, 
                 model:do_mpc.model._model.Model,
                 bounds:dict, 
                 goal_map=lambda x: x,
                    num_steps=100,
                    clip_reset=None,
                    same_state=None,
                    same_goal=None,
                    tol=0.1,
                    RL_env=False, # if the environment is used as a simulation model (RL) or as 'true' environment. See template_simulator.py for parametric uncertainty values.
                    uncertain_params = "include_truth", # set the uncertainty keyword for template_simulator.py.
                    smooth_reward=False,
                    eval="",
                    path=''):        
        super().__init__()

        self.RL_env = RL_env
        self.model = model
        self.template_simulator = template_simulator
        self.uncertain_params = uncertain_params
        self.num_steps = num_steps
        self.clip_reset = clip_reset
        self.same_state = same_state
        self.same_goal = same_goal
        self.goal_map = goal_map
        self.episode = 0
        self.eval = eval
        self.path=path
        self.smooth_reward = smooth_reward

        goal_high = goal_map(bounds["x_high"])
        goal_low = goal_map(bounds["x_low"])

        self.action_space = spaces.Box(low=bounds['u_low'], high=bounds['u_high'], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation" : spaces.Box(low=bounds['x_low'],high=bounds['x_high'], dtype=np.float32),
            "desired_goal" : spaces.Box(low=goal_low,high=goal_high, dtype=np.float32),
            "achieved_goal" : spaces.Box(low=goal_low,high=goal_high, dtype=np.float32),
        })

        self.tol = tol

    def step(self, action):
        self.t += 1

        self.action = action
        self.state = self._simulator(action)
        # do_mpc.tools.printProgressBar(self.t, self.num_steps, prefix='Closed-loop simulation:', length=50)
    
        obs = self._state_to_dict()

        info = self._get_info()
        reward, terminated, truncated = info["reward"], info["terminated"], info["TimeLimit.truncated"]

        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):

        if type(info) is not dict:
            state = [x["full_state"] for x in info]
            is_feasible = [1*self.observation_space["observation"].contains(
                x) - 1 for x in state]
        else:
            state = info["full_state"]
            is_feasible = 1*self.observation_space["observation"].contains(
                state) - 1

        axis = 0
        if achieved_goal.ndim == 2:
            axis = 1

        is_target = np.linalg.norm(desired_goal - achieved_goal, np.inf, axis=axis) < self.tol

        if self.smooth_reward:

            reward = np.exp(-0.5*(np.linalg.norm(desired_goal - achieved_goal, ord=2, axis=axis) / 0.01)**2)

        else:
        
            reward = 1*is_target
            # reward = 1*is_feasible + 1*is_target # can tune this, but we opt to use mpc for constraint handling

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode += 1

        if self.same_goal is None: # this should only be true during training -- for evaluation we feed setpoints to both the simulator and mpc
            self.goal = np.array([np.random.choice(np.array([0.6, 0.7, 0.8]))])
        else:
            self.goal = self.same_goal # value for the goal if it change between episodes

        self.simulator = self._define_simulator()
        self.simulator.reset_history()

        x = self._reset_state()
        self.action = self.action_space.sample()
        self.simulator.x0 = x
        self.state = self._fix_state(x)
        obs = self._state_to_dict()
        self.t = 0.0
        info = self._get_info()

        return obs, info

    def _state_to_dict(self):

        return {"observation" : self.state,
         "desired_goal" : self.goal,
         "achieved_goal" : self.goal_map(self.state)}
        
    def _reset_state(self, state=None):

        if self.same_state is not None: # similar to same_goal, dictates whether the system starts from the same state between episodes
            state = self.same_state
            return state
        elif self.clip_reset is not None:
            state = np.clip(self.observation_space["observation"].sample(), -self.clip_reset, self.clip_reset)
            return state
        else:
            return self.observation_space["observation"].sample()

    def _get_info(self):

        reward = self.compute_reward(self.goal_map(self.state), self.goal, {"full_state": self.state})

        distance = np.linalg.norm(self.goal_map(self.state) - self.goal).item() # possible to set `terminated = distance < 0.0001`
        terminated = False
        # terminated = not self.observation_space["observation"].contains(self.state) # again, if one wishes to train rl agent subject to constraints
        truncated = self.t == self.num_steps

        if truncated or terminated:
            self.save_episode()
        
        return {"time": self.t, "distance": distance, "reward": reward, "terminated": terminated, "TimeLimit.truncated": truncated, "full_state": self.state}
    
    def _simulator(self, action):
        # functionally the same as simulator.make_step() in do-mpc but processes the output and dtype
        a = self._fix_action(action)
        x = self.simulator.make_step(a)
        return self._fix_state(x)
    
    def _define_simulator(self):
        simulator = self.template_simulator(self.model, RL_env=self.RL_env, uncertain_params=self.uncertain_params, goal=self.goal)
        return simulator
    
    def _fix_state(self, x):
        return np.float32(np.reshape(x, self.observation_space["observation"].shape))

    def _fix_action(self, action):
        a = np.reshape(action, self.action_space.shape[::-1])
        return a.reshape(a.shape + (1,))
    
    def renderMPC(self, mpc, num_steps, path=""):
        # quick plotting tool; more refined helpers in the eval
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        self._run_closed_loop(mpc, self.simulator, num_steps=num_steps)

        fig = do_mpc.graphics.default_plot(self.simulator.data)
        plt.savefig(path)
        plt.close('all')

        return fig[0]

    def _run_closed_loop(self, controller, simulator, num_steps=10, x0=None):

        controller.reset_history()
        simulator.reset_history()
        self.reset()
        x0 = self.state.copy()

        controller.x0 = x0
        simulator.x0 = x0
        controller.set_initial_guess()

        for _ in range(num_steps):
            u0 = controller.make_step(x0)
            x0 = simulator.make_step(u0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        return
    
    def save_episode(self):
        
        do_mpc.data.save_results([self.simulator], result_name=f"{self.eval}" + "episode-" + f"{self.episode}", result_path=self.path)

        return
        
    def close(self):
        ...


if __name__ == '__main__':
    print("Checking CSTR")

    import sys
    sys.path.append('')
    from rlmpc.envs.examples.CSTR.template_model import template_model
    from rlmpc.envs.examples.CSTR.template_mpc import template_mpc
    from rlmpc.envs.examples.CSTR.template_simulator import template_simulator

    model = template_model()
    mpc = template_mpc(model, mpc_mode="baseline", RL_env=True)

    # simulator = template_simulator(model)
    max_x = np.array([2.0,2.0,140.0,140.0]).flatten()
    min_x = np.array([0.1,0.1,50.0,50.0]).flatten() # writing like this to emphasize do-mpc sizing convention
    max_u = np.array([10.0,0.0]).flatten()
    min_u = np.array([0.5,-8.50]).flatten()
    bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}

    # run environment
    env = DoMPCEnvGoal(template_simulator, model, bounds)
    env.reset(seed=0)
    for _ in range(50):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        print(obs)