# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

# 
from rlmpc.envs.DoMPCEnvGoal import DoMPCEnvGoal
from casadi import *
import do_mpc
# import onnx
from do_mpc.data import save_results, load_results
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    eval_dompc: bool = False
    """toggle whether to evaluate RL policy with vf+mpc in do-mpc"""
    wandb_project_name: str = "fml"
    """the wandb's project name"""
    wandb_entity: str = "nplawrence"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "cstr"
    """the environment id of the task"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    optimizer: str = "AdamW"
    """the optimizer used under the hood"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    n_sampled_goal: int = 1
    """use hindsight experience relabeling. Set to 0 to get baseline performance (no HER)."""
    goal_selection_strategy: str =  "future"
    """"HER goal selection strategy: "episode", "final", or "future" """
    error_vs_goal: str = "error"
    """whether to use the error signal or the goal signal as input to networks"""
    RL_env: bool = True
    """whether to train using random environment parameters for each timestep; RL_env=False indicates the nominal case of plant=simulator"""
    add_true_data: int = 0
    """whether to augment replay buffer with data from the true environment evaluation"""
    mpc_mode: str = "nstep_vfmpc"
    """run experiment with just MPC, no added value function"""
    n_horizon: int = 1
    """prediction horizon for MPC model"""
    smooth_reward: bool = True # use smooth_reward=False for binary reward
    """use a binary reward or an approximation based on a Gaussian"""
    linear_solver: str = "pardiso"
    """linear solver for IPOPT: pardiso, spral, MA27, MA57, HSL_MA86, mumps -- pardiso more memory efficient"""
    uncertain_params: str = "include_truth"
    """various uncertain parameters to be fed into the `RL_env` and robust mpc scheme; for do-mpc the first component is the 'true' value"""
    

def make_doMPC():

    model = template_model()
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = template_mpc(model, mpc_mode="baseline", RL_env=True) # RL_env=True indicates that soft constraints should be hard for the purposes of defining the env parameters
    simulator = template_simulator(model)

    return model, estimator, mpc, simulator

def get_bounds(mpc):
        model = mpc.model

        max_x = np.array([mpc.bounds['upper', '_x', key] for key in model._x.keys()]).flatten()
        min_x = np.array([mpc.bounds['lower', '_x', key] for key in model._x.keys()]).flatten()

        max_u = np.array([mpc.bounds['upper', '_u', key] for key in model._u.keys() if key != 'default']).flatten()
        min_u = np.array([mpc.bounds['lower', '_u', key] for key in model._u.keys() if key != 'default']).flatten()
        bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}

        return bounds

def make_env(template_simulator, model, bounds, seed, goal_map=lambda x: x, num_steps=100, tol=0.1, clip_reset=None,
            same_state=None, same_goal=None, RL_env=False, smooth_reward=False,
            path='', eval=""):

    def thunk():

        env = DoMPCEnvGoal(template_simulator, model, bounds=bounds, goal_map=goal_map, num_steps=num_steps, tol=tol, clip_reset=clip_reset,
                        same_state=same_state, same_goal=same_goal, RL_env=RL_env, smooth_reward=smooth_reward, uncertain_params=args.uncertain_params, path=path, eval=eval)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env
    
    return thunk


def cstr_evaluate(rb, goal=0.65, state=np.array([0.8, 0.4, 134.14, 130.0]), add_true_data=True):

    from rlmpc.utils.evaluate_vfmpc import evaluate

    eval_same_goal = np.array([goal])
    if args.mpc_mode == "vfmpc":
        vf_eval = vf
    elif args.mpc_mode == "nstep_vfmpc":
        vf_eval = vf_actor
    eval_mpc = template_mpc(model, vf=vf_eval, goal=eval_same_goal, n_horizon=args.n_horizon, silence_solver=True, 
                            mpc_mode=args.mpc_mode, solver=args.linear_solver, uncertain_params=args.uncertain_params)
    eval_path = exp_path
    eval_same_state = state
    eval_name = f"{ep_number}" + "-eval-"
    env_eval = DummyVecEnv([make_env(template_simulator, model, bounds, args.seed, RL_env=False, goal_map=goal_map, num_steps=100, smooth_reward=args.smooth_reward, tol=tol, clip_reset=clip_reset, same_state=eval_same_state, same_goal=eval_same_goal, path=eval_path, eval=eval_name)])
        
    episodic_returns = evaluate(
        eval_mpc,
        estimator,
        env_eval,
        rb,
        device,
        eval_episodes=1,
        seed=args.seed,
        add_true_data=add_true_data
    )
    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

    episode_path =  f"{run_name}/eval-fig-" + f"{ep_number}.pdf"

    eval_results = do_mpc.data.load_results(f"runs/{run_name}/"+eval_name+"episode-1.pkl")
    eval_data = episode_plot(eval_results['simulator'], path=episode_path)
    if args.track:
        wandb.log({"eval-figs": wandb.Image(eval_data)})
        plt.close(eval_data)

    return

# # ALGO LOGIC: initialize agent here:
# class SoftQNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()

#         if isinstance(env.observation_space, gym.spaces.Dict):
#             input_size = sum([get_flattened_obs_dim(env.observation_space["observation"]), get_flattened_obs_dim(env.observation_space["desired_goal"]), np.prod(env.action_space.shape)])
#         else:
#             input_size = np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape)

#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)

#         # self.actor = Actor(env)

#     def forward(self, x, a):
#         x = self._combine(x)
#         x = self.q_value(x, a)
#         return x
    
#     def q_value(self, x, a):
#         x = torch.cat([x, a], 1)
#         x = self.q_network(x)
#         return x
    
#     def q_network(self, z):
#         z = F.silu(self.fc1(z))
#         z = F.silu(self.fc2(z))
#         z = self.fc3(z)
#         return z

#     def _combine(self, x):

#         # print(x["desired_goal"] - x["achieved_goal"])
#         if args.error_vs_goal == "error":
#             x = torch.cat([x["observation"], x["desired_goal"] - x["achieved_goal"]], 1)
#         else:
#             x = torch.cat([x["observation"], x["desired_goal"]], 1)
        
#         return x

# class ValueNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()

#         self.qf = SoftQNetwork(env).to(device)

#     def forward(self, z):
#         return self.qf.q_network(z)
    
#     def _update(self, qf):
#         for param, value_param in zip(qf.parameters(), vf.qf.parameters()):
#             value_param.data.copy_(param.data)
#         return
    


# class Actor(nn.Module):
#     def __init__(self, env):
#         super().__init__()

#         self.LOG_STD_MAX = 2
#         self.LOG_STD_MIN = -5

#         if isinstance(env.observation_space, gym.spaces.Dict):
#             input_size = sum([get_flattened_obs_dim(env.observation_space["observation"]), get_flattened_obs_dim(env.observation_space["desired_goal"])])
#         else:
#             input_size = np.array(env.observation_space.shape).prod()

#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc_mean = nn.Linear(64, np.prod(env.action_space.shape))
#         self.fc_logstd = nn.Linear(64, np.prod(env.action_space.shape))
#         # action rescaling
#         self.register_buffer(
#             "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).unsqueeze(0)
#         )
#         self.register_buffer(
#             "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).unsqueeze(0)
#         )


#     def forward(self, x):
#         x = self._combine(x)
#         x = F.silu(self.fc1(x))
#         x = F.silu(self.fc2(x))
#         mean = self.fc_mean(x)
#         log_std = self.fc_logstd(x)
#         log_std = torch.tanh(log_std)
#         log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

#         return mean, log_std

#     def _combine(self, x):

#         if args.error_vs_goal == "error":
#             x = torch.cat([x["observation"], x["desired_goal"] - x["achieved_goal"]], 1)
#         else:
#             x = torch.cat([x["observation"], x["desired_goal"]], 1)
        

#         return x


#     def _explore_noise(self, x):

#         x = {"observation": torch.Tensor(x["observation"]), "desired_goal": torch.Tensor(x["desired_goal"])}
#         mean, log_std = self(x)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean.zero_(), std)
#         noise = normal.rsample() 

#         return noise

#     def get_action(self, x):
#         mean, log_std = self(x)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean

#     def get_deterministic_action(self, x):
#         # also assume x is combined (for passing to do-mpc)
#         x = F.silu(self.fc1(x))
#         x = F.silu(self.fc2(x))
#         mean = self.fc_mean(x)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return mean
    

def combine_batch(data1, data2, r=1.0):
    observations = {}
    next_observations = {}
    for key in data1.observations.keys():
        observations[key] = torch.concatenate((data1.observations[key], data2.observations[key]))

    actions = torch.concatenate((data1.actions, data2.actions))

    for key in data1.next_observations.keys():
        next_observations[key] = torch.concatenate((data1.next_observations[key], data2.next_observations[key]))

    dones = torch.concatenate((data1.dones, data2.dones))

    rewards = torch.concatenate((data1.rewards, data2.rewards))

    return DictReplayBufferSamples(observations=observations, actions=actions, next_observations=next_observations,
                                   dones=dones, rewards=rewards)


from rlmpc.eval.cstr_plot import episode_plot

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    if args.env_id == "cstr":
        from rlmpc.envs.examples.CSTR.template_model import template_model
        from rlmpc.envs.examples.CSTR.template_mpc import template_mpc
        from rlmpc.envs.examples.CSTR.template_simulator import template_simulator
        
        goal_map = lambda x: x[1]
        num_steps = 50
        tol = 0.1
        clip_reset = None
        # same_state = np.array([0.8, 0.4, 134.14, 130.0])  # 0.5
        same_state = None
        # same_goal = np.array([0.6])
        same_goal = None

    model, estimator, mpc_default, simulator = make_doMPC()
    bounds = get_bounds(mpc_default)
    # The "true" system that we don't know
    exp_path = f"runs/{run_name}/"
    envs = DummyVecEnv([make_env(template_simulator, model, bounds, args.seed, RL_env=args.RL_env, goal_map=goal_map, num_steps=num_steps, tol=tol, clip_reset=clip_reset, same_state=same_state, same_goal=same_goal, smooth_reward=args.smooth_reward, path=exp_path)])
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    # max_action = float(envs.action_space.high[0])

    from rlmpc.utils.networks import SoftQNetwork, ValueNetwork, Actor, ActorValueNetwork
    
    state_size = sum([get_flattened_obs_dim(envs.observation_space["observation"]), get_flattened_obs_dim(envs.observation_space["desired_goal"])])
    action_size = np.prod(envs.action_space.shape)
    actor = Actor(input_size=state_size, action_size=action_size, action_high=envs.action_space.high, action_low=envs.action_space.low).to(device)
    qf1 = SoftQNetwork(input_size=state_size+action_size).to(device)
    qf2 = SoftQNetwork(input_size=state_size+action_size).to(device)
    qf1_target = SoftQNetwork(input_size=state_size+action_size).to(device)
    qf2_target = SoftQNetwork(input_size=state_size+action_size).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    vf = ValueNetwork(input_size=state_size+action_size).to(device)
    vf.qf.load_state_dict(qf1.state_dict())
    vf_actor = ActorValueNetwork(vf,actor)
    if args.optimizer == "AdamW":
        q_optimizer = optim.AdamW(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.policy_lr)
    else:
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        if args.optimizer == "AdamW":
            a_optimizer = optim.AdamW([log_alpha], lr=args.q_lr)
        else:
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # envs.observation_space.dtype = np.float32
    rb = HerReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        envs,
        device,
        n_sampled_goal=args.n_sampled_goal,
        goal_selection_strategy=args.goal_selection_strategy,
        copy_info_dict=True
    )
    rb_env = HerReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        envs,
        device,
        n_sampled_goal=args.n_sampled_goal,
        goal_selection_strategy=args.goal_selection_strategy,
        copy_info_dict=True
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # obs, _ = envs.reset(seed=args.seed)
    # TRY NOT TO MODIFY: start the game
    envs.seed(seed=args.seed)
    obs = envs.reset()
    ep_number = 0

    ## INITIALIZING RL-based MPC
    # mpc = template_mpc(model, vf=vf, goal=obs["desired_goal"], n_horizon=args.n_horizon, silence_solver=True,
    #                 mpc_mode=args.mpc_mode, solver=args.linear_solver, uncertain_params=args.uncertain_params)
    # mpc.x0 = np.reshape(obs["observation"], envs.observation_space["observation"].shape[::-1])
    # # mpc.u0 = u0
    # mpc.set_initial_guess()  

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = {}
            for key, value in obs.items():
                obs_tensor[key] = torch.Tensor(value).to(device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs, rewards, dones, infos = envs.step(actions)
        # if dones: 
        #     ep_number += 1

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos[0]:
            for info in infos:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("misc/episode_number", ep_number)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, inf in enumerate(infos):
            if inf["TimeLimit.truncated"]:
                real_next_obs = inf["terminal_observation"].copy()

        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        # if global_step > args.learning_starts:
        if ep_number > 0:
            data = rb.sample(args.batch_size)
            # data_env = rb_env.sample(32) # roughly balance the ratio of sim/real data wrt to episode numbers
            # data = combine_batch(data_model, data_env)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        
        ## UPDATE mpc
        if dones: # update MPC at end of episode
            ep_number += 1

            if ep_number % 4 == 0 and args.track and ep_number > 0:
                episode_path =  f"runs/{run_name}/fig-episode-" + f"{ep_number}.pdf"


                results = do_mpc.data.load_results(f"runs/{run_name}/episode-" + f"{ep_number}.pkl")
                ep_data = episode_plot(results['simulator'], path=episode_path)
                wandb.log({"figs": wandb.Image(ep_data)})
                plt.close(ep_data)

            if args.add_true_data:
                eval_freq = 50
            else:
                eval_freq = 1000

            if (ep_number-1) % eval_freq == 0 and args.eval_dompc:
                
                vf._update(qf1)
                vf_actor = ActorValueNetwork(vf,actor)
                state_low = [0.30, 0.20, 110.0, 110.0]
                state_high = [0.9, 0.5, 130.0, 130.0]
                eval_state = np.random.uniform(low=state_low, high=state_high)
                cstr_evaluate(rb, state=eval_state, add_true_data=args.add_true_data)

                ## curious if this is the reason for significant slowdown
                # envs = DummyVecEnv([make_env(template_simulator, model, bounds, args.seed, RL_env=args.RL_env, goal_map=goal_map, num_steps=num_steps, tol=tol, clip_reset=clip_reset, same_state=same_state, same_goal=same_goal, path=exp_path)])




                
    if args.track:


        results = do_mpc.data.load_results(f"runs/{run_name}/episode-" + f"{ep_number}.pkl")
        ep_data = episode_plot(results['simulator'], path=episode_path)
        wandb.log({"figs": wandb.Image(ep_data)})



        vf._update(qf1)
        vf_actor = ActorValueNetwork(vf,actor)
        if args.eval_dompc:
            cstr_evaluate(rb, add_true_data=args.add_true_data)


        torch.save(vf.state_dict(), exp_path+"/vf.pth")
        # torch.save(vf, exp_path+"/vf.pth")
        torch.save(actor.state_dict(), exp_path+"/actor.pth")
        # vf_test = ValueNetwork(envs).to(device)
        # vf_test.load_state_dict(torch.load(exp_path+"/vf.pth", weights_only=True))
        # vf_test.eval()

        wandb.finish()

        print("close")

    envs.close()
    writer.close()