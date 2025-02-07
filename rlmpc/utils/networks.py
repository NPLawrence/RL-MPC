import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, input_size:int=7):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = self._combine(x)
        x = self.q_value(x, a)
        return x
    
    def q_value(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.q_network(x)
        return x
    
    def q_network(self, z):
        z = F.silu(self.fc1(z))
        z = F.silu(self.fc2(z))
        z = self.fc3(z)
        return z

    def _combine(self, x):

        x = torch.cat([x["observation"], x["desired_goal"] - x["achieved_goal"]], 1)
        # x = torch.cat([x["observation"], x["desired_goal"]], 1)
        
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_size:int=7):
        super().__init__()

        # self.qf = SoftQNetwork(env).to(device)
        self.qf = SoftQNetwork(input_size=input_size)

    def forward(self, z):
        return self.qf.q_network(z)
    
    def _update(self, qf):
        for param, value_param in zip(qf.parameters(), self.qf.parameters()):
            value_param.data.copy_(param.data)
        return
    
class ActorValueNetwork(nn.Module):
    def __init__(self, qf, actor):
        super().__init__()

        self.qf = qf
        self.actor = actor

    def forward(self, z):
        a = self.actor.get_deterministic_action(z)
        z = torch.cat([z, a], 1)
        return self.qf(z)
    
    # def _update(self, qf):
    #     for param, value_param in zip(qf.parameters(), self.qf.parameters()):
    #         value_param.data.copy_(param.data)
    #     return


class Actor(nn.Module):
    def __init__(self, input_size:int=5, action_size:int=2, action_high=np.array([ 10.0, 0.0]), action_low=np.array([ 0.5, -8.5])):
        super().__init__()

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_size)
        self.fc_logstd = nn.Linear(64, action_size)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32).unsqueeze(0)
        )


    def forward(self, x):
        x = self._combine(x)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def _combine(self, x):

        x = torch.cat([x["observation"], x["desired_goal"] - x["achieved_goal"]], 1)
        # x = torch.cat([x["observation"], x["desired_goal"]], 1)

        return x


    def _explore_noise(self, x):

        x = {"observation": torch.Tensor(x["observation"]), "desired_goal": torch.Tensor(x["desired_goal"])}
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean.zero_(), std)
        noise = normal.rsample() 

        return noise

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_deterministic_action(self, x):
        # also assume x is combined (for passing to do-mpc)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        mean = self.fc_mean(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean