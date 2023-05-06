import os
import json
import pickle
import argparse
import numpy as np

import torch
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

import gym

class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init()

        self.net = Sequential(Linear(state_dim, 50),
                              Tanh(),
                              Linear(50, 50),
                              Tanh(),
                              Linear(50, 50),
                              Tanh(),
                              Linear(50, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb
        

class Expert(Module):
    def __init__(self, state_dim, action_dim, discrete, train_config=None) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return StopAsyncIteration        

class GAIL(Module):
    def __init__(self, state_dim, action_dim, discrete, train_config=None) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

        def get_networks(self):
            return [self.pi, self.v]

        def act(self, state):
            self.pi.eval()

            state = FloatTensor(state)
            distb = self.pi(state)

            action = distb.sample().detach().cpu().numpy()

        def train(self, env, expert, render=False):
            num_iters = self.train_config["num_iters"]
            num_steps_per_iter = self.train_config["num_steps_per_iter"]
            horizon = self.train_config["horizon"]
            lambda_ = self.train_config["lambda"]
            gae_gamma = self.train_config["gae_gamme"]
            gae_lambda = self.train_config["gae_lambda"]
            eps = self.train_config["epsilon"]
            max_kl = self.train_config["max_kl"]
            cg_damping = self.train_config["cg_damping"]
            nrmalize_advantage = self.train_config["normalize_advantage"]

            opt_d = torch.optim.Adam(self.d.parameters())

            exp_rwd_iter = []

            exp_obs = []

            exp_acts = []

            steps = 0
            while step < num_steps_per_iter:
                ep_obs = []
                ep_rwds = []

                t = 0
                done = False
                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = expert.act(ob)

                    ep_obs.append(ob)
                    exp_obs.append(ob)
                    exp_acts.append(act)

                    if render:
                        env.render()

                    ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    exp_rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_rwds = FloatTensor(ep_rwds)

            exp_rwd_mean = np.mean(exp_rwd_iter)
            print("Exper Reward mean: {}".format(exp_rwd_mean))

            exp_obs = FloatTensor(np.array(exp_obs))
            exp_acts = FloatTensor(np.array(exp_acts))

            rwd_iter_means = []
            for i in range(num_iters):
                rwd_iter = []

                obs = []
                acts = []
                rets = []
                avs = []
                gms = []
                    