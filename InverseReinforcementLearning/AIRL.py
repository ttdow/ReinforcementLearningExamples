import numpy as np
import os
import time
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import gym
from gym.spaces import Discrete

from pathlib import Path

# Returns a count of the parameters in a module.
def count(module):
    return np.sum([np.prod(p.shape) for p in module.parameters()])

def mlp(x, hidden_layers, activation=nn.Tanh, size=2, output_activation=nn.Identity()):
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)

        # For discriminator
        if activation.__name__ == 'ReLU':
            net_layers.append(activation(inplace=True))
        elif activation.__name__ == 'LeakyReLU':
            net_layers.append(activation(0.2, inplace=True))
        else:
            net_layers.append(activation())
        x = size

    net_layers.append(nn.Linear(x, hidden_layers[-1]))
    net_layers += [nn.Identity()]

    return nn.Sequential(*net_layers)

class Actor(nn.Module):
    def __init__(self, **args):
        super(Actor, self).__init__()

    # Gives policy for given observations and optionally actions log prob under
    #   that policy.
    def forward(self, obs, ac=None):
        pi = self.sample_policy(obs)
        log_p = None

        if isinstance(self, CategoricalPolicy):
            ac = ac.unsqueeze(-1)

        if ac is not None:
            log_p = self.log_p(pi, ac)
        return pi, log_p

class Discriminator(nn.Module):
    # Discriminates between expert data and samples from learned policy. It
    #   recovers the advantage f_theta_pi used in training the policy.
    # The discriminator has:
    #   g_theta(s): A state-only dependant reward function. This allows
    #       extraction of rewards that are disentangled from the dynamics
    #       of the environment in which they were trained.
    #   h_phi(s): The shaping term. Mitigates unwanted shaping on the reward
    #       term of g_theta(s)
    #   f_theta_pi = g_theta(s) + gamma * h_phi(s) - h_phi(s) (i.e. advantage)
    def __init__(self, obs_dim, gamma=0.99, **args):
        super(Discriminator, self).__init__()
        self.gamma = gamma

        # *g(s) = *r(s) + const
        #  g(s) recovers the optimal reward function + c
        self.g_theta = mlp(obs_dim, hidden_layers=[32, 1])#, **args['g_args'])

        # *h(s) = *V(s) + const 
        #  h(s) recovers optimal value function + c)
        self.h_phi = mlp(obs_dim, hidden_layers=[32, 32, 1])# **args['h_args'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *data):
        # Returns the estimated reward function/advantage estimate. Given by:
        #   f(s, a, s') = g(s) + gamma * h(s') - h(s)
        # data = [obs, obs_n, dones]

        obs, obs_n, dones = data
        g_s = torch.squeeze(self.g_theta(obs), axis=-1)

        shaping_term = self.gamma * (1 - dones) * self.h_phi(obs_n).squeeze() - self.h_phi(obs).squeeze(-1)

        f_theta_phi = g_s + shaping_term

        return f_theta_phi

    def discr_value(self, log_p, *data):
        # Calculates the discriminator output
        #   D = exp(f(s, a, s')) / [exp(f(s, a, s')) + pi(a|s)]

        adv = self(*data)
        exp_adv = torch.exp(adv)
        value = exp_adv / (exp_adv + torch.exp(log_p) + 1e-8)

        return self.sigmoid(value)

class CategoricalPolicy(Actor):
    # Categorical policy for discrete action space
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, size=2):
        super(CategoricalPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes + [act_dim], activation, size=size)

    def sample_policy(self, obs):
        # Get new policy
        logits = self.logits(obs)
        pi = torch.distributions.Categorical(logits=logits)
        return pi

    @classmethod
    def log_p(cls, p, a):
        # Log probabilities of actions wrt pi
        return p.log_prob(a)

# Multilayer Perceptron agent actor neural net
class MLPActor(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size=[32, 32], activation=nn.Tanh, size=2, **args):
        super(MLPActor, self).__init__()

        self.obs_dim = obs_space.shape[0]

        self.discrete = True if isinstance(act_space, Discrete) else False
        self.act_dim = act_space.n if self.discrete else act_space.shape[0]

        #if self.discrete:
        self.pi = CategoricalPolicy(self.obs_dim, self.act_dim, hidden_size, size=size, activation=activation)
        #else:
        #self.pi = MLPGaussianPolicy(self.obs_dim, self.act_dim, hidden_size, size=size, activation=activation)

        self.disc = Discriminator(self.obs_dim, **args)

    # Get distribution under current obs and action sample from pi
    def step(self, obs):
        with torch.no_grad():
            pi_new = self.pi.sample_policy(obs)
            a = pi_new.sample()

            log_p = self.pi.log_p(pi_new, a)
        
        return a.numpy(), log_p.numpy()

# Transitions buffer. Stores tansitions for a single episode.
class ReplayBuffer:
    def __init__(self, act_dim, obs_dim, size=10000, expert_data_path='', expert_buffer=None):
        self.rewards = np.zeros([size], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.n_states = np.zeros([size, obs_dim], dtype=np.float32)

        self.dones = np.zeros([size], dtype=np.float32)
        self.log_prob = np.zeros([size], dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

        expert_data = Path("data/airl_expert_data.npz")
        self.expt_buff = ExpertBuffer(expert_data)

    # Store transition
    def store(self, act, states, n_states, rew, dones, log_p):
        idx = self.ptr % self.max_size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.n_states[idx] = n_states
        self.log_prob[idx] = log_p
        self.dones[idx] = dones

        self.ptr += 1
        self.size = min(self.ptr + 1, self.max_size)

    # Returns recent transitions of size batch_size.
    def sample_recent(self, batch_size):
        assert self.ptr >= batch_size

        return (torch.as_tensor(self.actions[-batch_size:], dtype=torch.float32),
                torch.as_tensor(self.rewards[-batch_size:], dtype=torch.float32),
                torch.as_tensor(self.states[-batch_size:], dtype=torch.float32),
                torch.as_tensor(self.n_states[-batch_size:], dtype=torch.float32),
                torch.as_tensor(self.dones[-batch_size:], dtype=torch.float32),
                torch.as_tensor(self.log_prob[-batch_size:], dtype=torch.float32))

    # Randomly sample trajectories upto the most recent itr_limit iterations.
    def sample_random(self, batch_size, itr_limit=20):
        lowest_itr = itr_limit * batch_size # Batch size makes 1 iter
        low_ = 0
        if self.ptr > lowest_itr:
            low_ = lowest_itr
        idx = np.random.randint(low=self.ptr - low_, high=self.ptr, size=batch_size)

        return (torch.as_tensor(self.actions[idx], dtype=torch.float32),
                torch.as_tensor(self.rewards[idx], dtype=torch.float32),
                torch.as_tensor(self.states[idx], dtype=torch.float32),
                torch.as_tensor(self.n_states[idx], dtype=torch.float32),
                torch.as_tensor(self.dones[idx], dtype=torch.float32),
                torch.as_tensor(self.log_prob[idx], dtype=torch.float32))

# Expert demonstrations buffer. 
# Loading the data works with files written using:
#   np.savez(file_path, **{key: value})
# So the loaded file can be accessed again by key:
#   value = np.load(file_path)[key]
# Data format: { 'iteration_count': transitions }
#   where, trasitions = key, value pair of expert data samples of size(-1) 'steps_per_epoch' 
class ExpertBuffer():
    def __init__(self, path):
        data_file = np.load(path, allow_pickle=True)
        self.load_rollouts(data_file)

        self.ptr = 0

    # Convert a list of rollout dictionaries into seperate arrays concatenated
    #   across the arrays rollout.
    def load_rollouts(self, data_file):
        # Get all iteration's transitions
        data = [traj for traj in data_file.values()]

        try:
            # Traj in x batch arrays. Unroll to 1.
            data = np.concatenate(data)
        except ValueError:
            data = [d[None] for d in data]
            data = np.concatenate(data)

        self.obs = np.concatenate([path['observation'] for path in data])
        self.obs_n = np.concatenate([path['next_observation'] for path in data])
        self.dones = np.concatenate([path['terminal'] for path in data])

        self.size = self.dones.shape[0]

    # Fetch random expert demonstrations of size batch_size.
    def get_random(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)

        return (torch.as_tensor(self.obs[idx], dtype=torch.float32),
                torch.as_tensor(self.obs_n[idx], dtype=torch.float32),
                torch.as_tensor(self.dones[idx], dtype=torch.float32))

    # Samples expert trajectories by order of saved iterations.
    def get(self, batch_size):
        if self.ptr + batch_size > self.size:
            self.ptr = 0
        
        idx = slice(self.ptr, self.ptr + batch_size)

        return (torch.as_tensor(self.obs[idx], dtype=torch.float32),
                torch.as_tensor(self.obs_n[idx], dtype=torch.float32),
                torch.as_tensor(self.dones[idx], dtype=torch.float32))

def AIRL(env, n_epochs, steps_per_epoch, max_eps_length, clip_ratio, entropy_reg):#, **args):
    # entropy_reg (float) = Entropy regularizer temperature for Soft PPO (SPPO)
    #   Higher values of temperature encourage stochasticity in the policy while
    #   lower values make it more deterministic.
    # clip_ratio (float) = Clips the old policy objective. Determines how far
    #   the new policy can go from the old policy while still improving the
    #   objective.
    # max_kl (float) = KL divergence regulator. Used for early stopping when
    #   the KL between the new and old policy exceeds this threshold we think
    #   is appropriate (0.01 - 0.05)
    # pi_lr (float) = Learning rate for the policy
    # disc_lr (float) = Learning rate for the discriminator
    # seed (int) = Random seed generator
    # real_label (int) = Label for expert data
    # pi_label (int) = Label for policy samples
    # expert_data_path (str) = Path to expert demonstrations.
    # kl_start (int) = Epoch at which to start checking the kl divergence
    #   between the old learned policy and new learned policy. Kl starts
    #   high (> 1) and drastically diminishes to below 0.1 as the policy
    #   learns.

    torch.manual_seed(0)
    np.random.seed(0)

    obs_space = env.observation_space
    act_space = env.action_space

    act_dim = act_space.shape[0] if not isinstance(act_space, gym.spaces.Discrete) else act_space.n
    obs_dim = obs_space.shape[0]

    hidden_size = [64, 64]
    size = 2

    actor = MLPActor(obs_space=obs_space, act_space=act_space)

    params = [count(module) for module in (actor.pi, actor.disc, actor.disc.g_theta, actor.disc.h_phi)]

    memory = ReplayBuffer(act_dim, obs_dim, size=int(1e6), expert_data_path='', expert_buffer=ExpertBuffer)

    pi_optimizer = optim.Adam(actor.pi.parameters(), 2e-4, betas=[0.5, 0.9])

    discr_optimizer = optim.Adam(actor.disc.parameters(), 1e-4, betas=[0.5, 0.9])

    loss_criterion = nn.BCELoss()

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join('data', env.unwrapped.spec.id + '_' + run_t)

    logger = SummaryWriter(log_dir=path)

    # Hold epoch losses for logging
    pi_losses, disc_losses, delta_disc_logs, delta_pi_logs = [], [], [], []
    pi_kl, entropy_logs = [], []
    disc_logs = []
    disc_outputs = [] # Discriminator predictions
    err_demo_logs, err_sample_logs = [], []
    first_run_ret = None

    # Pi loss. Uses expert demonstrations to find advantage estimate from the
    #   learned reward function.
    def compute_pi_loss(log_p_old, act_b, *expert_demos):
        obs_b, obs_n_b, dones_b = expert_demos

        # Returns new_pi normal_distribution, logp_act
        pi_new, log_p = actor.pi(obs_b, act_b)
        log_p_ = log_p.type(torch.float32)

        # Predict advantage using learned reward function
        # r_t_hat(s, a) = f(s, a) - log(pi(a|s))
        # r_t_hat(s, a) = A(s, a)
        adv_b = actor.disc(obs_b, obs_n_b, dones_b) - log_p_old

        adv_b = (adv_b - adv_b.mean()) / adv_b.std()

        pi_diff = log_p_ - log_p_old

        pi_ratio = torch.exp(pi_diff)

        # Soft PPO update - encourage entropy in the policy
        #   i.e. Act as randomly as possible while maximizing objective
        #   Example case: pi might learn to take a certain action for a given
        #       state every time because it has some good reward, but forgo, 
        #       trying other actions which might have higher reward.

        # A_old_pi(s, a) = A(s, a) - entropy_reg * log(pi_old(a, s))
        adv_b = adv_b - (entropy_reg * log_p_old)

        min_adv = torch.where(adv_b >= 0 , (1 + clip_ratio) * adv_b, (1 - clip_ratio) * adv_b)

        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))
        kl = -(pi_diff).mean().item()
        entropy = pi_new.entropy().mean().item()

        return pi_loss, kl, entropy

    # Discriminator loss
    #   log(D_theta_phi(s, a, s') - log(1 - D_theta_pi(s, a, s')) (Eq. 1)
    #   Minimize the likelihood of policy samples while increase likelihood of
    #       expert demonstations.
    #   D_theta_pi = exp(f(s, a, s')) / [exp(f(s, a, s')) + pi(a|s)]
    #   Substitute this in Eq. 1:
    #       = f(s, a, s') - log(p(a|s))
    def compute_disc_loss(*traj, log_p, label):
        output = actor.disc.discr_value(log_p, *traj).view(-1)

        err_d = loss_criterion(output, label)

        # Average output across the batch of the Discriminator
        # For expert data, should start at ~1 and converge to 0.5
        # For sample data, should start at 0 and converge to 0.5
        d_x = output

        # Call err_demo.backward first
        return d_x, err_d

    # Performs gradient update on pi and discriminator
    def update(epoch):
        batch_size = steps_per_epoch
        real_label = 1
        pi_label = 0

        data = memory.sample_recent(batch_size)
        act, rew, obs, obs_n, done, log_p = data
        sample_disc_data = data[2:-1]

        random_demos = False
        exp_data = memory.expt_buff.get(batch_size) if not random_demos else memory.expt_buff.get_random(batch_size)

        #print(exp_data[0])

        # Loss before update
        pi_loss_old, kl, entropy = compute_pi_loss(log_p, act, *exp_data)

        label = torch.full((batch_size, ), real_label, dtype=torch.float32)

        demo_info = compute_disc_loss(*exp_data, log_p=log_p, label=label)
        pi_samples_info = compute_disc_loss(*sample_disc_data, log_p=log_p, label=label.fill_(pi_label))

        _, err_demo_old = demo_info
        _, err_pi_samples_old = pi_samples_info

        disc_loss_old = (err_demo_old + err_pi_samples_old).mean().item()

        # Train with expert demonstrations
        #   log(D(s, a, s'))
        for i in range(5): #40
            print("Training discriminator: ", i+1, "/5")
            actor.disc.zero_grad()

            av_demo_output, err_demo = compute_disc_loss(*exp_data, log_p=log_p, label=label.fill_(real_label))

            err_demo.backward()

            # Train with policy samples
            #   -log(D(s, a, s'))
            label.fill_(pi_label)
            av_pi_output, err_pi_samples = compute_disc_loss(*sample_disc_data, log_p=log_p, label=label)

            err_pi_samples = -err_pi_samples
            err_pi_samples.backward()
            loss = err_demo + err_pi_samples

            discr_optimizer.step()

        av_pi_output = av_pi_output.mean().item()
        av_demo_output = av_demo_output.mean().item()
        err_demo = err_demo.item()
        err_pi_samples = err_pi_samples.item()
        disc_loss = loss.item()

        kl_start = epoch >= 20

        for i in range(5): #80
            print("Training policy: ", i+1, "/5")
            pi_optimizer.zero_grad()

            pi_loss, kl, entropy = compute_pi_loss(log_p, act, *exp_data)
            if kl_start and kl > 1.5 * 1.0:
                print('Max kl reached: ', kl, ' iter:', i)
                break

            pi_loss.backward()
            pi_optimizer.step()

        logger.add_scalar('PiStopIter', i, epoch)

        pi_loss = pi_loss.item()

        pi_losses.append(pi_loss)
        pi_kl.append(kl)
        disc_logs.append(disc_loss)
        disc_outputs.append((av_demo_output, av_pi_output))

        delta_disc_loss = disc_loss_old - disc_loss
        delta_pi_loss = pi_loss_old.item() - pi_loss

        delta_disc_logs.append(delta_disc_loss)
        delta_pi_logs.append(delta_pi_loss)
        err_demo_logs.append(err_demo)
        err_sample_logs.append(err_pi_samples)
        entropy_logs.append(entropy)

        logger.add_scalar('loss/pi', pi_loss, epoch)
        logger.add_scalar('loss/D', disc_loss, epoch)
        logger.add_scalar('loss/D[demo]', err_demo, epoch)
        logger.add_scalar('loss/D[pi]', err_pi_samples, epoch)

        logger.add_scalar('loss/Delta-Pi', delta_pi_loss, epoch)
        logger.add_scalar('loss/Delta-Disc', delta_disc_loss, epoch)

        logger.add_scalar('Disc-Output/Expert', av_demo_output, epoch)
        logger.add_scalar('Disc-Output/LearnedPolicy', av_pi_output, epoch)

        logger.add_scalar('Kl', kl, epoch)
        logger.add_scalar('Entropy', entropy, epoch)

    start_time = time.time()

    obs = env.reset()
    eps_len = 0
    eps_ret = 0

    n_epochs = 5
    for t in range(n_epochs):
        print("Epochs: ", t+1, "/", n_epochs)
        eps_len_logs = []
        eps_ret_logs = []

        if t == 0:
            obs = obs[0]

        for step in range(steps_per_epoch):
            a, log_p = actor.step(torch.as_tensor(obs, dtype=torch.float32))

            obs_n, rew, done, _, _ = env.step(a)

            eps_len += 1
            eps_ret += rew

            memory.store(a, obs, obs_n, rew, done, log_p)
            obs = obs_n

            terminal = done or eps_len == max_eps_len

            if terminal or step == steps_per_epoch - 1:
                if terminal:
                    eps_len_logs += [eps_len]
                    eps_ret_logs += [eps_ret]

                obs = env.reset()
                obs = obs[0]
                eps_len = 0
                eps_ret = 0

        update(t + 1)

        # Logs
        l_t = t + 1

        RunTime = time.time() - start_time
        AverageEpisodeLen = np.mean(eps_len_logs)

        logger.add_scalar('AvEpsLen', AverageEpisodeLen, l_t)
        MaxEpisodeLen = np.max(eps_len_logs)
        MinEpisodeLen = np.min(eps_len_logs)
        AverageEpsReturn = np.mean(eps_ret_logs)
        MaxEpsReturn = np.max(eps_ret_logs)
        MinEpsReturn = np.min(eps_ret_logs)

        logger.add_scalar('EpsReturn/Max', MaxEpsReturn, l_t)
        logger.add_scalar('EpsReturn/Min', MinEpsReturn, l_t)
        logger.add_scalar('EpsReturn/Average', AverageEpsReturn, l_t)

        # Retrieved by index, not the time step
        Pi_Loss = pi_losses[t]
        Disc_Loss = disc_logs[t]
        Kl = pi_kl[t]
        delta_disc_loss = delta_disc_logs[t]
        delta_pi_loss = delta_pi_logs[t]
        disc_outs = disc_outputs[t]

        if t == 0:
            first_run_ret = AverageEpsReturn

        all_logs = {
            'AverageEpsReturn': AverageEpsReturn,
            'MinEpsReturn': MinEpsReturn,
            'MaxEpsReturn': MaxEpsReturn,
            'KL': Kl,
            'Entropy': entropy_logs[t],
            'AverageEpisodeLen': AverageEpisodeLen,
            'Pi_Loss': Pi_Loss,
            'Disc_Loss': Disc_Loss,
            'FirstEpochAvReturn': first_run_ret,
            'Delta-Pi': delta_pi_loss,
            'Delta-D': delta_disc_loss,
            'Disc-DemoLoss': err_demo_logs[t],
            'Disc-SamplesLoss': err_sample_logs[t],
            'AvDisc-Demo-Output': disc_outs[0],
            'AvDisc-PiSamples-Output': disc_outs[1],
            'RunTime': RunTime
        }

        print('\n', t+1)
        print('-' * 35)

        for k, v in all_logs.items():
            print(k, v)

        print('\n\n\n')

    print("Finished")

n_epochs=50
steps_per_epoch=5000
max_eps_len=1000
clip_ratio=0.2
entropy_reg=0.1

env = gym.make('MountainCar-v0')
AIRL(env, n_epochs, steps_per_epoch, max_eps_len, clip_ratio, entropy_reg)