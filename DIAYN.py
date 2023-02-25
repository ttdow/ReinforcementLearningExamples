import gym
import numpy as np
import random
from collections import namedtuple
from abc import ABC

import torch
from torch import from_numpy
from torch import nn
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from torch.distributions import Normal

Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state'))

class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        random.seed(self.seed)

    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        assert len(self.buffer) <= self.buffer_size

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)
    
    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions[0]
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()
    
    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)

        # Reparameterization trick ???
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)

        # Enforcing action bounds
        log_prob -= torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob

class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions[0]
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)

class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)

class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits

class SACAgent:
    def __init__(self, p_z, **config):
        self.config = config
        self.n_states = self.config['n_states'][0]
        self.n_skills = self.config['n_skills']
        self.batch_size = self.config['batch_size']
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config['seed'])
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills, 
                                            n_actions=self.config['n_actions'],
                                            action_bounds=self.config['action_bounds'],
                                            n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config['n_actions'],
                                              n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config['n_actions'],
                                              n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

        self.discriminator = Discriminator(n_states=self.n_states,
                                           n_skills=self.n_skills,
                                           n_hidden_filters=self.config['n_hiddens']).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config['lr'])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config['lr'])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config['lr'])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config['lr'])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config['lr'])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to('cpu')
        z = torch.ByteTensor([z]).to('cpu')
        done = torch.BoolTensor([done]).to('cpu')
        action = torch.Tensor([action]).to('cpu')
        next_state = from_numpy(next_state).float().to('cpu')
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config['n_actions']).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config['alpha'] * log_probs.detach()
            
            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

params = {"lr": 3e-4,
          "batch_size": 256,
          "max_n_episodes": 5000,
          "max_episode_len": 1000,
          "gamma": 0.99,
          "alpha": 0.1,
          "tau": 0.005,
          "n_hiddens": 300,
          "seed": 0,
          "n_skills": 50,
          "mem_size": 50000}

env = gym.make('MountainCarContinuous-v0')
n_states = env.observation_space.shape
n_actions = env.action_space.shape
action_bounds = [env.observation_space.low[0], env.action_space.high[0]]
print(n_states, ",", n_actions, ",", action_bounds)

params.update({"n_states": n_states,
               "n_actions": n_actions,
               "action_bounds": action_bounds})

print("params:", params)
env.close()
del env, n_states, n_actions, action_bounds

env = gym.make('MountainCarContinuous-v0')

p_z = np.full(params['n_skills'], 1 / params['n_skills'])
agent = SACAgent(p_z=p_z, **params)
#logger = Logger(agent, **params)

min_episode = 0
last_logq_zs = 0
np.random.seed(params['seed'])
env.observation_space.seed(params['seed'])

#logger.on()

for episode in range(1 + min_episode, params['max_n_episodes'] + 1):
    z = np.random.choice(params['n_skills'])
    state = env.reset()
    state = state[0]
    state = concat_state_latent(state, z, params['n_skills']) # Concats one-hot encoding of skills to state
    episode_reward = 0
    logq_zses = []

    max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)

    for step in range(1, 1 + max_n_steps):
        action = agent.choose_action(state)
        next_state, reward, done, trunc, info = env.step(action)
        next_state = concat_state_latent(next_state, z, params['n_skills'])
        agent.store(state, z, done, action, next_state)
        logq_zs = agent.train()
        if logq_zs is None:
            logq_zses.append(last_logq_zs)
        else:
