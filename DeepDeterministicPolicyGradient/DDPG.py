import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

class Replay_Buffer(object):

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    # Add the transition to memory.
    def add(self, state, action, next_state, reward, done):

        # Save the transition data.
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        # Increment memory pointer or loop if at memory end.
        self.ptr = (self.ptr + 1) % self.max_size

        # Increment memory size if not already full.
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):

        # Get random batch_size indexes from memory.
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Return as (state, action, next_state, reward, done).
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
    
class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        action = self.max_action * torch.tanh(x)

        return action
    
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state, action), 1)))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value

class DDPG(object):
    
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
        
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = Replay_Buffer(state_dim, action_dim)

        self.discount = discount
        self.tau = tau

        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        return action
    
    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * local_param.data)

    def save(self):
        torch.save(self.actor.state_dict(), "./actor.pt")
        torch.save(self.actor_optimizer.state_dict(), "./actor_optim.pt")
        torch.save(self.critic.state_dict(), "./critic.pt")
        torch.save(self.critic_optimizer.state_dict(), "./critic_optim.pt")

    def load(self):
        self.actor.load_state_dict(torch.load("./actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load("./actor_optim.pt"))
        self.critic.load_state_dict(torch.load("./critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load("./critic_optim.pt"))

    def train(self, batch_size=100):

        # Get transition samples.
        state, action, next_state, reward, done = self.memory.sample(batch_size)

        # Use samples to determine target Q-values.
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (done * self.discount * target_q).detach()

        # Use samples to compute current Q-values.
        current_q = self.critic(state, action)

        # Critic loss is MSE of current and target Q-values.
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic network.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss is -mean of critic predictions.
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update actor network.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Perform soft update for target networks.
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

def train(env, agent, time_steps, start_time_step, expl_noise, batch_size):
    
    state = env.reset()

    if type(state) == tuple:
            state = state[0]

    done = False
    trunc = False

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    evaluations = []
    episode_rewards = []

    for ts in range(1, int(time_steps+1)):
        
        episode_timesteps += 1

        if ts < start_time_step:
            action = env.action_space.sample()
        else:
            noise = np.random.normal(0, agent.max_action * expl_noise * ((time_steps - ts + 2) / time_steps), size=agent.action_dim)
            action = agent.select_action(np.array(state))
            action = action + noise
            action = action.clip(-agent.max_action, agent.max_action)

        next_state, reward, done, trunc, info = env.step(action)

        agent.memory.add(state, action, next_state, reward, done)

        state = next_state

        episode_reward += reward

        if ts >= start_time_step:
            agent.train(batch_size)

        if done or trunc:
            episode_rewards.append(episode_reward)
            state = env.reset()

            if type(state) == tuple:
                state = state[0]

            done = False
            trunc = False
            episode_num += 1

            if episode_num % 10 == 0 and episode_num > 0:
                print("Episode {} - Reward: {:0.2f}, Timesteps: {}".format(episode_num, episode_reward, episode_timesteps))
                agent.save()

            episode_reward = 0
            episode_timesteps = 0

    plt.plot(episode_rewards)
    plt.show()

def test(agent):
    
    test_episodes = 100

    rewards = []

    for episode in range(test_episodes):

        total_reward = 0

        state = env.reset()[0]

        while True:
            action = agent.select_action(state)
            next_state, reward, done, trunc, _ = env.step(np.float32(action))
            total_reward += reward

            if done or trunc:
                print("Episode: {} Total Reward: {:0.2f}".format(episode, total_reward))
                break

            state = next_state

        rewards.append(total_reward)

    plt.plot(rewards)
    plt.show()

# Setup learning environment.
env = gym.make("Pendulum-v1", g=9.81)#, render_mode="human")

# Get environment details.
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Agent hyperparameters.
time_steps = 1e6
start_time_step = 25e3
expl_noise = 0.25
batch_size = 256
evaluate_frequency = 5e3

# Specify tensor device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create the Deep Deterministic Policy Gradient agent.
agent = DDPG(state_dim, action_dim, max_action, device)
#agent.load()

train(env, agent, time_steps, start_time_step, expl_noise, batch_size)
#test(agent)

env.close()