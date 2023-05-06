import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import gym

class ActorCritic(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic_linear1 = nn.Linear(input_size, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(input_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, output_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):

        if type(x) == tuple:
            x = x[0]

        x = Variable(torch.from_numpy(x).float().unsqueeze(0))

        # Actor branch returns probability of selecting each action
        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=-1)

        # Critic branch returns a categorization of state (i.e. good/bad)
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)

        return policy_dist, value
    
# Create environment
env = gym.make('CartPole-v1')

# Set hyperparameters
hidden_size = 256
lr = 3e-4
gamma = 0.99
num_episodes = 3000
num_steps = 300

# Create RL agent
actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_size)#, lr, gamma)
ac_optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

# Loggables
all_lengths = []
average_lengths = []
all_rewards = []
entropy_term = 0

for episode in range(num_episodes):

    log_probs = []
    values = []
    rewards = []

    state = env.reset()
    done = False

    # Step through an episode
    for steps in range(num_steps):

        policy_dist, value = actor_critic.forward(state)
        value = value.detach().numpy()[0, 0]
        dist = policy_dist.detach().numpy()

        action = np.random.choice(2, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))
        new_state, reward, done, _, _ = env.step(action)

        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        entropy_term += entropy
        state = new_state

        if done or steps == num_steps-1:
            Qval, _ = actor_critic.forward(new_state)
            Qval = Qval.detach().numpy()[0, 0]
            all_rewards.append(np.sum(rewards))
            all_lengths.append(steps)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 10 == 0:
                print("Episode: " + str(episode) + " Reward: " + str(np.sum(rewards)) + " Steps: " + str(steps) + " Average Length: " + str(average_lengths[-1]))
            break

    # Compute Q-values
    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + gamma * Qval
        Qvals[t] = Qval

    # Update actor-critic
    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(log_probs)

    advantage = Qvals - values
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()

    ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()

# Plot results
smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]

plt.plot(all_rewards)
plt.plot(smoothed_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.plot(all_lengths)
plt.plot(average_lengths)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.show()