import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym

'''
# Define the policy network
class Policy(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
    
# Define the policy gradient algorithm
def policy_gradient(env_name, hidden_size, learning_rate, num_episodes):

    # Create the environment
    env = gym.make(env_name)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create the policy network
    policy = Policy(obs_size, action_size, hidden_size)

    # Create the optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Iterate over episodes
    for episode in range(num_episodes):
        
        # Initialize episode
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        log_probs = []
        rewards = []

        # Iterate over timesteps in episode
        while not done:

            # Choose an action based on the policy
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action_probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # Take the chosen action and observe the next state and reward
            obs, reward, done, info, _ = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward

        # Compute the discounted rewards and normalize them
        returns = []
        discount = 0.99
        c = 0

        if len(returns) > 0:
            c = returns[-1] * discount

        for i in rewards:
            i = i + c
        
        returns.append(rewards)
        
        returns.reverse()
        returns = torch.tensor(returns).float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Compute the loss and update the policy
        policy_loss = torch.stack(log_probs) * returns.unsqueeze(1)
        policy_loss = -policy_loss.mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Print the episode reward
        print('Episode: %d, Reward: %d' % (episode, episode_reward))

# Test the algorithm on the CartPole-v1 environment
policy_gradient('CartPole-v1', 32, 0.01, 1000)
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReinforcePolicy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(ReinforcePolicy, self).__init__()

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.softmax(x, dim=-1)

        return y

    # Given a state, take an action
    def act(self, state):
        
        if type(state) == tuple:
            state = state[0]

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)
    
def reinforce(policy: ReinforcePolicy, optimizer, n_training_episodes, max_t, gamma, print_every):

    # Help us to calculate the score during the training.
    scores_deque = deque(maxlen=100)
    scores = []

    # Iteratively improve policy pi_theta
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()

        # Generate an episode S_0, A_0, r_0, ..., S_T-1, A_T-1, r_T-1 as per pi_theta
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)

            rewards.append(reward)

            if done:
                break

        cum_sum = sum(rewards)
        scores_deque.append(cum_sum)
        scores.append(cum_sum)

        # Calculate return R(tau)
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        # Compute the discounted returns at each timestep
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Policy Gradient Theorem
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        
        policy_loss = torch.cat(policy_loss).sum()

        # Optimize using gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

# Evaluate the agent and return average reward and std of reward.
def evaluate_agent(env, max_steps, n_eval_episodes, policy):

    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
        
# Create the environment
env_id = "CartPole-v1"
env = gym.make(env_id)

# Create the evaluation environment (?)
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0] # 4
a_size = env.action_space.n             # 2

# Define hyperparameters
cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size
}

# Create policy and place it to the device
cartpole_policy = ReinforcePolicy(cartpole_hyperparameters['state_space'],
                                  cartpole_hyperparameters['action_space'],
                                  cartpole_hyperparameters['h_size']).to(device)

cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters['lr'])

checkpoint = torch.load('reinforce_cartpole.pt')
cartpole_policy.load_state_dict(checkpoint['model_state_dict'])
cartpole_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Run training
scores = reinforce(cartpole_policy,
                   cartpole_optimizer,
                   cartpole_hyperparameters['n_training_episodes'],
                   cartpole_hyperparameters['max_t'],
                   cartpole_hyperparameters['gamma'],
                   100)

torch.save({'model_state_dict': cartpole_policy.state_dict(),
            'optimizer_state_dict': cartpole_optimizer.state_dict()
           }, 'reinforce_cartpole.pt')

# Evaluate agent
eval = evaluate_agent(eval_env,
               cartpole_hyperparameters['max_t'],
               cartpole_hyperparameters['n_evaluation_episodes'],
               cartpole_policy)

print(eval)