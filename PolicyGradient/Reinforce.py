# Policy-based methods aim to optimize the policy directly without
#   having an intermediate step of learning a value function.

# Policy-gradient is a subclass of policy-based methods, a category
#   of algorithms that aims to optimize the policy directly without
#   using a value function using different techniques.

# Policy-gradient aims to optimize the policy directly by estimating
#   the weights of the optimal policy using gradient ascent.

# A policy is a function that given a state, outputs a distribution
#   over actions.

# Stochastic: output a probability distribution over actions
#   pi(a|s) = P[A|s], where A = action space

# Our goal with policy-gradients is to control the probability 
#   distribution of actions by tuning the policy such that good
#   actions (that maximize the return) are sampled more frequently
#   in the future.

# Psuedocode:
# Training Loop:
#   Collect an episode with the pi (policy)
#   Calculate the return (sum of rewards)
#   Update the weights of the pi:
#       if positive return -> increase the probability of each (state, action)
#           pairs taken during the episode
#       if negative return -> decrease the probability of each (state, action)
#           pairs taken during the episode

# Reinforce (Monte Carlo Policy Gradient)
#   pi_theta(a | s) = P[a|s;theta], where: theta = policy parameter
#       pi_theta(a_t | s_t) = probability of the agent selecting action a_t from state s_t

# How do we know the policy is good? Score/objective function:
#   J(theta) = E_tau~pi[R(tau)], where R(tau) = return/cumulative reward
#       and tau = trajectory = sequence of states and actions
#   R(tau) = r_t+1 + gamma*r_t+2 + (gamma**2)*r_t+3 + ...

# Policy gradient:
#   grad_theta * J(theta) = sum(grad_theta * log(pi_theta(a_t | s_t) * R(tau)))

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
#import gym_pygame

import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the environment
env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")
env.reset()
env.render()

# Create the evaluation environment
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("Observation space:")
print("The state space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("Action space:")
print("The action space is: ", a_size)
print("Action space sample", env.action_space.sample()) # Take a random action

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        # Create two fully connected layers
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    # Define the forward pass
    def forward(self, x):
        # state goes to fc1 then we apply ReLU activation function
        x = F.relu(self.fc1(x))

        # fc1 output goes to fc2
        x = self.fc2(x)

        # We output the softmax
        return F.softmax(x, dim=1)

    def act(self, state):
        # Given a state, take action
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

#debug_policy = Policy(s_size, a_size, 64).to(device)
#out = debug_policy.act(env.reset()[0])

#  Reinforce algorithm:
#    Start with policy model pi_theta
#3   repeat:
#4      Generate an episode S_0, A_0, r_0, ..., S_T-1, A_T-1, r_T-1 following pi_theta(.)
#5      for t from T - 1 to 0:
#6        G_t = sum(gamma**(k-t) * r_k)
#7      L(theta) = 1/T * sum(G_t * log(pi_theta(A_t | S_t)))
#8      Optimize pi_theta using grad(L(theta))
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of psuedocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]

        # Line 4 of psuedocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        # Compute the discounted returns at each timestep, as the sum of the gamma-discounted
        #   return at time t (G_t) + the reward at time t
        # In O(N) time, where N is the number of time steps
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma * disc_return_t * rewards[t])

        # Standardization of returns to make training more stable
        eps = np.finfo(np.float32).eps.item()
        # eps is the smallest representable float which is added to the stdev of the returns
        #   to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7 in psuedocode
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8 in psuedocode
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    
    return scores

cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size
}

# Create a policy and place it to the device
cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = reinforce(cartpole_policy, cartpole_optimizer, cartpole_hyperparameters["n_training_episodes"], cartpole_hyperparameters["max_t"], cartpole_hyperparameters["gamma"], 100)

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    # Evaluate the agent for 'n_eval_episodes' episodes and returns average reward and std of reward
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            
            state = new_state
        
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

eval = evaluate_agent(eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy)

print(eval)

#while(1):
#    env.reset()
#    env.render()