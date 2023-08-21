import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import gym

# Simple feed-forward, fully-connected NN.
class Net(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Net, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)

        return y

class PPO():
    def __init__(self, env):

        # Define learning environment.
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Initalize actor and critic NNs.
        self.actor = Net(self.state_dim, self.action_dim)
        self.critic = Net(self.state_dim, 1)

        # Initialize training hyperparameters.
        self.init_hyperparameters()

        # Create our variable for the matrix. [action_dim]
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5) 

        # Create the covariance matrix.
        self.cov_mat = torch.diag(self.cov_var)

        # Define the optimizers.
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

    def init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600

        self.gamma = 0.95

        self.n_updates_per_iteration = 5

        self.clip = 0.2

        self.lr = 0.005

    def get_action(self, state):

        # Query the actor network for a mean action.
        mean = self.actor(state)

        # Create our multivariate normal distribution.
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distributionand get its log prob.
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action.
        return action.detach().numpy(), log_prob.detach()

    def compute_rewards_to_go(self, rewards):

        # The rewards-to-go per episode per batch to return. [timesteps] 
        rewards_to_go = []

        # Iterate through each episode backwards to maintain same order.
        for episode_rewards in reversed(rewards):
            
            # Initialize discounted reward for this episode.
            discounted_reward = 0

            # Calculate discounted reward by progressing backward through episode.
            for reward in reversed(episode_rewards):

                # This step's reward is equal to the reward acquired plus
                # the discounted reward from the steps after.
                discounted_reward = reward + discounted_reward * self.gamma

                # Track the reward at each step.
                rewards_to_go.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor.
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float)

        return rewards_to_go
    
    def evaluate(self, states, actions):

        # Query critic network for a value for each state in the batch of states.
        value = self.critic(states).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        mean = self.actor(states)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)

        # Return predicted values and log probs.
        return value, log_probs

    def rollout(self):

        # Batch data.
        states = []         # [timesteps, states]
        actions = []        # [timesteps, actions]
        log_probs = []      # [timesteps]
        rewards = []        # [episodes, timesteps]
        rewards_to_go = []  # [timesteps]
        lengths = []        # [episodes]

        # Number of timesteps run so far this batch.
        t = 0

        # Loop until max timesteps per batch.
        while t < self.timesteps_per_batch:

            # Rewards this episode.
            ep_rewards = []

            state = self.env.reset()

            if type(state) == tuple:
                state = state[0]

            done = False    

            for ep_t in range(self.max_timesteps_per_episode):
                
                # Increment timesteps ran this batch so far.
                t += 1

                # Collect state observations.
                states.append(state)

                # Use actor NN to determine action.
                action, log_prob = self.get_action(state)

                # Step environment by performing selected action.
                state, reward, done, _, _ = self.env.step(action)

                # Collect reward, action, and log prob.
                ep_rewards.append(reward)
                actions.append(action)
                log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and reward.
            lengths.append(ep_t+1) # +1 because timesteps start at 0
            rewards.append(ep_rewards)

        for ep_reward in rewards:
            print(sum(ep_reward) / len(ep_reward))            

        # Reshape data as tensors in the shape specified before returning.
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        log_probs = torch.tensor(log_probs, dtype=torch.float)

        # Compute the rewards over the batch of episodes.
        rewards_to_go = self.compute_rewards_to_go(rewards)

        # Return the batched data.
        return states, actions, log_probs, rewards_to_go, lengths

    def learn(self, total_timesteps):

        t_so_far = 0 # Timesteps simulated so far.

        while t_so_far < total_timesteps:

            # Perform a rollout of the current policy over several episodes to generate data.
            states, actions, log_probs, rewards_to_go, lengths = self.rollout()

            # Calculate how many timesteps we collected this batch.
            t_so_far += np.sum(lengths)

            # Calculate the value V_{phi, k}.
            value, _ = self.evaluate(states, actions)

            # Calculate the advantage A_k.
            advantage = rewards_to_go - value.detach()

            # Normalize advantages.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):

                # Calculate V_phi and phi_theta(a_t | s_t).
                value, curr_log_probs = self.evaluate(states, actions)

                # Calculate the ratios.
                ratios = torch.exp(curr_log_probs - log_probs)

                # Calculate the surrogate losses.
                surrogate1 = ratios * advantage
                surrogate2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * advantage

                # Calculate loss for both NNs.
                actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
                critic_loss = nn.MSELoss()(value, rewards_to_go)

                # Calculate gradients and perform backward propagation for actor network.
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network.
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

env = gym.make('Pendulum-v1')
agent = PPO(env)
agent.learn(10000)