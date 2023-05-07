import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

import gym

class PPO():

    # Initialize the PPO model, including the hyperparameters
    def __init__(self, policy_class, env, **hyperparameters):

        # Make sure the environment is compatible with our code.
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO.
        self._init_hyperparameters(hyperparameters)

        # Extract environment information.
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {'delta_t': time.time_ns(),
                       't_so_far': 0,
                       'i_so_far': 0,
                       'batch_lens': [],
                       'batch_rews': [],
                       'actor_losses': []}
        
    # Train the actor and critic networks.
    def learn(self, total_timesteps):

        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:

            # Collect batch simulations.
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch.
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations.
            i_so_far += 1

            # Logging timesteps so far and iterations so far.
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate the advantage at the k-th iteration.
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            # Normalize advantages to reduce variance.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs.
            for _ in range(self.n_updates_per_iteration):

                # Calculate V_phi and phi_theta(a_t | s_t).
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / phi_theta_k(a_t | s_t).
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate the actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network.
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network.
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss.
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training thus far.
            self._log_summary()

            # Save our model if it's time.
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pt')
                torch.save(self.critic.state_dict(), './ppo_critic.pt')

    # Collect the batch of data for simulation.
    def rollout(self):

        # Batch data.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic reward data.
        ep_rews = []

        # Track timesteps this batch.
        t = 0

        # Keep simulating until we've run more than or equal to the specified timesteps per batch.
        while t < self.timesteps_per_batch:

            # Reset rewards for this episode.
            ep_rews = []

            # Reset the environment.
            obs = self.env.reset()

            if type(obs) == tuple:
                obs = obs[0]

            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps.
            for ep_t in range(self.max_timesteps_per_episode):

                # If render is specified, render the environment.
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                # Update timestep counter.
                t += 1

                # Track observations in this batch.
                batch_obs.append(obs)

                # Calculate action and make a step in the environment.
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                # Track recent reward, action, and action log probability.
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break.
                if done:
                    break
            
            # Track episodic lengths and rewards.
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        print(len(batch_obs))

        # Reshape data as tensors in the shape specified in function description, before returning.
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    # Compute the reward-to-go of each timestep in a batch given the rewards.
    def compute_rtgs(self, batch_rews):

        # The rewards-to-go (rtg) per episode per batch to return. [num_timesteps]
        batch_rtgs = []

        # Iterate through each episode.
        for ep_rews in reversed(batch_rews):

            # The discounted reward so far.
            discounted_reward = 0

            # Iterate through all rewards in the episode. We go backwards for smoother calculation
            # of each discounted return.
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor.
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
    
    # Queries an action from the actor network.
    def get_action(self, obs):

        # Query the actor network for a mean action.
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution.
        action = dist.sample()

        # Calculate the log probability for that action.
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution.
        return action.detach().numpy(), log_prob.detach()
    
    # Estimate the values of each observation and the log probs of each action in the most
    # recent batch with the most recent iteration of the actor network.
    def evaluate(self, batch_obs, batch_acts):

        # Query critic network for a value V for each batch_obs. [num_timesteps]
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch and log probabilities
        # log_probs of each action in the bacth.
        return V, log_probs

    # Initialize default and custom values for hyperparameters.
    def _init_hyperparameters(self, hyperparameters):

        # Initialize default values for hyperparameters.
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        # Misc. parameters.
        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None

        # Change any default values to custom values for specified hyperparameters.
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified.
        if self.seed != None:

            # Check if our seed is valid.
            assert(type(self.seed) == int)

            # Set the seed.
            torch.manual_seed(self.seed)

    # Print to stdout what we've logged so far in the most recent batch.
    def _log_summary(self):

        # Calculate logging values.
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']

        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages.
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements.
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data.
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

# A standard in_dim-64-64-out_dim feed forward neural network.
class Net(nn.Module):

    # Initialize the network and set up the layers.
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    # Runs a forward pass through the neural network.
    def forward(self, obs):

        if type(obs) == tuple:
            obs = obs[0]

        # Convert observation to tensor if it's a numpy array.
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

# Trains the model.
def train(env, hyperparameters):

    # Create a model for PPO.
    model = PPO(policy_class=Net, env=env, **hyperparameters)

    model.learn(total_timesteps=200_000_000)

def main():
    hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }
    
    # Create gym environment.
    env = gym.make('Pendulum-v1', render_mode="rgb_array")

    train(env, hyperparameters)

if __name__ == '__main__':
    main()