import gym
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env_id = "CartPole-v1"
env = gym.make(env_id)

# Create expert
ppo_expert = PPO('MlpPolicy', env_id, verbose=1)
ppo_expert.learn(total_timesteps=3e4)
ppo_expert.save("ppo_expert")

mean_reward, std_reward = evaluate_policy(ppo_expert, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward} +/- {std_reward}")

# Create student
a2c_student = A2C('MlpPolicy', env_id, verbose=1)

# Create expert dataset
num_interactions = int(4e4)

if isinstance(env.action_space, gym.spaces.Box):
    expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
    expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0]))
else:
    expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
    expert_actions = np.empty((num_interactions,) + env.action_space.shape)

obs = env.reset()

for i in tqdm(range(num_interactions)):
    action, _ = ppo_expert.predict(obs, deterministic=True)
    expert_observations[i] = obs
    expert_actions[i] = action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

np.savez_compressed("expert_data", expert_actions=expert_actions, expert_observations=expert_observations)

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
    
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])
    
    def __len__(self):
        return len(self.observations)

# Convert expert data set into training/testing data for student
expert_dataset = ExpertDataSet(expert_observations, expert_actions)
train_size = int(0.8 * len(expert_dataset))
test_size = len(expert_dataset) - train_size
train_expert_dataset, test_expert_dataset = random_split(expert_dataset, [train_size, test_size])

def pretrain_agent(student, batch_size=64, epochs=1000, 
                   scheduler_gamma=0.7, learning_rate=1.0, 
                   log_interval=100, no_cuda=True, seed=1, test_batch_size=64):

    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only   
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    action = model(data)

                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()
                
                test_loss = criterion(action_prediction, target)

        test_loss /= len(test_loader.dataset)

    # Load previously created expert dataset for training and testing
    train_loader = torch.utils.data.DataLoader(dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Define an optimizer and a learning rate scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
    
    # Implant the trained policy network back into the RL student agent
    a2c_student.policy = model

# Evaluate the agent before pretraining
mean_reward, std_reward = evaluate_policy(a2c_student, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward} +/- {std_reward}")

# Run the pretraining
pretrain_agent(a2c_student, epochs=3, scheduler_gamma=0.7, learning_rate=1.0, 
               log_interval=100, no_cuda=True, seed=1, batch_size=64, test_batch_size=1000)
a2c_student.save("a2c_student")

# Test student's learning
mean_reward, std_reward = evaluate_policy(a2c_student, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward} +/- {std_reward}")