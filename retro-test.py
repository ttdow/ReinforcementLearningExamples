import gym
from gym.spaces import Box
import retro

import torch
import torch.nn as nn
from torchvision import transforms as T

from collections import deque, namedtuple
import numpy as np
import random
import time
from PIL import Image

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super.__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = torch.from_numpy(observation).float()
        #print(observation.shape)
        print(observation.shape)
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation
    
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        width, height, channels = input_dim
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=0, end_dim=2),
            nn.Linear(37632, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, obs):
        return self.online(obs)
    
class Agent():
    def __init__(self, state_dim, action_dim):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = CNN(self.state_dim, self.action_dim)

        self.exploration_rate = 0
        self.exploration_decay_rate = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=60000)
        self.batch_size = 1
        self.learn_every = 50

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = random.randint(0, 11)
            #action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = torch.transpose(state, 0, 2)
            state = torch.transpose(state, 1, 2)
            action_values = self.net(state)
            action_idx = torch.argmax(action_values, axis=0).item()

        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def remember(self, state, next_state, actions, reward): #, done):
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        reward = torch.tensor([reward], device=self.device)
        #done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward)) #, done))

    def learn(self):
        #if self.curr_step < self.burnin:
        #    return None, None


        if self.curr_step % self.learn_every != 0:
            return
        
        if len(self.memory) < self.batch_size:
            return
        
        experiences = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*experiences))

        '''
        # Compute Q(s, a)
        q_values = []
        for state in batch.state:
            x = torch.tensor(state, dtype=torch.float32)
            x = torch.transpose(x, 0, 2)
            x = torch.transpose(x, 1, 2)    
            q_values.append(self.net(x))

        # Compute Q(s', a')
        next_q_values = []
        for next_state in batch.next_state:
            x = torch.tensor(next_state, dtype=torch.float32)
            x = torch.transpose(x, 0, 2)
            x = torch.transpose(x, 1, 2)
            next_q_values.append(self.net(x))

        rewards = []
        for reward in batch.reward:
            rewards.append(reward)

        for x in next_q_values:
            x = x * self.gamma
        '''

        x = torch.tensor(batch.state[0], dtype=torch.float32)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2) 
        q_value = self.net(x)

        x = torch.tensor(batch.next_state[0], dtype=torch.float32)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        next_q_value = self.net(x)

        reward = batch.reward[0]

        expected_q_value = next_q_value + reward

        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        time0 = time.time()

        self.optimizer.step()

        time1 = time.time()

        print(loss)

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
env = ResizeObservation(env, shape=84)

agent = Agent(env.observation_space.shape, env.action_space.shape[0])

n_episodes = 10
for episode in range(n_episodes):
    obs = env.reset()

    while True:

        action_idx = agent.act(obs)
        action = np.zeros(12, dtype=np.int8)
        action[action_idx] = 1

        next_obs, reward, done, info = env.step(action)

        agent.remember(obs, next_obs, action, reward)#, done)

        agent.learn()

        obs = next_obs

        env.render()

        if done:
            break

env.close()