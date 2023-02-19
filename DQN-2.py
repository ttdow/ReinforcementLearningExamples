import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque, namedtuple
import random, datetime, os, copy

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        width, height = input_dim
        channels = 1

        if width != 240:
            raise ValueError(f"Expecting input width: 240, got {width}")
        if height != 256:
            raise ValueError(f"Expecting input height: 256, got {height}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(728, 256),
            nn.ReLU(),
            nn.Linear(256, 12),
        )
    
    def forward(self, input):
        return self.cnn(input)

# -----------------------------------------------------------------------------
# Initialize env
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

next_state, reward, done, trunc, info = env.step(env.action_space.sample())

env = GrayScaleObservation(env)
state = env.reset()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

print(env.observation_space.shape)
model = MarioNet(env.observation_space.shape, env.action_space).float()
model = model.to(device='cpu')

next_state, reward, done, trunc, info = env.step(env.action_space.sample())

#next_state = torch.Tensor(next_state.copy().astype(np.float32).transpose(), device='cpu')
print(next_state.shape)
action_values = model.forward(next_state)
print(action_values.shape)

'''
episodes = 10
for e in range(episodes):
    state = env.reset()

    # Play the hame
    while(True):

        # Run agent on state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(ation)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Update state
        state = next_state

        # Check for end of game
        if done or info["flag_get"]:
            break
'''