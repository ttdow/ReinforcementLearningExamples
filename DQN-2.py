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

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        #Return only every 'skip'-th frame
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # Repeat action and sum reward
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info

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
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels, height, width = input_dim

        if width != 84:
            raise ValueError(f"Expecting input width: 240, got {width}")
        if height != 84:
            raise ValueError(f"Expecting input height: 256, got {height}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, input):
        return self.cnn(input)

# -----------------------------------------------------------------------------
# Initialize env
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

next_state, reward, done, trunc, info = env.step(env.action_space.sample())

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Initialize model
model = MarioNet(env.observation_space.shape, 12).float()
model = model.to(device='cpu')

def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x

# Run game loop
episodes = 10
for e in range(episodes):
    state = env.reset()
    state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
    state = torch.tensor(state, device='cpu').unsqueeze(0)

    while(True):

        # Run model over current state
        action_values = model(state)

        # Perform highest-value action
        action = torch.argmax(action_values, axis=1).item()
        next_state, reward, done, trunc, info = env.step(action)

        # Remember

        # Learn

        # Update state
        #next_state = first_if_tuple(next_state).__array__()
        #next_state = torch.tensor(next_state, device='cpu')
        state = next_state
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device='cpu').unsqueeze(0)

        if done or info['flag_get']:
            break