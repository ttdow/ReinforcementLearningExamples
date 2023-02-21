import torch
from torch import nn
from torch import optim
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

# Environment wrappers --------------------------------------------------------
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

# CNN -------------------------------------------------------------------------
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, input):
        actions = self.cnn(input)
        return actions

# Agent -----------------------------------------------------------------------
class Agent:
    def __init__(self, input_dims):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 5e-4
        self.learning_rate = 0.001
        self.action_space = [i for i in range(12)]
        self.memory_size = 50000
        self.batch_size = 64
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = MarioNet((4, 84, 84), 12)

        #self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        #self.new_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
        #self.action_memory = np.zeros(self.memory_size, dtype=np.float32)
        #self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            #state = torch.tensor([observation]).to((self.Q_eval.device))
            action_values = self.Q_eval.forward(state)
            next_action = torch.argmax(action_values).item()
        else:
            next_action = np.random.choice(self.action_space)

        return next_action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(self.reward.memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward.memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

# Initialize env --------------------------------------------------------------
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

#next_state, reward, done, trunc, info = env.step(env.action_space.sample())

# Apply envrionment wrappers
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Initialize model
#model = MarioNet(env.observation_space.shape, 12).float()
agent = Agent(1)
#model = model.to(device='cpu')

#Agent(input_dims=[28224])

def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x

# Game loop -------------------------------------------------------------------
episodes = 10
for e in range(episodes):
    state = env.reset()
    state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
    state = torch.tensor(state, device='cpu').unsqueeze(0)

    while(True):

        # Run model over current state
        #action_values = model(state)
        action = agent.choose_action(state)

        # Perform highest-value action
        #action = torch.argmax(action_values, axis=1).item()
        next_state, reward, done, trunc, info = env.step(action)

        print(info)

        # Remember

        # Learn

        # Update state
        #next_state = first_if_tuple(next_state).__array__()
        #next_state = torch.tensor(next_state, device='cpu')
        #state = next_state
        #state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        #state = torch.tensor(state, device='cpu').unsqueeze(0)

        if done or info['flag_get']:
            break