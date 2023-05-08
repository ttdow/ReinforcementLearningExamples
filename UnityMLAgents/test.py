import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (EngineConfigurationChannel,)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init___()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linea(64, 2)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y = self.fc3(x)

        return y
    
class Agent():

    def __init__(self):

        self.net = Net()
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.003)
        self.loss_fn = nn.MSELoss()
        self.memory = []
        self.batch_size = 128
        self.steps = 0
        self.update_freq = 1000

    def act(self, obs, epsilon):
        
        if random.random() < epsilon:
            return 

# Create Unity ML Agent environment.
config_channel = EngineConfigurationChannel()
config_channel.set_configuration_parameters(width=1800, height=900, time_scale=1.0)
env = UnityEnvironment(file_name=None, seed=1, side_channels=[config_channel])

# Reset the environment. Returns None.
env.reset()

# Determine behavior in Unity scene.
behavior_names = list(env.behavior_specs.keys())
behavior_name = behavior_names[0]

# Extract observation and action spaces from behavior.
behavior_specs = env.behavior_specs[behavior_name]
obs_spec = behavior_specs[0][0]
act_spec = behavior_specs[1]

#print(obs_spec)
#print(obs_spec.shape) # (8,) => (targetPos, agentPos, xVelocity, zVelocity)
#print(act_spec.continuous_size) # 2 (x, z)

decision_steps = env.get_steps(behavior_name)[0]
terminal_steps = env.get_steps(behavior_name)[1]

print(len(decision_steps))
print(len(terminal_steps))

while(1):

    # Get random action from action space.
    action = act_spec.random_action(1)

    # Set the action as the agent's action to take.
    env.set_actions(behavior_name, action)

    # Step the environment forward.
    env.step()
