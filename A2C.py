import gym
import pybullet_envs

import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn

# Policy-based methods aim to optimize the policy directly without using a
#   value function.

# A2C (Advantage Actor Critic) uses an Actor-Critic method, a hybrid 
#   architecture combining value-based and policy-based methods that help 
#   to stabilize the training by reducing variance.
# An Actor that controls how our agent behaves (policy-based method).
# A Critic that measures how good the action taken is (value-based method).

# Initally the Actor tries random actions and the Critic provides feedback.
# The Actor then uses this feedback to improve their policy and the Critic
# updates their policy to give better feedback.

# We learn two function approximations:
#   1. A Policy that controls how our agent acts: pi_theta(s, a)
#   2. A value function to assist the policy update by measuring how good
#       the action taken is q_hat_w(s, a)

# The Actor-Critic process:
#   1. At each timestep, t, we get the current state, S_t, from the environment
#       and pass it as input through our Actor and Critic
#   2. Our Policy takes the state and outputs an action, A_t
#   3. The Critic takes that action as input and, using S_t and A_t, computes
#       the value of taking that action at that state: the Q-value
#   4. The action performed in the environment outputs a new state, S_t+1, 
#       and a reward, R_t+1
#   5. The Actor updates its policy parameters using the Q value.
#   6. The Actor then takes A_t+1 in S_t+1
#   7. The Critic updates its value function...

# Advantage function:
#   Calculates how better taking the action at a state is compared to the
#       average value of the state.
#   A(s, a) = Q(s, a) - V(s), where V(s) = the average value of the state
#   This calculates the extra reward we get if we take this action at that
#       state compared to the mean reward we get at that state - i.e. the
#       advantage of choosing this action. If the advantage is positive,
#       then the average value of that state is increasing, if the advantage
#       is negative then the average value of that state is decreasing -
#       i.e. gradient ascent.

# We can use the TD error as a good estimator of the advantage function:
#   A(s, a) = Q(s, a) - V(s)
#           = r + gamma * V(s') - V(s) = TD error4

# Create env
env_id = "AntBulletEnv-v0"
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space

# Visual shape of state space and action space
print("Observation space:")
print("State space size: ", s_size)
print("Sample observation: ", env.observation_space.sample())

print("Action space:")
print("Action space size: ", a_size)
print("Sample action: ", env.action_space.sample())

# Normalize input features
#env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

# Create A2C model
model = A2C(policy="MlpPolicy", env=env, gae_lambda=0.9, gamma=0.99, learning_rate=0.00096, max_grad_norm=0.5, n_steps=8,
            vf_coef=0.4, ent_coef=0.0, tensorboard_log="./tensorboard", policy_kwargs=dict(log_std_init=-2, ortho_init=False,
            ), normalize_advantage=False, use_rms_prop=True, use_sde=True, verbose=1)

# Train the model
model.learn(2000)

# Save the model and VecNormalize statistics when saving the agent
#model.save("a2c-AntBulletEnv-v0")
#env.save("vec_normalize.pkl")

# Load the statistics
eval_env = model
#eval_env = VecNormalize(env, eval_env)

# Do not update them at test time
eval_env.training = False

# Reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-AntBulletEnv-v0")

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")