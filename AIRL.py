import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals # needed to load environments

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

env = gym.make("seals/CartPole-v0")

expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64
)
expert.learn(100000)

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng
)

venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10
)
reward_net = BasicShapedRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
airl_trainer.train(300000)
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(np.mean(learner_rewards_after_training))
print(np.mean(learner_rewards_before_training))

plt.hist(
    [learner_rewards_before_training, learner_rewards_after_training],
    label=["untrained", "trained"]
)

plt.legend()
plt.show()

class Policy_net:
    def __init__(self, name: str, env):
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        with 


# Set up environment
env = gym.make('MountainCar-v0')
observation_space = env.observation_space
action_space = env.action_space

