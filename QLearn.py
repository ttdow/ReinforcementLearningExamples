import gym
import time
import random
import numpy as np
#from tqdm.notebook import trange

env=gym.make("FrozenLake-v1", render_mode="rgb_array", map_name="4x4", is_slippery=False)
env.reset()
env.render()
print('Initial state of the system.')

# Observation space - states
state_space = env.observation_space.n
print("There are ", state_space, " possible states.")

#actions: left=0, down=1, right=2, up=3
action_space = env.action_space.n
print("There are ", action_space, " possible actions.")

# Initialize Q-Table to all zeros
def initialize_q_table(state_space, action_space):
    Qtable = np.random.uniform(0, 1.0, (state_space, action_space))
    return Qtable;

qtable = initialize_q_table(state_space, action_space)

# Define E greedy policy - i.e. randomly explore or exploit
def epsilon_greedy_policy(q, state, epsilon):
    random_val =  random.uniform(0,1)
    if random_val > epsilon:
        action = np.argmax(q[state])
    else:
        action = env.action_space.sample()
    return action

def greedy_policy(q, state):
    action = np.argmax(q[state])
    return action

# Training parameters
n_training_episodes = 10000
learning_rate = 0.7

# Evaluation parameters
n_eval_episodes = 100

# Environment parameters
env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.005

def train(n_training_episodes, min_epsilon, max_epsilon, epsilon_decay, env, max_steps, q):
    for episode in  range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)

        # Reset the environment
        state = env.reset()
        state = state[0]
        step = 0
        done = False

        # Repeat
        for step in range(max_steps):
            # Take action
            action = epsilon_greedy_policy(q, state, epsilon)

            # Update next_state
            new_state, reward, done, trunc, info = env.step(action)

            # Update Q-Table
            q[state][action] = q[state][action] + learning_rate * (reward + gamma * np.max(q[new_state]) - q[state][action])

            #env.render()
            #time.sleep(1)

            # If done, finish the episode
            if done:
                break

            # Update current state
            state = new_state

        q_avg = np.mean(q)
        print("Q_Avg=", q_avg, "Epsilon=", epsilon)
    
    return q

def evaluate_agent(env, max_steps, n_eval_episodes, q, seed):
    episode_rewards = []

    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()

        state = state[0]
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum reward
            action = np.argmax(q[state][:])
            new_state, reward, done, trunc, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            
            state = new_state
        
        episode_rewards.append(total_rewards_ep)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

qtable = train(n_training_episodes, min_epsilon, max_epsilon, epsilon_decay, env, max_steps, qtable)
print(qtable)
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, qtable, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")



env.close()