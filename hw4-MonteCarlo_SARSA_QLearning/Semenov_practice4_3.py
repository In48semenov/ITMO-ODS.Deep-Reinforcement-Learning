import sys
sys.path.append("../")

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.MonteCarlo-SARSA-QLearning"

ENV = gym.make("Taxi-v3")


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def MonteCarlo(
    env, 
    episode_n, 
    trajectory_len=500, 
    gamma=0.99, 
    epsilone_strategy: str = "STANDARD",
    constant: float = None,
) -> tuple[list[float], list[int]]:
    total_rewards = []
    trajectories_len = []
    epsilones = []
    
    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))
    counter = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n)):
        if episode == 0:
            epsilon = 1
        elif epsilone_strategy == "STANDARD":
            epsilon = 1 - episode / episode_n
        elif epsilone_strategy == "CONSTANT":
            epsilon = constant
        elif epsilone_strategy == "ADDITIVE":
            if epsilon - constant/episode_n > 0:
                epsilon -= constant/episode_n
        elif epsilone_strategy == "MULTIPLICATIVE":
            epsilon *= constant
        else:
            raise ValueError
        
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        
        state = env.reset()
        for trjct_idx in range(trajectory_len):
            trajectory['states'].append(state)
            
            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            trajectory['actions'].append(action)
            
            state, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)
            
            if done:
                break
                
        total_rewards.append(sum(trajectory['rewards']))
        trajectories_len.append(trjct_idx)
        
        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]
            
        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1
        
        epsilones.append(epsilon)
            
    return total_rewards, trajectories_len, epsilones


EPSILON_STRATEGIES = {
    'STANDARD': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.99, epsilone_strategy="STANDARD", constant=None),
    'CONSTANT': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.99, epsilone_strategy="CONSTANT", constant=1),
    'ADDITIVE': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.99, epsilone_strategy="ADDITIVE", constant=0.1),
    'MULTIPLICATIVE': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.99, epsilone_strategy="MULTIPLICATIVE", constant=0.8),
}

TASK_NAME = f"Task-3-v1"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(EPSILON_STRATEGIES)

total_rewards_all = {}
trajectories_len_all = {}
epsilones = {}

for epsilone_name, algorithm in EPSILON_STRATEGIES.items():
    print(f'Start fit {epsilone_name} ...')
    total_rewards, trajectories_len, epsilone = MonteCarlo(**algorithm)

    total_rewards_all[epsilone_name] = total_rewards
    trajectories_len_all[epsilone_name] = trajectories_len
    epsilones[epsilone_name] = epsilone

    print(f'End fit {epsilone_name}')

for epsilone_name, total_rewards in total_rewards_all.items():
    plt.plot(total_rewards, label=epsilone_name)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
manager.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)
plt.show()

for epsilone_name, trajectories_len in trajectories_len_all.items():
    plt.plot(trajectories_len, label=epsilone_name)
plt.xlabel("Episode")
plt.ylabel("Number of requests to envirements")
plt.grid(True)
plt.legend()
manager.report_plot(title="Training plot trajectories length", series="trajectories_length", plt=plt)
plt.show()

for epsilone_name, epsilone in epsilones.items():
    plt.plot(epsilone, label=epsilone_name)
plt.xlabel("Episode")
plt.ylabel("Epsilon value")
plt.grid(True)
plt.legend()
manager.report_plot(title="Training plot trajectories length", series="trajectories_length", plt=plt)
plt.show()

manager.task.close()
