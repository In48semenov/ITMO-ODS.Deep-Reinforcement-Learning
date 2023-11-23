import sys
sys.path.append("../")

import time
from typing import Dict, List, Union, Tuple

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.MonteCarlo-SARSA-QLearning"

ENV = gym.make("Taxi-v3")


class CrossEntropyAgent:

    def __init__(self, state_n, action_n, q_param: float):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        self.q_param = q_param

    def get_action(self, state) -> int:
        action = np.random.choice(
            np.arange(self.action_n), p=self.model[state],
        )
        return int(action)

    def _get_elite_trajectories(
        self,
        trajectories: List[Dict[str, List[int]]],
        total_rewards: List[int],
    ) -> List[Dict[str, List[int]]]:
        quantile = np.quantile(total_rewards, self.q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory["rewards"])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        return elite_trajectories

    def fit(
        self,
        trajectories: List[Dict[str, List[int]]],
        total_rewards: List[int],
    ) -> None:
        elite_trajectories = self._get_elite_trajectories(
            trajectories, total_rewards
        )
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"],
                                     trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def CrossEntropy(env, episode_n, q_params=0.6, trajectory_len=500) -> tuple[list[float], list[int]]:
    STATE_N = env.observation_space.n
    ACTION_N = env.action_space.n
    MAX_LEN = 10000

    def get_trajectory(
        env,
        agent,
        max_len: int = 10000,
        visualize: bool = False,
        return_max_iter: bool = False
    ) -> Union[Dict, Tuple[Dict, int]]:
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

        # An observation is an integer that encodes the corresponding state
        state = env.reset()
        iter_end = 0
        for iter_idx in range(max_len):
            trajectory["states"].append(state)

            action = agent.get_action(state)
            trajectory["actions"].append(action)

            state, reward, done, _ = env.step(action)
            trajectory["rewards"].append(reward)

            if visualize:
                time.sleep(0.5)
                env.render()

            iter_end = iter_idx
            if done:
                break

        if return_max_iter:
            return trajectory, iter_end
        else:
            return trajectory
        
    agent = CrossEntropyAgent(state_n=STATE_N, action_n=ACTION_N, q_param=q_params)

    total_rewards_for_logging = []
    trajectories_len = []
    for _ in tqdm(range(episode_n)):
        # policy evaluation
        trajectories = [
            get_trajectory(env, agent, MAX_LEN) for _ in range(trajectory_len)
        ]
        total_rewards = [
            np.sum(trajectory["rewards"]) for trajectory in trajectories
        ]

        total_rewards_for_logging.append(np.mean(total_rewards))

        trajectories_len.append(
            np.mean(
                [len(trajectory["rewards"]) for trajectory in trajectories]
            )
        )
        # policy improvement
        agent.fit(trajectories, total_rewards)

    return total_rewards_for_logging, trajectories_len

def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99) -> tuple[list[float], list[int]]:
    total_rewards = []
    trajectories_len = []
    
    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))
    counter = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n)):
        epsilon = 1 - episode / episode_n
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
            
    return total_rewards, trajectories_len
    

def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5) -> tuple[list[float], list[int]]:
    total_rewards = np.zeros(episode_n)
    trajectories_len = []
    
    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))
    
    for episode in tqdm(range(episode_n)):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for trjct_idx in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)
            
            qfunction[state][action] += alpha * (reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])
            
            state = next_state
            action = next_action
            
            total_rewards[episode] += reward
            
            if done:
                break

        trajectories_len.append(trjct_idx)

    return list(total_rewards), trajectories_len


def QLearning(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5) -> tuple[list[float], list[int]]:
    total_rewards = np.zeros(episode_n)
    trajectories_len = []

    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))

    for episode in tqdm(range(episode_n)):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()
        for trjct_idx in range(trajectory_len):
            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)

            qfunction[state][action] += alpha * (reward + gamma * np.max(qfunction[next_state]) - qfunction[state][action])

            state = next_state

            total_rewards[episode] += reward

            if done:
                break
        
        trajectories_len.append(trjct_idx)

    return list(total_rewards), trajectories_len


ALGORITHMS = {
    'CrossEntropy': dict(env=ENV, episode_n=1000, q_params=0.6, trajectory_len=100),
    'MonteCarlo': dict(env=ENV, episode_n=1000, trajectory_len=1000, gamma=0.99),
    'SARSA': dict(env=ENV, episode_n=1000, trajectory_len=1000, gamma=0.999, alpha=0.5),
    'QLearning': dict(env=ENV, episode_n=1000, trajectory_len=1000, gamma=0.999, alpha=0.5)
}

TASK_NAME = f"Task-1-v1"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(ALGORITHMS)

total_rewards_all = {}
trajectories_len_all = {}

for algorithm_name, algorithm in ALGORITHMS.items():
    print(f'Start fit {algorithm_name} ...')
    if algorithm_name == 'MonteCarlo':
        total_rewards, trajectories_len = MonteCarlo(**algorithm)
    elif algorithm_name == 'SARSA':
        total_rewards, trajectories_len = SARSA(**algorithm)
    elif algorithm_name == 'QLearning':
        total_rewards, trajectories_len = QLearning(**algorithm)
    else:
        total_rewards, trajectories_len = CrossEntropy(**algorithm)

    total_rewards_all[algorithm_name] = total_rewards
    trajectories_len_all[algorithm_name] = trajectories_len

    print(f'End fit {algorithm_name}')

for algorithm_name, total_rewards in total_rewards_all.items():
    plt.plot(total_rewards, label=algorithm_name)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
manager.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)
plt.show()

for algorithm_name, trajectories_len in trajectories_len_all.items():
    plt.plot(trajectories_len, label=algorithm_name)
plt.xlabel("Episode")
plt.ylabel("Number of requests to envirements")
plt.grid(True)
plt.legend()
manager.report_plot(title="Training plot trajectories length", series="trajectories_length", plt=plt)
plt.show()

manager.task.close()
