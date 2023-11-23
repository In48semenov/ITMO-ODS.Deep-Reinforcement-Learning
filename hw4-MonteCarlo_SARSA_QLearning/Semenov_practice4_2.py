import sys
sys.path.append("../")

from collections import defaultdict
from typing import Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.MonteCarlo-SARSA-QLearning"

ENV = gym.make("CartPole-v1")

CART_POSITION_BINS = np.linspace(-4.8, 4.8, 100)
CART_VELOCITY_BINS = np.linspace(-100, 100, 1000)
POLE_ANGLE_BINS = np.linspace(-0.42, 0.42, 100)
POLE_VELOCITY_BINS = np.linspace(-100, 100, 1000)


class CEM(nn.Module):

    def __init__(self, state_dim: int, action_n: int, lr: float=1e-2):
        super().__init__()

        self.state_dim = state_dim
        self.action_n = action_n
        self.eps = 1

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, self.action_n),
        )

        self.softmax = nn.Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input: torch.Tensor):
        return self.network(_input)
    
    def get_action(self, state: np.ndarray, num_iterations: Union[int, None] = None):
        state = torch.FloatTensor(state)
        logits = self.forward(state) 
        action_probs = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_probs)
        return action
    
    def update_policy(
        self, 
        elite_trajectories: List[Dict[str, Union[int, np.ndarray]]],
        return_loss: bool = False
    ) -> Union[None, float]:
        
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))

        predict_actions = self.forward(elite_states)
        if predict_actions.shape[0] != elite_actions.shape[0]:
            predict_actions = predict_actions[ :elite_actions.shape[0], :]
        loss = self.loss(predict_actions, elite_actions)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if return_loss:
            return loss.item()
        
    
def get_trajectory(
    env,
    agent,
    max_len: int = 200,
    visualize: bool = False,
) -> Dict[str, Union[np.ndarray, int, float]]:
    trajectory = {
        "states": [],
        "actions": [],
        "total_reward": 0,
    }

    # An observation is an integer that encodes the corresponding state
    state = env.reset()
    trajectory['states'].append(state)

    for idx in range(max_len):
        action = agent.get_action(state)

        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward

        if visualize:
            env.render()

        if done:
            break

        trajectory['states'].append(state)

    return trajectory, idx


def get_elite_trajectories(trajectories, q_param) -> List[Dict[str, Union[int, np.ndarray]]]:
    total_rewards = [trajectory["total_reward"] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory["total_reward"] > quantile]


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def CrossEntropy(env, episode_n, lr=1e-2, q_params=0.6, trajectory_len=500) -> tuple[list[float], list[int]]:
    STATE_N = 4  # env.observation_space.n
    ACTION_N = env.action_space.n
    agent = CEM(STATE_N, ACTION_N, lr)

    mean_total_reward_for_logging = []
    mean_trajectories_len_for_logging = []
    for episode in tqdm(range(episode_n)):
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_len)]
        
        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory, _ in trajectories])
        mean_trajectories_len = np.mean([length for _, length in trajectories])

        elite_trajectories = get_elite_trajectories([trajectory for trajectory, _ in trajectories], q_params)
        
        if len(elite_trajectories) > 0:
            _ = agent.update_policy(elite_trajectories, return_loss=True)
 
        mean_total_reward_for_logging.append(mean_total_reward)
        mean_trajectories_len_for_logging.append(mean_trajectories_len)
    
    return mean_total_reward_for_logging, mean_trajectories_len_for_logging


def observation_round_and_add_qf(
    state: np.ndarray,
):
    cart_pos, cart_vel, pole_ang, pole_vel = state
    
    discrete_state = (
        np.digitize(cart_pos, CART_POSITION_BINS),
        np.digitize(cart_vel, CART_VELOCITY_BINS),
        np.digitize(pole_ang, POLE_ANGLE_BINS),
        np.digitize(pole_vel, POLE_VELOCITY_BINS)
    )

    return discrete_state


def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99) -> tuple[list[float], list[int]]:
    total_rewards = []
    trajectories_len = []
    
    action_n = env.action_space.n
    qfunction = defaultdict(lambda: [0 for _ in range(action_n)])
    counter = defaultdict(lambda: [0 for _ in range(action_n)])
    
    for episode in tqdm(range(episode_n)):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        
        state = env.reset()
        for trjct_idx in range(trajectory_len):
            state_round = observation_round_and_add_qf(state)  #, qfunction, action_n)
            trajectory['states'].append(state_round)
            
            action = get_epsilon_greedy_action(qfunction[state_round], epsilon, action_n)
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
    
    action_n = env.action_space.n
    qfunction = defaultdict(lambda: [0 for _ in range(action_n)])
    
    for episode in tqdm(range(episode_n)):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()
        for trjct_idx in range(trajectory_len):
            state_round = observation_round_and_add_qf(state)
            action = get_epsilon_greedy_action(qfunction[state_round], epsilon, action_n)

            next_state, reward, done, _ = env.step(action)
            next_state_round = observation_round_and_add_qf(next_state)
            next_action = get_epsilon_greedy_action(qfunction[next_state_round], epsilon, action_n)
            
            qfunction[state_round][action] += (
                alpha * (reward + gamma * qfunction[next_state_round][next_action] - qfunction[state_round][action])
            )

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

    action_n = env.action_space.n  # 3
    qfunction = defaultdict(lambda: [0 for _ in range(action_n)])

    for episode in tqdm(range(episode_n)):
        epsilon = 1 / (episode + 1)
        
        state = env.reset()
        for trjct_idx in range(trajectory_len):
            state_round = observation_round_and_add_qf(state)  #, qfunction, action_n)
            action = get_epsilon_greedy_action(qfunction[state_round], epsilon, action_n)
            
            next_state, reward, done, _ = env.step(action)
            next_state_round = observation_round_and_add_qf(next_state)  #, qfunction, action_n)

            qfunction[state_round][action] += alpha * (reward + gamma * np.max(qfunction[next_state_round]) - qfunction[state_round][action])

            state = next_state

            total_rewards[episode] += reward

            if done:
                break
        
        trajectories_len.append(trjct_idx)

    return list(total_rewards), trajectories_len


ALGORITHMS = {
    'DeepCrossEntropy': dict(env=ENV, lr=1e-2, episode_n=1000, q_params=0.8, trajectory_len=100),
    'MonteCarlo': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.99),
    'SARSA': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.999, alpha=0.5),
    'QLearning': dict(env=ENV, episode_n=10000, trajectory_len=1000, gamma=0.999, alpha=0.5)
}

TASK_NAME = f"Task-2-v1"
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

manager.task.close()
