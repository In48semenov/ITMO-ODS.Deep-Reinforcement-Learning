import sys
sys.path.append("../")

from typing import Dict, List, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.DQN"

ENV = gym.make("Acrobot-v1")


# ==== DEEP CROSS-ENTROPY METHOD ====
class DeppCrossEntropy(nn.Module):

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

    for _ in range(max_len):
        action = agent.get_action(state)

        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward

        if visualize:
            env.render()

        if done:
            break

        trajectory['states'].append(state)

    return trajectory


def get_elite_trajectories(trajectories, q_param) -> List[Dict[str, Union[int, np.ndarray]]]:
    total_rewards = [trajectory["total_reward"] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory["total_reward"] > quantile]


def fit_depp_cross_entropy(
    env, agent, episode_n, q_params=0.6, trajectory_len=500
) -> List[float]:
    mean_total_reward_for_logging = []
    for _ in tqdm(range(episode_n)):
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_len)]
    
        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])

        elite_trajectories = get_elite_trajectories([trajectory for trajectory in trajectories], q_params)
        
        if len(elite_trajectories) > 0:
            _ = agent.update_policy(elite_trajectories, return_loss=True)
 
        mean_total_reward_for_logging.append(mean_total_reward)
    
    return mean_total_reward_for_logging


# ==== DQN METHOD ====
class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions
    

class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action
    
    def _fit_one_step(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))
    
            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]
            
            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()
            
            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max=500) -> List[float]:
        totals_rewards = []
        for _ in tqdm(range(episode_n)):
            total_reward = 0

            state = env.reset()
            for _ in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state)

                state = next_state
                total_reward += reward

                if done:
                    break

            totals_rewards.append(total_reward)
        
        return totals_rewards


STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.n

ALGORITHM_PARAMETERS = {
    "DQN-v1": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01),  # Best result
    "DQN-v2": dict(gamma=0.99, lr=1e-2, batch_size=128, epsilon_decrease=0.01, epilon_min=0.01),
    "DQN-v3": dict(gamma=0.90, lr=1e-2, batch_size=128, epsilon_decrease=0.01, epilon_min=0.01),
    "DQN-v4": dict(gamma=0.90, lr=1e-2, batch_size=128, epsilon_decrease=0.001, epilon_min=0.01),
    "DQN-v5": dict(gamma=0.99, lr=1e-2, batch_size=256, epsilon_decrease=0.01, epilon_min=0.01),
    "DQN-v6": dict(gamma=0.99, lr=1e-1, batch_size=64, epsilon_decrease=0.001, epilon_min=0.001),
    "DQN-v7": dict(gamma=0.90, lr=1e-1, batch_size=128, epsilon_decrease=0.001, epilon_min=0.001),
    "DQN-v8": dict(gamma=0.90, lr=1e-1, batch_size=256, epsilon_decrease=0.001, epilon_min=0.001),
    "DQN-v9": dict(gamma=0.99, lr=1e-3, batch_size=256, epsilon_decrease=0.001, epilon_min=0.01),
    "DQN-v10": dict(gamma=0.90, lr=1e-3, batch_size=128, epsilon_decrease=0.001, epilon_min=0.001),
    "DeepCrossEntropy": dict(lr=1e-2, q_params=0.6, trajectory_len=500),
}


AGENTS = {
    "DQN-v1": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v1"]),
    "DQN-v2": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v2"]),
    "DQN-v3": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v3"]),
    "DQN-v4": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v4"]),
    "DQN-v5": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v5"]),
    "DQN-v6": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v6"]),
    "DQN-v7": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v7"]),
    "DQN-v8": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v8"]),
    "DQN-v9": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v9"]),
    "DQN-v10": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-v10"]),
    "DeepCrossEntropy": DeppCrossEntropy(STATE_DIM, ACTION_DIM, ALGORITHM_PARAMETERS["DeepCrossEntropy"]["lr"]),
}

EPISODE_N = 100
T_MAX = 500
ALGORITHM_PARAMETERS["episode_n"] = EPISODE_N
ALGORITHM_PARAMETERS["t_max"] = T_MAX

TASK_NAME = f"Task1-v1"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(ALGORITHM_PARAMETERS)

plt.figure(figsize=(10, 15))
for agent_name, agent in AGENTS.items():
    print(f"Start training: {agent_name} ...")

    if "DQN" in agent_name:
        rewards = agent.fit(env=ENV, episode_n=EPISODE_N, t_max=T_MAX)
    elif "DeepCrossEntropy" in agent_name:
        rewards = fit_depp_cross_entropy(
            env=ENV,agent=agent, episode_n=EPISODE_N, 
            q_params=ALGORITHM_PARAMETERS["DeepCrossEntropy"]["q_params"], 
            trajectory_len=ALGORITHM_PARAMETERS["DeepCrossEntropy"]["trajectory_len"]
        )
    else:
        raise ValueError(f"Unknown agent {agent_name}")

    ALGORITHM_PARAMETERS[agent_name]["name_algorithm"] = agent_name
    plt.plot(rewards, label=ALGORITHM_PARAMETERS[agent_name])

    print(f"End training: {agent_name}")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend(loc="lower right")

manager.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)

manager.task.close()
