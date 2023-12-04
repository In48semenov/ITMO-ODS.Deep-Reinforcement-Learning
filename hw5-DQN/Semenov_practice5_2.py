import sys
sys.path.append("../")

import copy
from collections import OrderedDict
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


class DQNHardTargetUpdate:
    def __init__(
        self, 
        state_dim,
        action_dim, 
        gamma=0.99, 
        lr=1e-3, 
        batch_size=64, 
        epsilon_decrease=0.01, 
        epilon_min=0.01, 
        q_fix_update_epoch: int = None,
        q_fix_update_step: int = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.q_fix_update_epoch = q_fix_update_epoch
        self.q_fix_update_step = q_fix_update_step
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
    
    def _fit_one_step(self, state, action, reward, done, next_state, step):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))
    
            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function_fix(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]
            
            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()
            
            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease
            
            if self.q_fix_update_step is not None and (step + 1) % self.q_fix_update_step == 0:
                self.q_function_fix = self.q_function

    def fit(self, env, episode_n, t_max: int=500) -> List[float]:
        totals_rewards = []
        for episode_idx in tqdm(range(episode_n)):
            total_reward = 0

            state = env.reset()
            for step in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state, int((episode_idx + 1) * step))

                state = next_state
                total_reward += reward

                if done:
                    break
            
            if self.q_fix_update_epoch is not None and (episode_idx + 1) % self.q_fix_update_epoch == 0:
                self.q_function_fix = self.q_function

            totals_rewards.append(total_reward)
        
        return totals_rewards


class DQNSoftTargetUpdate:
    def __init__(
        self, 
        state_dim,
        action_dim, 
        gamma=0.99, 
        lr=1e-3, 
        batch_size=64, 
        epsilon_decrease=0.01, 
        epilon_min=0.01,
        tau: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def _mix_weights(self,):
        """
        Updated model weight for q_function_fix.
        """
        new_weights = []
        for (name_1, param_1), (name_2, param_2) in zip(self.q_function.named_parameters(), 
                                                        self.q_function_fix.named_parameters()):
            assert name_1 == name_2, "Model structure is not identical"
            new_weights.append(
                [
                    name_2, 
                    self.tau * param_1.data + (1 - self.tau) * param_2.data
                ]
            )
        self.q_function_fix.load_state_dict(OrderedDict(new_weights))

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
    
            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function_fix(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]
            
            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            self._mix_weights()
            
            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max: int=500) -> List[float]:
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


class DoubleDQN:

    def __init__(
        self, 
        state_dim,
        action_dim, 
        gamma=0.99, 
        lr=1e-3, 
        batch_size=64, 
        epsilon_decrease=0.01, 
        epilon_min=0.01,
        tau: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def _mix_weights(self,):
        """
        Updated model weight for q_function_fix.
        """
        new_weights = []
        for (name_1, param_1), (name_2, param_2) in zip(self.q_function.named_parameters(), 
                                                        self.q_function_fix.named_parameters()):
            assert name_1 == name_2, "Model structure is not identical"
            new_weights.append(
                [
                    name_2, 
                    self.tau * param_1.data + (1 - self.tau) * param_2.data
                ]
            )
        self.q_function_fix.load_state_dict(OrderedDict(new_weights))

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

            argmax_actions_from_q_fix = torch.argmax(self.q_function_fix(next_states), dim=1)
            targets = (
                rewards 
                + self.gamma 
                * (1 - dones) 
                * self.q_function(next_states)[torch.arange(self.batch_size), argmax_actions_from_q_fix]
            )
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]
            
            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            self._mix_weights()
            
            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max: int=500) -> List[float]:
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
    "DQN-best-result": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01),
    "DQN-Hard-Target-Update-v1": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_epoch=1, q_fix_update_step=None),
    "DQN-Hard-Target-Update-v2": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_epoch=10, q_fix_update_step=None),
    "DQN-Hard-Target-Update-v3": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_epoch=50, q_fix_update_step=None),
    "DQN-Hard-Target-Update-v4": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_step=10, q_fix_update_epoch=None),
    "DQN-Hard-Target-Update-v5": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_step=50, q_fix_update_epoch=None),
    "DQN-Hard-Target-Update-v6": dict(
        gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, q_fix_update_step=250, q_fix_update_epoch=None),
    "DQN-Soft-Target-Update-v1": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.01),
    "DQN-Soft-Target-Update-v2": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.1),
    "DQN-Soft-Target-Update-v3": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.5),
    "DQN-Soft-Target-Update-v4": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.9),
    "DQN-Double-v1": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.01),
    "DQN-Double-v2": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.1),
    "DQN-Double-v3": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.5),
    "DQN-Double-v4": dict(gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01, epilon_min=0.01, tau=0.9),
}


AGENTS = {
    "DQN-best-result": DQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-best-result"]),
    "DQN-Hard-Target-Update-v1": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v1"]),
    "DQN-Hard-Target-Update-v2": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v2"]),
    "DQN-Hard-Target-Update-v3": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v3"]),
    "DQN-Hard-Target-Update-v4": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v4"]),
    "DQN-Hard-Target-Update-v5": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v5"]),
    "DQN-Hard-Target-Update-v6": DQNHardTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Hard-Target-Update-v6"]),
    "DQN-Soft-Target-Update-v1": DQNSoftTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Soft-Target-Update-v1"]),
    "DQN-Soft-Target-Update-v2": DQNSoftTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Soft-Target-Update-v2"]),
    "DQN-Soft-Target-Update-v3": DQNSoftTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Soft-Target-Update-v3"]),
    "DQN-Soft-Target-Update-v4": DQNSoftTargetUpdate(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Soft-Target-Update-v4"]),
    "DQN-Double-v1": DoubleDQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Double-v1"]),
    "DQN-Double-v2": DoubleDQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Double-v2"]),
    "DQN-Double-v3": DoubleDQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Double-v3"]),
    "DQN-Double-v4": DoubleDQN(STATE_DIM, ACTION_DIM, **ALGORITHM_PARAMETERS["DQN-Double-v4"]),
}

EPISODE_N = 200
T_MAX = 500
ALGORITHM_PARAMETERS["episode_n"] = EPISODE_N
ALGORITHM_PARAMETERS["t_max"] = T_MAX

TASK_NAME = f"Task2-v1"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(ALGORITHM_PARAMETERS)

plt.figure(figsize=(10, 15))
for agent_name, agent in AGENTS.items():
    print(f"Start training: {agent_name} ...")

    rewards = agent.fit(env=ENV, episode_n=EPISODE_N, t_max=T_MAX)
    
    ALGORITHM_PARAMETERS[agent_name]["name_algorithm"] = agent_name
    plt.plot(rewards, label=ALGORITHM_PARAMETERS[agent_name])

    print(f"End training: {agent_name}")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend(loc="lower right")

manager.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)

manager.task.close()
