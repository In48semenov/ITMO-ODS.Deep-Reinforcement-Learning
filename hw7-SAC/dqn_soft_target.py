import copy
from collections import OrderedDict
from typing import List

import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=0)
        )

    def forward(self, states):
        actions = self.q_network(states)
        return actions


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
        loop = tqdm(range(episode_n), )
        for episode_idx in loop:
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
            loop.set_description(f"Episode {episode_idx}")
            loop.set_postfix(dict(total_reward=total_reward))
        
        return totals_rewards


def dqn_soft_fit(env, state_dim, action_dim, episode_n, t_max: int = 500, **algorithm_parameters):
    agent = DQNSoftTargetUpdate(state_dim, action_dim, **algorithm_parameters)
    print("Start training DQN-Soft-Target-Network...")
    totals_rewards = agent.fit(env=env, episode_n=episode_n, t_max=t_max)
    print("Training finished.")
    return totals_rewards
