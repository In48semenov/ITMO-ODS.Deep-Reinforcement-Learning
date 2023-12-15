import sys
sys.path.append("../")

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm

from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.PPO"


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.9, batch_size=128, 
                 epsilon=0.2, epoch_n=30, pi_lr=1e-4, v_lr=5e-4, advantage_old: bool = True):

        super().__init__()
        
        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, action_dim))  # , nn.Softmax(dim=0))
        
        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 1))
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.advantage_old = advantage_old

    def get_action(self, state):
        logits = self.pi_model(torch.FloatTensor(state))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.numpy()

    def fit(self, states, actions, rewards, dones, next_states):
        
        states, actions, rewards, dones, next_states = map(np.array, [states, actions, rewards, dones, next_states])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns, rewards, next_states = map(torch.FloatTensor, [states, actions, returns, rewards, next_states])

        logits = self.pi_model(states)
        dist = Categorical(logits=logits)
        old_log_probs = dist.log_prob(actions).detach()

        for _ in range(self.epoch_n):
            
            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_states_next = next_states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                if self.advantage_old:
                    b_advantage = b_returns.detach() - self.v_model(b_states)
                else:
                    b_advantage = b_rewards.detach() + self.gamma * self.v_model(b_states_next) - self.v_model(b_states)
                
                logits = self.pi_model(b_states)
                b_dist = Categorical(logits=logits)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon,  1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))
                
                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()
                
                if self.advantage_old:
                    v_loss = torch.mean(b_advantage ** 2)
                else:
                    v_loss = torch.mean((self.v_model(b_states) - b_returns.detach()) ** 2)
    
                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


env = gym.make('Acrobot-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

epsilon = 0.5
gamma = 0.99
epoch_n = 100
pi_lr = 1e-4
v_lr = 5e-2
agent = PPO(state_dim, action_dim, advantage_old=True, epoch_n=epoch_n, gamma=gamma, epsilon=epsilon, pi_lr=pi_lr, v_lr=v_lr)

episode_n = 50
trajectory_n = 50

TASK_NAME = f"Task3-v1"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(dict(episode_n=episode_n, trajectory_n=trajectory_n,
                        gamma=gamma, batch_size=128, 
                        epsilon=epsilon, epoch_n=epoch_n,
                        pi_lr=pi_lr, v_lr=v_lr))

total_rewards = []

print("Start training...")
# loop = tqdm(range(episode_n),)
for episode in range(episode_n):

    states, actions, rewards, dones, next_states = [], [], [], [], []

    for t_idx in range(trajectory_n):
        total_reward = 0

        state = env.reset()
        for _ in range(500):
            states.append(state)
            
            action = agent.get_action(state)
            actions.append(action)

            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

        print(f"Episode: {episode} - trajectory: {t_idx}: {total_reward}")

    agent.fit(states, actions, rewards, dones, next_states)

print("Training finished.")

plt.plot(total_rewards, label="old" if agent.advantage_old else "new")
plt.title('Total Rewards')
plt.xlabel("Trajectory")
plt.ylabel("Total Reward")
plt.legend()

manager.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)

manager.task.close()