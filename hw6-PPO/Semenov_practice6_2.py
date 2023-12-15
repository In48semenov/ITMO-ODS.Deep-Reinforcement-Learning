import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Normal
from tqdm import tqdm


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.9, batch_size=128, 
                 epsilon=0.2, epoch_n=30, pi_lr=1e-4, v_lr=5e-4, advantage_old: bool = True):

        super().__init__()
        
        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 256), nn.ReLU(),
                                      nn.Linear(256, 128), nn.ReLU(),
                                      nn.Linear(128, 2 * action_dim), nn.Tanh())
        
        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 1))
        self.tanh = nn.Tanh()

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.advantage_old = advantage_old

    def get_action(self, state):
        mean_1, log_std_1, mean_2, log_std_2 = self.pi_model(torch.FloatTensor(state))
        dist_1, dist_2 = Normal(mean_1, torch.exp(log_std_1)), Normal(mean_2, torch.exp(log_std_2))
        action_1 = dist_1.sample().numpy()  # .reshape(1)
        action_2 = dist_2.sample().numpy()  #.reshape(1)
        return np.array([action_1, action_2])  # np.concatenate((action_1, action_2))

    def fit(self, states, actions, rewards, dones, next_states):
        
        states, actions, rewards, dones, next_states = map(np.array, [states, actions, rewards, dones, next_states])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns, rewards, next_states = map(torch.FloatTensor, [states, actions, returns, rewards, next_states])

        mean_1, log_std_1, mean_2, log_std_2 = self.pi_model(states).T
        mean_1, log_std_1, mean_2, log_std_2 = mean_1.unsqueeze(1), log_std_1.unsqueeze(1), mean_2.unsqueeze(1), log_std_2.unsqueeze(1)
        dist_1, dist_2 = Normal(mean_1, torch.exp(log_std_1)), Normal(mean_2, torch.exp(log_std_2))
        old_log_probs = torch.concat([dist_1.log_prob(actions[:, 0].reshape(-1, 1)), dist_2.log_prob(actions[:, 1].reshape(-1, 1))], axis=1).detach()

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
                
                b_mean_1, b_log_std_1, b_mean_2, b_log_std_2 = self.pi_model(b_states).T
                b_mean_1, b_log_std_1, b_mean_2, b_log_std_2 = b_mean_1.unsqueeze(1), b_log_std_1.unsqueeze(1), b_mean_2.unsqueeze(1), b_log_std_2.unsqueeze(1)
                b_dist_1, b_dist_2 = Normal(b_mean_1, torch.exp(b_log_std_1)), Normal(b_mean_2, torch.exp(b_log_std_2))
                b_new_log_probs = torch.concat([b_dist_1.log_prob(b_actions[:, 0].reshape(-1, 1)), b_dist_2.log_prob(b_actions[:, 1].reshape(-1, 1))], axis=1)

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


env = gym.make('LunarLander-v2', continuous=True)
state_dim = 8
action_dim = env.action_space.shape[0]


epoch_n = 100
epsilon = 0.2
gamma = 0.99
pi_lr = 1e-4
v_lr = 5e-4
agent = PPO(state_dim, action_dim, advantage_old=False, epoch_n=epoch_n, epsilon=epsilon, gamma=gamma, pi_lr=pi_lr, v_lr=v_lr)
episode_n = 50
trajectory_n = 100

total_rewards = []

print("Start training...")
# loop = tqdm(range(episode_n),)
for episode in range(episode_n):

    states, actions, rewards, dones, next_states = [], [], [], [], []

    for t_idx in range(trajectory_n):
        total_reward = 0

        state = env.reset()
        for _ in range(1000):
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
plt.grid()
plt.savefig("total_rewards_v3.png")