import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical
from tqdm import tqdm


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=1e-3, tau=1e-2, 
                 batch_size=64, pi_lr=1e-3, q_lr=1e-3, temperature=0.9):
        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), 
                                      nn.Linear(64, 64), nn.ReLU(), 
                                      nn.Linear(64, action_dim))

        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(), 
                                      nn.Linear(64, 64), nn.ReLU(), 
                                      nn.Linear(64, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(), 
                                      nn.Linear(64, 64), nn.ReLU(), 
                                      nn.Linear(64, 1))

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.temperature = torch.tensor(temperature)
        self.memory = []

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), q_lr)
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

    def get_action(self, state):
        logits = self.pi_model(torch.FloatTensor(state))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.numpy().reshape(1)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_and_actions = torch.concatenate((next_states, next_actions), dim=1)
            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)
            next_min_q_values = torch.min(next_q1_values, next_q2_values)
            targets = rewards + self.gamma * (1 - dones) * (next_min_q_values - self.alpha * next_log_probs)

            states_and_actions = torch.concatenate((states, F.one_hot(actions.type(torch.int64))), dim=1)
            q1_loss = torch.mean((self.q1_model(states_and_actions) - targets.detach()) ** 2)
            q2_loss = torch.mean((self.q2_model(states_and_actions) - targets.detach()) ** 2)
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            pred_actions, log_probs = self.predict_actions(states)
            states_and_pred_actions = torch.concatenate((states, pred_actions), dim=1)
            q1_values = self.q1_model(states_and_pred_actions)
            q2_values = self.q2_model(states_and_pred_actions)
            min_q_values = torch.min(q1_values, q2_values)
            pi_loss = - torch.mean(min_q_values - self.alpha * log_probs)
            self.update_model(pi_loss, self.pi_optimizer)
            
    def update_model(self, loss, optimizer, model=None, target_model=None):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if model != None and target_model != None:
            for param, target_param in zip(model.parameters(), target_model.parameters()):
                new_terget_param = (1 - self.tau) * target_param + self.tau * param
                target_param.data.copy_(new_terget_param)

    def predict_actions(self, states):
        logits = self.pi_model(torch.FloatTensor(states))
        dist = RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs.reshape(-1, 1)
    

def sac_fit(env, state_dim, action_dim, episode_n: int = 100, t_max: int = 500, **algorithm_parameters):
    agent = SAC(state_dim, action_dim, **algorithm_parameters)

    total_rewards = []
    print("Start training SAC...")
    loop = tqdm(range(episode_n), )
    for episode_idx in loop:

        total_reward = 0
        state = env.reset()
        for _ in range(t_max):
            action = agent.get_action(state)[0]
            next_state, reward, done, _ = env.step(action)
        
            agent.fit(state, action, reward, done, next_state)
        
            total_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(total_reward)
        loop.set_description(f"Episode {episode_idx}")
        loop.set_postfix(dict(total_reward=total_reward))
    print("Training finished.")
    return total_rewards