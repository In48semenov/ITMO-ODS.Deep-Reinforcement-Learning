from typing import Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn


class CEM(nn.Module):

    def __init__(self, state_dim: int, lr: float=1e-2,):
        super().__init__()

        self.state_dim = state_dim
        self.eps = 100

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        

    def forward(self, _input: torch.Tensor):
        return self.network(_input) + torch.normal(0, self.eps, size=(1, ))
    
    def get_action(self, state: np.ndarray, train: bool = False) -> np.float32:
        state = torch.FloatTensor(state)
        action = self.forward(state).detach() 
        action = torch.clip(action, min=-2, max=2)
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
        elite_actions = torch.FloatTensor(np.array(elite_actions))

        predict_actions = self.forward(elite_states)
        # predict_actions = torch.clip(predict_actions, min=-1, max=1)
        # predict_actions = torch.sign(predict_actions) * torch.pow(abs(predict_actions), 0.0015)
        print(elite_actions)
        loss = self.loss(predict_actions, elite_actions)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if return_loss:
            return loss.item()

    
def get_trajectory(
    env,
    agent,
    max_len: int = 10000,
    visualize: bool = False,
    train: bool = False
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
        action = agent.get_action(state, train)

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
    # print(total_rewards, quantile)
    return [trajectory for trajectory in trajectories if trajectory["total_reward"] > quantile]



env = gym.make('Pendulum-v1')
state_dim = 3
lr = 0.01

episode_n = 10
trajectory_n = 100
trajectory_len = 999 
q_param = 0.8
agent = CEM(state_dim, lr)

for episode in range(episode_n):
    trajectories = [
        get_trajectory(env, agent, trajectory_len, train=True) 
        for _ in range(trajectory_n)
    ]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    # agent.eps /= 2
    log = f' === episode: {episode}, mean_total_reward = {mean_total_reward}'

    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    if len(elite_trajectories) > 0:
        loss = agent.update_policy(elite_trajectories, return_loss=True)
        log += f', loss = {loss}'

    print(log)

agent.eps = 0
trajectories = get_trajectory(env, agent, trajectory_len, visualize=True)
print(f"Total reward: {trajectories['total_reward']}")


