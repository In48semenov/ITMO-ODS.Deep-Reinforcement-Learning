import sys
sys.path.append("../")

from typing import Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from hw.utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.DeepCEM"


class CEM(nn.Module):

    def __init__(self, state_dim: int, lr: float=1e-2, eps: float = 1):
        super().__init__()

        self.state_dim = state_dim
        self.eps = eps

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        

    def forward(self, _input: torch.Tensor):
        return self.network(_input)
    
    def get_noise_negative(self, ) -> torch.Tensor:
        random_mean = np.random.uniform(-1, 0)
        random_std_dev = np.random.uniform(0, 1)
        noise = np.random.normal(random_mean, random_std_dev, size=(1, ))
        noise = torch.FloatTensor(np.clip(noise, -1, 1))
        return noise
    
    def get_noise_positive(self, ) -> torch.Tensor:
        random_mean = np.random.uniform(0, 1)
        random_std_dev = np.random.uniform(0, 1)
        noise = np.random.normal(random_mean, random_std_dev, size=(1, ))
        noise = torch.FloatTensor(np.clip(noise, -1, 1))
        return noise

    def get_action(self, state: np.ndarray, iter: int = None) -> np.float32:
        state = torch.FloatTensor(state)
        action = self.forward(state).detach()
        if iter is not None:
            if (iter + 1) % 2 == 1:
                noise = torch.normal(-0.5, self.eps, size=action.shape)
                if noise > 0:
                    noise *= -1
                action = torch.clip(action, -1, 0).numpy() - self.tanh(noise).numpy()
                return action
            elif (iter + 1) % 2 == 0:
                noise = torch.normal(0.5, self.eps, size=action.shape)
                if noise < 0:
                    noise *= -1
                action = torch.clip(action, 0, 1).numpy()  + self.tanh(noise).numpy() 
                return action
        else:
            return action.numpy()
    
    def update_policy(
        self, 
        elite_trajectories: List[Dict[str, Union[int, np.ndarray]]],
        return_loss: bool = False
    ) -> Union[None, float]:
        
        self.optimizer.zero_grad()

        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))

        predict_actions = self.forward(elite_states)
        loss = self.loss(predict_actions, elite_actions)
        
        loss.backward()
        self.optimizer.step()

        if return_loss:
            return loss.item()

    
def get_trajectory(
    env,
    agent,
    max_len: int = 10000,
    visualize: bool = False,
    episode: int = None,
) -> Dict[str, Union[np.ndarray, int, float]]:
    trajectory = {
        "states": [],
        "actions": [],
        "total_reward": 0,
        "max_reward": float("-inf")
    }

    # An observation is an integer that encodes the corresponding state
    state = env.reset()
    trajectory['states'].append(state)
    for _ in range(max_len):
        if episode is not None and episode < 20:
            action = agent.get_action(state, episode)
        else:
            action = agent.get_action(state)

        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward
        
        if reward > trajectory["max_reward"]:
            trajectory["max_reward"] = reward

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



env = gym.make('MountainCarContinuous-v0')
env.seed(17)
state_dim = 2
lr = 1e-2

episode_n = 30
trajectory_n = 200
trajectory_len = 999 
q_param = 0.8
eps = 0.5

TASK_NAME = f"Task-MountainCarContinuousV0-2-v13"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(
    {
        "Episode_n": episode_n,
        "Trajectory_n": trajectory_n,
        "Trajectory_len": trajectory_len,
        "Q_param": q_param,
        "Learning rate": lr,
        "Eps": "noise-neg-and-pos",
        "Loss": "MSELoss + TANG"
    }
)

agent = CEM(state_dim, lr, eps)

for episode in range(episode_n):
    trajectories = [
        get_trajectory(env, agent, trajectory_len, episode=episode)  
        for _ in range(trajectory_n)
    ]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])

    if (episode + 1) % 2 == 0:
        agent.eps /= 2

    log = f' === episode: {episode}, mean_total_reward = {mean_total_reward}'

    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    if len(elite_trajectories) > 0:
        loss = agent.update_policy(elite_trajectories, return_loss=True)
        log += f', loss = {loss}'

    manager.report_metrics(
        "Training reward", "Mean Total Reward", mean_total_reward, episode
    )
    manager.report_metrics(
        "Training Loss", "Loss", loss, episode
    )
    print(log)

agent.eps = 0
trajectories = get_trajectory(env, agent, trajectory_len, visualize=True)

manager.report_metrics(
    "Test", "Total reward", trajectories['total_reward'], iteration=0
)
print(f"Max Reward: {trajectories['max_reward']}")
print(f"Total reward: {trajectories['total_reward']}")
