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
        
        # if num_iterations is not None:
        #     action_probs = self.softmax(
        #         (1 - self.eps) * action_probs + self.eps * torch.ones(self.action_n) / self.action_n
        #     ).numpy()
        #     self.eps /= num_iterations
        
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
    num_iterations: Union[int, None] = None,
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
        # if num_iterations is not None:
        #     action = agent.get_action(state, num_iterations)
        # else:
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


env = gym.make('Acrobot-v1')
state_dim = 6
action_n = 3
lr = 1e-2

episode_n = 70
trajectory_n = 50
trajectory_len = 500  # v1: 500; v0: 200
q_param = 0.6

TASK_NAME = f"Task-Acrobot-1-v10"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(
    {
        "Episode_n": episode_n,
        "Trajectory_n": trajectory_n,
        "Trajectory_len": trajectory_len,
        "Q_param": q_param,
        "Learning rate": lr,
        "Loss": "CrossEntropyLoss"
    }
)

agent = CEM(state_dim, action_n, lr)

for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
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
        
trajectories = get_trajectory(env, agent, trajectory_len, visualize=True)

manager.report_metrics(
    "Test", "Total reward", trajectories['total_reward'], iteration=0
)
print(f"Total reward: {trajectories['total_reward']}")
manager.task.close()

