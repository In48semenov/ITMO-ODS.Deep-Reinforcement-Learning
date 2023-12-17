from typing import Dict, List, Union

import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm


class DeepCEM(nn.Module):

    def __init__(self, state_dim: int, action_n: int, lr: float=1e-2):
        super().__init__()

        self.state_dim = state_dim
        self.action_n = action_n
        self.eps = 1

        self.network = nn.Sequential(
            nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, self.action_n), nn.Softmax(dim=0))
        )

        self.softmax = nn.Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input: torch.Tensor):
        return self.network(_input)
    
    def get_action(self, state: np.ndarray):
        state = torch.FloatTensor(state)
        action_softmax = self.forward(state)
        action_softmax = action_softmax.detach().numpy()
        action = np.random.choice(self.action_n, p=action_softmax)
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
    max_len: int = 500,
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



def deep_cem_fit(env, state_dim, action_n, lr, episode_n, trajectory_n, q_param, t_max: int = 500):
    agent = DeepCEM(state_dim, action_n, lr)

    total_rewards = []
    print("Start training DeepCEM...")
    loop = tqdm(range(episode_n), position=0, leave=True)
    for episode_idx in loop:
        trajectories = [get_trajectory(env, agent, t_max) for _ in range(trajectory_n)]

        curr_total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
        mean_total_reward = np.mean(curr_total_rewards)

        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories) > 0:
            _ = agent.update_policy(elite_trajectories, return_loss=False)

        loop.set_description(f"Episode {episode_idx}")
        loop.set_postfix(dict(mean_total_reward=mean_total_reward))

        total_rewards.extend(curr_total_rewards)
    print("Training finished.")
    return total_rewards
