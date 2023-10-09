import time
from typing import Dict, List, Union, Tuple

import gym
import numpy as np

from utils.clearml_manager import Manager


# GAME PARAMETERS
env = gym.make('Taxi-v3')
texi_state_n = 25
passenger_state_n = 5
destination_state_n = 4
STATE_N = texi_state_n * passenger_state_n * destination_state_n
ACTION_N = 6
MAX_LEN = 10000

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.CEM"


class CrossEntropyAgent:

    def __init__(self, state_n, action_n, q_param: float):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        self.q_param = q_param

    def get_action(self, state) -> int:
        action = np.random.choice(
            np.arange(self.action_n), p=self.model[state],
        )
        return int(action)

    def _get_elite_trajectories(
        self,
        trajectories: List[Dict[str, List[int]]],
        total_rewards: List[int],
    ) -> List[Dict[str, List[int]]]:
        quantile = np.quantile(total_rewards, self.q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory["rewards"])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        return elite_trajectories

    def fit(
        self,
        trajectories: List[Dict[str, List[int]]],
        total_rewards: List[int],
    ) -> None:
        elite_trajectories = self._get_elite_trajectories(
            trajectories, total_rewards
        )
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"],
                                     trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_trajectory(
    env,
    agent,
    max_len: int = 10000,
    visualize: bool = False,
    return_max_iter: bool = False
) -> Union[Dict, Tuple[Dict, int]]:
    trajectory = {
        "states": [],
        "actions": [],
        "rewards": [],
    }

    # An observation is an integer that encodes the corresponding state
    state = env.reset()
    iter_end = 0
    for iter_idx in range(max_len):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        state, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)

        if visualize:
            time.sleep(0.5)
            env.render()

        iter_end = iter_idx
        if done:
            break

    if return_max_iter:
        return trajectory, iter_end
    else:
        return trajectory


# MODEL PARAMETERS
ITERATION_N = 25
TRAJECTORY_N = 500
# Траекторий, которые дойдут до финиша мало => q можно брать большим
Q_PARAM = 0.6

TASK_NAME = f"Task-1-v11"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(
    {
        "ITERATION_N": ITERATION_N,
        "TRAJECTORY_N": TRAJECTORY_N,
        "Q_PARAM": Q_PARAM
    }
)


agent = CrossEntropyAgent(state_n=STATE_N, action_n=ACTION_N, q_param=Q_PARAM)

for iteration in range(ITERATION_N):
    # policy evaluation
    trajectories = [
        get_trajectory(env, agent, MAX_LEN) for _ in range(TRAJECTORY_N)
    ]
    total_rewards = [
        np.sum(trajectory["rewards"]) for trajectory in trajectories
    ]
    print(
        f"Iteration: {iteration}; Mean Total Reward: {np.mean(total_rewards)}"
    )
    manager.report_metrics(
        "Training", "Mean Total Reward", np.mean(total_rewards), iteration
    )

    # policy improvement
    agent.fit(trajectories, total_rewards)

trajectory, iter_end = get_trajectory(
    env, agent, max_len=1000, visualize=True, return_max_iter=True
)
print(f"Total reward: {np.sum(trajectory['rewards'])}")

manager.report_metrics(
    "Test", "Total reward", np.sum(trajectory['rewards']), iteration=0
)
manager.report_metrics("Game over", "Iter end", iter_end, iteration=0)
manager.task.close()

