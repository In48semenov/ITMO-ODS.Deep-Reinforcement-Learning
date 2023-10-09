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

    def __init__(
        self,
        state_n: int,
        action_n: int,
        q_param: float,
        laplace_smoothing: bool = False,
        policy_smoothing: bool = False,
        lambda_value: Union[int, float, None] = None
    ):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        self.q_param = q_param
        self.laplace_smoothing = laplace_smoothing
        self.policy_smoothing = policy_smoothing
        self.lambda_value = lambda_value

    def calc_deterministic_policies(self, n_deterministic: int) -> None:
        self.deterministic_policies = []
        for idx in range(n_deterministic):
            curr_deterministic_policy = []
            for state in range(self.state_n):
                action = np.random.choice(
                    np.arange(self.action_n), p=self.model[state],
                )
                curr_deterministic_policy.append(action)
            self.deterministic_policies.append(curr_deterministic_policy)

        return None

    def get_action(
        self,
        state,
        idx_deterministic_policy: Union[int, None] = None
    ) -> int:
        if idx_deterministic_policy is not None:
            action = self.deterministic_policies[idx_deterministic_policy][
                state]
        else:
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
            total_reward = sum(trajectory["rewards"])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        return elite_trajectories

    def _get_simple(self, state, new_model) -> np.ndarray:
        if np.sum(new_model[state]) > 0:
            new_model[state] /= np.sum(new_model[state])
        else:
            new_model[state] = self.model[state].copy()

        return new_model

    def _get_laplace_smoothing(self, state, new_model) -> np.ndarray:
        new_model[state] += np.full_like(
            new_model[state],
            fill_value=self.lambda_value
        )
        new_model[state] /= np.sum(new_model[state])
        return new_model

    def _get_policy_smoothing(self, state, new_model) -> np.ndarray:
        new_model = self._get_simple(state, new_model)
        new_model[state] = self.lambda_value * new_model[state] + (
            1 - self.lambda_value
        ) * self.model[state]

        return new_model

    def fit(
        self,
        elite_trajectories: List[Dict[str, List[int]]],
        # total_rewards: List[in
    ) -> None:
        # elite_trajectories = self._get_elite_trajectories(
        #     trajectories, total_rewards
        # )
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"],
                                     trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if not self.laplace_smoothing and not self.policy_smoothing:
                new_model = self._get_simple(state, new_model)
            elif self.lambda_value:
                if self.laplace_smoothing:
                    new_model = self._get_laplace_smoothing(state, new_model)
                elif 0 < self.lambda_value <= 1:
                    new_model = self._get_policy_smoothing(state, new_model)
                else:
                    raise(
                        "Lambda value for `policy smoothing`"
                        "must be in the beam (0, 1]."
                    )
            else:
                raise("Lambda must be define.")

        self.model = new_model
        return None


def get_trajectory(
    env,
    agent,
    max_len: int = 10000,
    idx_deterministic_policy: Union[int, None] = None,
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

        if idx_deterministic_policy is not None:
            action = agent.get_action(
                state, idx_deterministic_policy=idx_deterministic_policy
            )
        else:
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
DETERMINISTIC_POLICY_N = 100
# Траекторий, которые дойдут до финиша мало => q можно брать большим
Q_PARAM = 0.3
LAPLACE_SMOOTHING = False
POLICY_SMOOTHING = True
LAMBDA_VALUE = 0.9

TASK_NAME = f"Task-3-Deterministic-Simple-v3"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
manager.log_params(
    {
        "ITERATION_N": ITERATION_N,
        "TRAJECTORY_N": TRAJECTORY_N,
        "DETERMINISTIC_POLICY_N": DETERMINISTIC_POLICY_N,
        "Q_PARAM": Q_PARAM,
        "ALGORITHM": "Policy_Smoothing",
    }
)

agent = CrossEntropyAgent(
    state_n=STATE_N,
    action_n=ACTION_N,
    q_param=Q_PARAM,
    laplace_smoothing=LAPLACE_SMOOTHING,
    policy_smoothing=POLICY_SMOOTHING,
    lambda_value=LAMBDA_VALUE
)

for iteration in range(ITERATION_N):

    # sample deterministic policies
    agent.calc_deterministic_policies(DETERMINISTIC_POLICY_N)

    elite_trajectories_by_policies = []
    rewards_by_policies = []
    for idx_policy in range(DETERMINISTIC_POLICY_N):
        # policy evaluation
        trajectories = [
            get_trajectory(
                env, agent, MAX_LEN, idx_deterministic_policy=idx_policy
            )
            for _ in range(TRAJECTORY_N)
        ]
        total_rewards = [
            np.sum(trajectory["rewards"]) for trajectory in trajectories
        ]
        print(
            f"Iteration: {iteration}; "
            f"Deterministic idx: {idx_policy}; "
            f"Mean Total Reward: {np.mean(total_rewards)}"
        )
        rewards_by_policies.append(np.mean(total_rewards))

        # policy improvement
        quantile = np.quantile(total_rewards, Q_PARAM)
        curr_elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                curr_elite_trajectories.append(trajectory)

        elite_trajectories_by_policies.extend(curr_elite_trajectories)

    manager.report_metrics(
        "Training", "Mean Total Reward", np.mean(rewards_by_policies), iteration
    )

    agent.fit(elite_trajectories_by_policies)

trajectory, iter_end = get_trajectory(
    env, agent, max_len=1000, visualize=True, return_max_iter=True
)
print(f"Total reward: {sum(trajectory['rewards'])}")

manager.report_metrics(
    "Test", "Total reward", np.sum(trajectory['rewards']), iteration=0
)
manager.report_metrics("Game over", "Iter end", iter_end, iteration=0)
manager.task.close()

