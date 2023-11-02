import sys
sys.path.append("../")

import time
import typing as tp

import numpy as np

from frozen_lake import FrozenLakeEnv
from hw.utils.clearml_manager import Manager


# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.PolicyValueIterations"

env = FrozenLakeEnv()

def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * env.get_reward(state, action, next_state)
                q_values[state][action] += gamma * env.get_transition_prob(state, action, next_state) * v_values[next_state]
    return q_values


def init_v_values():
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def policy_evaluation_step(v_values, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            new_v_values[state] = np.max([q_values[state][action] for action in env.get_possible_actions(state)])
        else:
            new_v_values[state] = 0
    return new_v_values


def policy_evaluation(gamma, eval_iter_n):
    v_values = init_v_values()
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(v_values, gamma)
    q_values = get_q_values(v_values, gamma)
    return q_values

def policy_improvement(q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state): 
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy


def fit(gamma, eval_iter_n) -> tp.Dict[tp.Tuple[int], tp.Dict[str, int]]:
    q_values = policy_evaluation(gamma, eval_iter_n)
    policy = policy_improvement(q_values)
    return policy

eval_iter_n = 100
GAMMA = [np.round(gamma, 3) for gamma in np.linspace(0, 0.9, 27)]
GAMMA += [np.round(gamma, 3) for gamma in np.linspace(0.901, 1, 100)]
COUNT_RUN_ENV = 1000
COUNT_EPISODE = 1000

TASK_NAME = "Value_Iteration"
manager = Manager(project=PROJECT_NAME, task_name=TASK_NAME)

mean_rewards = []
mean_access_to_env = []
for idx_gamma, gamma in enumerate(GAMMA):
    policy = fit(gamma, eval_iter_n)

    total_rewards = []
    access_to_env = []
    for _ in range(COUNT_RUN_ENV):
        total_reward = 0
        state = env.reset()
        for idx in range(COUNT_EPISODE):
            action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        
        total_rewards.append(total_reward)
        access_to_env.append(idx)

    mean_rewards.append(np.mean(total_rewards))
    mean_access_to_env.append(np.mean(access_to_env))

    print(
        f" === IDX: {idx_gamma}; Gamma: {gamma}; "
        f"Mean Rewards: {mean_rewards[-1]}; "
        f"Mean Access: {mean_access_to_env[-1]}"
    )
    
    manager.report_metrics(
        "Gamma-Reward", "Mean Reward", mean_rewards[-1], idx_gamma
    )
    manager.report_metrics(
        "Access-to-Environment", "Mean access", mean_access_to_env[-1], idx_gamma
    )
    manager.report_metrics(
        "Gamma Values", "Gamma", gamma, idx_gamma
    )

best_idx_gamma = np.array(mean_rewards).argmax()
best_gamma = GAMMA[best_idx_gamma]

best_idx_access = np.array(mean_access_to_env).argmin()
best_access_gamma = GAMMA[best_idx_access]

# TEST
# With best gamma
policy = fit(best_gamma, eval_iter_n)
total_reward = 0
state = env.reset()
for idx in range(COUNT_EPISODE):
    action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
    state, reward, done, _ = env.step(action)
    total_reward += reward

    env.render()

    if done:
        break

manager.report_metrics(
    f"Gamma-Reward-Test (GAMMA: {best_gamma} with max Reward)", "Reward", total_reward, 0
)
manager.report_metrics(
    f"Access-to-Environment-Test (GAMMA: {best_gamma} with max Reward)", "Access", idx, 0
)

# With best access
policy = fit(best_access_gamma, eval_iter_n)
total_reward = 0
state = env.reset()
for idx in range(COUNT_EPISODE):
    action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
    state, reward, done, _ = env.step(action)
    total_reward += reward

    env.render()
    time.sleep(0.5)

    if done:
        break

manager.report_metrics(
    f"Gamma-Reward-Test (GAMMA: {best_gamma} with min Access to ENV)", "Reward", total_reward, 0
)
manager.report_metrics(
    f"Access-to-Environment-Test (GAMMA: {best_gamma} with min Access to ENV)", "Access", idx, 0
)

manager.task.close()