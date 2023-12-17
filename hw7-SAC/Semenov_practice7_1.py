import sys
sys.path.append("../")

import gym
import numpy as np
import matplotlib.pyplot as plt

from deep_cross_entropy import deep_cem_fit
from dqn_soft_target import dqn_soft_fit
from ppo import ppo_fit
from sac import sac_fit
from utils.clearml_manager import Manager

# CLEAR-ML PARAMETERS
PROJECT_NAME = "RL.SAC"
TASK_NAME = f"Algorithm-Comparison-v1"

# ENVIRONMENT PARAMETERS
ENV = gym.make('CartPole-v1')
STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.n

EPISODE_N = 100
TRAJECTORY_N = 20
T_MAX = 500

# COMMON PARAMETERS
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-2
LR = 1e-4

REPEATS_N = 3

ALGORITHM_PARAMETERS = {
    "Deep-Cross-Entropy": dict(lr=LR, q_param=0.8),
    "DQN-Soft-Target-Update": dict(gamma=GAMMA, lr=LR, batch_size=BATCH_SIZE, epsilon_decrease=0.01, espilon_min=0.01, tau=TAU),
    "PPO": dict(epsilon=0.2, gamma=GAMMA, epoch_n=100, pi_lr=LR, v_lr=5e-2, batch_size=BATCH_SIZE),
    "SAC": dict(gamma=GAMMA, alpha=1e-3, tau=TAU, batch_size=BATCH_SIZE, pi_lr=LR, q_lr=5e-4, temperature=1),
}

MANADGER = Manager(project=PROJECT_NAME, task_name=TASK_NAME)
ALGORITHM_PARAMETERS["REPEATS_N"] = REPEATS_N
MANADGER.log_params(ALGORITHM_PARAMETERS)

TOTAL_REWARDS = {
    "Deep-Cross-Entropy": [],
    "DQN-Soft-Target-Update": [],
    "PPO": [],
    "SAC": [],
}


if __name__ == "__main__":
    for rep_idx in range(REPEATS_N):
        print(f" ============= REPEATS_N: {rep_idx + 1} ============= ")
        TOTAL_REWARDS["Deep-Cross-Entropy"].append(deep_cem_fit(
            env=ENV, 
            state_dim=STATE_DIM, 
            action_n=ACTION_DIM,
            episode_n=EPISODE_N,
            trajectory_n=TRAJECTORY_N,
            t_max=T_MAX,
            **ALGORITHM_PARAMETERS["Deep-Cross-Entropy"]))
        TOTAL_REWARDS["DQN-Soft-Target-Update"].append(
            dqn_soft_fit(
                env=ENV, 
                state_dim=STATE_DIM, 
                action_dim=ACTION_DIM, 
                episode_n=EPISODE_N, 
                t_max=T_MAX, 
                **ALGORITHM_PARAMETERS["DQN-Soft-Target-Update"]
            )
        )
        TOTAL_REWARDS["PPO"].append(
            ppo_fit(
                env=ENV, 
                state_dim=STATE_DIM, 
                action_dim=ACTION_DIM, 
                episode_n=EPISODE_N,
                trajectory_n=TRAJECTORY_N, 
                t_max=T_MAX, 
                **ALGORITHM_PARAMETERS["PPO"])
        )
        TOTAL_REWARDS["SAC"].append(
            sac_fit(
                env=ENV, 
                state_dim=STATE_DIM, 
                action_dim=ACTION_DIM, 
                episode_n=EPISODE_N,
                t_max=T_MAX, 
                **ALGORITHM_PARAMETERS["SAC"])
        )

    for alg in TOTAL_REWARDS:
        TOTAL_REWARDS[alg] = np.mean(TOTAL_REWARDS[alg], axis=0)
        plt.plot(TOTAL_REWARDS[alg], label=alg)

    plt.xlabel("Trajectories")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    MANADGER.report_plot(title="Training plot rewards", series="total_rewards", plt=plt)
    MANADGER.task.close()
