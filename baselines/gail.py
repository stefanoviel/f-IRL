import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env


def parse_args():
    """
    Parse command-line arguments for environment name, number of expert trajectories, etc.
    """
    parser = argparse.ArgumentParser(description="Train GAIL on a MuJoCo environment from raw .pt expert data.")
    parser.add_argument("--env_name", type=str, default="Ant-v2",
                        help="MuJoCo Gym environment ID, e.g. Ant-v2.")
    parser.add_argument("--num_expert_trajs", type=int, default=5,
                        help="Number of expert episodes to use.")
    parser.add_argument("--train_steps", type=int, default=100000,
                        help="Number of GAIL training timesteps (generator steps).")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes (before/after training).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def load_expert_trajectories(env_name: str, num_trajs: int):
    """
    Loads expert states and actions from .pt files:
      - states: expert_data/states/{env_name}.pt (shape: [N, T, state_dim])
      - actions: expert_data/actions/{env_name}.pt (shape: [N, T, act_dim])

    Returns a list of `Trajectory` objects, each a full episode.
    """
    states_path = f"expert_data/states/{env_name}.pt"
    actions_path = f"expert_data/actions/{env_name}.pt"

    # Load raw expert data
    expert_states_all = torch.load(states_path).numpy()  # shape: (N, T, state_dim)
    expert_actions_all = torch.load(actions_path).numpy() # shape: (N, T, act_dim)

    # Keep only the first `num_trajs` trajectories
    expert_states = expert_states_all[:num_trajs]
    expert_actions = expert_actions_all[:num_trajs]

    # Build a list of Trajectory objects
    trajectories = []
    for i in range(expert_states.shape[0]):
        states_i = expert_states[i]  # shape: (T, state_dim)
        actions_i = expert_actions[i]  # shape: (T, act_dim)

        # We assume each trajectory is "done" at the final step
        # imitation.data.types.Trajectory: obs, acts, infos, terminal_obs
        # We don't strictly need 'infos', so we can supply an empty list.

        # obs is shape (T, obs_dim)
        # acts is shape (T, act_dim)
        # final obs can be appended as terminal_obs
        obs = states_i[:-1]
        next_obs = states_i[1:]
        terminal_obs = states_i[-1]

        # If you want to store transitions in a single trajectory:
        # - The length of obs and acts must match
        # - The final state is terminal_obs
        # For T steps, typically we have T-1 "transitions" if fully Markov.
        # But we can do a simpler approach:
        #   obs = states_i[:-1], acts = actions_i[:-1], terminal_obs= states_i[-1]
        # This means the last transition is from states_i[-2] -> states_i[-1].

        # We'll do T steps in a trajectory. So we'll let
        # obs = states_i[0:-1], acts = actions_i[0:-1],
        # terminal_obs = states_i[-1].

        # If you have the same shape for states/actions (T, ...),
        # you can keep them "aligned". We'll do:
        # obs = states_i[0:T-1], acts = actions_i[0:T-1].
        # That leaves the final state as terminal_obs.

        # Or if your environment doesn't need an offset, you can keep them 1:1:
        #   obs = states_i[0:T], acts = actions_i[0:T], but then you need to define
        #   terminal_obs as states_i[-1], with the same length for obs & acts.
        # Because GAIL doesn't strictly require next_obs, either approach is fine.

        # We'll adopt "T-1" approach here:
        traj = Trajectory(
            obs=obs,           # shape (T-1, obs_dim)
            acts=actions_i[:-1],
            infos=[{}]*(len(obs)),  # dummy infos
            terminal_obs=terminal_obs
        )
        trajectories.append(traj)

    return trajectories


def main():
    args = parse_args()

    # Logging directory: logs/{env_name}/exp_{num_expert_trajs}/gail
    log_dir = f"logs/{args.env_name}/exp_{args.num_expert_trajs}/gail"
    os.makedirs(log_dir, exist_ok=True)

    # Create a vectorized environment
    env_fn = lambda: gym.make(args.env_name)
    env = env_fn()

    # Load expert trajectories
    trajectories = load_expert_trajectories(args.env_name, args.num_expert_trajs)

    # Create a new PPO learner (the "generator") that GAIL will train
    learner = PPO(
        policy=MlpPolicy,
        env=env,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=4e-4,
        gamma=0.95,
        n_epochs=5,
        seed=args.seed,
    )

    # Create a reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Initialize the GAIL trainer
    # We pass in the demonstration data as a list of Trajectory objects
    gail_trainer = GAIL(
        demonstrations=trajectories,    # our loaded expert episodes
        demo_batch_size=1024,           # Discriminator batch size
        gen_replay_buffer_capacity=4096, # Generator replay buffer
        n_disc_updates_per_round=4,     # Discriminator updates per PPO update
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    # Evaluate policy (untrained) before GAIL
    env.reset(seed=args.seed)
    rewards_before, _ = evaluate_policy(
        learner,
        env,
        n_eval_episodes=args.eval_episodes,
        return_episode_rewards=True,
    )
    mean_before = np.mean(rewards_before)

    # Train GAIL
    gail_trainer.train(args.train_steps)

    # Evaluate policy after GAIL training
    env.reset(seed=args.seed)
    rewards_after, _ = evaluate_policy(
        learner,
        env,
        n_eval_episodes=args.eval_episodes,
        return_episode_rewards=True,
    )
    mean_after = np.mean(rewards_after)

    print(f"Mean return before training: {mean_before:.2f}")
    print(f"Mean return after training : {mean_after:.2f}")

    # Simple bar plot
    plt.figure(figsize=(6,4))
    plt.bar(["Before", "After"], [mean_before, mean_after],
            color=["red", "blue"], alpha=0.6)
    plt.ylabel("Mean Return")
    plt.title(f"GAIL on {args.env_name}\n({args.num_expert_trajs} expert episodes)")
    plot_path = os.path.join(log_dir, "gail_return_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved reward comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()


# python gail_mujoco_from_pt.py --env_name Ant-v5 --num_expert_trajs 16