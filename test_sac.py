import gym
import torch
import numpy as np
from common.sac import SAC, ReplayBuffer

def test_mountaincar():
    # Environment setup
    env_name = "MountainCarContinuous-v0"
    env_fn = lambda: gym.make(env_name)
    
    # Create environment to get dimensions
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=int(1e6),
        device=device
    )
    
    # Create SAC agent
    sac = SAC(
        env_fn=env_fn,
        replay_buffer=replay_buffer,
        hidden_sizes=[256, 256],  # Typical hidden layer sizes
        lr=3e-4,
        batch_size=256,
        start_steps=1000,
        update_after=1000,
        max_ep_len=1000,
        device=device,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        automatic_alpha_tuning=True
    )
    
    # Train the agent
    training_results = sac.learn_mujoco(print_out=True)
    
    # Test the trained agent
    test_returns = []
    for _ in range(10):
        o = env.reset()
        ep_ret = 0
        done = False
        while not done:
            a = sac.get_action(o, deterministic=True)
            o, r, done, _ = env.step(a)
            ep_ret += r
        test_returns.append(ep_ret)
    
    print(f"\nFinal Test Results over {len(test_returns)} episodes:")
    print(f"Average Return: {np.mean(test_returns):.2f}")
    print(f"Std Return: {np.std(test_returns):.2f}")
    
if __name__ == "__main__":
    test_mountaincar() 