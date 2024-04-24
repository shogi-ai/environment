"""Test the Shogi environment."""

import gymnasium as gym

from env import ShogiEnv

# Register the environment
gym.register(id="Shogi-v0", entry_point="env:ShogiEnv", kwargs={})

# Setup the environment
env: ShogiEnv = gym.make("Shogi-v0")
obs = env.reset()
env.render()

while True:
    # Player 1
    action = env.sample_action()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    env.render()

    # Player 2
    action = env.sample_action()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    env.render()
