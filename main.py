import gymnasium as gym

from env import ShogiEnv

# Register the environment
gym.register(id="Shogi-v0", entry_point="env:ShogiEnv", kwargs={})

# Test the environment
env: ShogiEnv = gym.make("Shogi-v0")
obs = env.reset()
env.render()

while True:
    action = env.sample_action(0)

    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    env.render()

    # Player 2
    action = env.sample_action(1)

    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    env.render()
