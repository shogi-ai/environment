"""Example environment"""

import gymnasium as gym
import pygame

# Register the environment
gym.register(
    id="MazeGame-v0", entry_point="example.env:MazeGameEnv", kwargs={"maze": None}
)
# Maze config

maze = [
    ["S", "", ".", "."],
    [".", "#", ".", "#"],
    [".", ".", ".", "."],
    ["#", ".", "#", "G"],
]
# Test the environment
env = gym.make("MazeGame-v0", maze=maze)
obs = env.reset()
env.render()

DONE = False
while True:
    pygame.event.get()
    action = env.action_space.sample()  # Random action selection
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    print("Reward:", reward)
    print("Done:", DONE)

    pygame.time.wait(200)
