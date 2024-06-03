"""Test the Shogi environment."""

import time
import warnings
import gymnasium as gym

from src.environment.env import ShogiEnv
from src.agent.shogi_agent import ShogiAgent

warnings.filterwarnings("ignore")

gym.register(id="Shogi-v0", entry_point="environment.env:ShogiEnv")
env: ShogiEnv = gym.make("Shogi-v0")
agent = ShogiAgent()


def play_game(environment: ShogiEnv, player: ShogiAgent) -> (float, bool, bool):
    losses = []
    rewards = []
    terminated = False
    truncated = False
    environment.reset()
    agent.reset()

    while not terminated and not truncated:
        current_state = env.get_observation()

        # Take action
        action, mask_index = player.select_action(environment)
        state, reward, terminated, truncated, _ = environment.step(action)

        # Update the player
        player.adaptive_e_greedy()
        loss = player.train_model(
            mask_index,
            reward,
            (terminated or truncated),
            current_state,
            state,
        )

        rewards.append(reward)
        losses.append(loss)

    return rewards, terminated, truncated, losses


start = time.time()
# reward_list, _terminated, _truncated, loss_list = play_game(env, agent)
env.render()
print(env.get_observation())
end = time.time()
print(end - start)
