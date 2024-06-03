from src.agent.shogi_agent import ShogiAgent
from src.environment.env import ShogiEnv
from src.game.ai_action import ActionTaken


def game(environment: ShogiEnv, agent: ShogiAgent) -> list[ActionTaken]:
    action_history = []
    terminated = False
    truncated = False
    state = environment.reset()
    agent.reset()

    while not terminated and not truncated:
        initial_state = state
        initial_moves, _ = agent.mask_and_valid_moves(environment)

        # Take action
        action, _ = agent.select_action(environment)
        state, reward, terminated, truncated, _ = environment.step(action)

        # Get new possible moves
        next_moves, _ = agent.mask_and_valid_moves(environment)

        # Update the player
        ai_action = ActionTaken(
            priority=reward,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            current_state=initial_state,
            current_moves=initial_moves,
            next_state=state,
            next_moves=next_moves,
        )
        action_history.append(ai_action)
    return action_history
