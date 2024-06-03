import numpy as np


class ActionTaken:
    def __init__(
        self,
        priority: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        current_state: np.array,
        current_moves: int,
        next_state: np.array,
        next_moves: int,
    ):
        self.priority: int = priority
        self.action: int = action
        self.reward: float = reward
        self.terminated: bool = terminated
        self.truncated: bool = truncated
        self.current_state = current_state
        self.current_moves = current_moves
        self.next_state = next_state
        self.next_moves = next_moves

    def __str__(self):
        return self.priority

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority
