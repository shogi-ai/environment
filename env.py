"""Shogi environment for reinforcement learning."""

import random
import shogi

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ShogiEnv(gym.Env):
    """
    Shogi environment for reinforcement learning.

    This environment simulates a game of Shogi, a Japanese variant of chess.
    It provides an action space representing all possible moves and an observation space representing the Shogi board.
    The game continues until a player wins, the game reaches a stalemate, or a specified number of moves is reached.

    Attributes:
        board: Shogi board object.
        action_space: MultiDiscrete space representing all possible moves.
        observation_space: MultiDiscrete space representing the Shogi board.
        move: Number of moves made in the game.

    Methods:
        __init__: Initialize the Shogi environment.
        reset: Reset the environment to its initial state.
        sample_action: Sample a random legal move.
        step: Take a step in the environment based on the action.
        _get_observation: Get the current observation of the Shogi board.
        render: Render the current state of the Shogi board.
    """

    def __init__(self):
        """
        Initialize the Shogi environment.

        Initializes the Shogi board, action space, and observation space.
        """
        super(ShogiEnv, self).__init__()
        self.board = shogi.Board()

        # Action space represents all possible moves in Shogi
        self.action_space = spaces.MultiDiscrete(
            [81, 81]
        )  # From square (x, y) to square (x, y)
        self.action_space.sample = self.sample_action

        # Observation space represents the Shogi board
        self.observation_space = spaces.MultiDiscrete(
            [81, 17]
        )  # 9x9 board with 17 possible pieces

        # Keep track of moves
        self.move = 0
        self.max_moves = 200

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, any] | None = None,
    ):
        """
        Reset the environment to its initial state.

        Resets the Shogi board and move counter.

        Args:
            seed: Random seed for environment (not used).
            options: Additional options (not used).

        Returns:
            tuple: Initial observation.
        """
        self.board = shogi.Board()
        self.move = 0
        return (self._get_observation(),)

    def sample_action(self):
        """
        Sample a random legal move for the specified player.

        Args:
            player (int): Player ID (0 or 1).

        Returns:
            shogi.Move: Random legal move for the specified player.
        """

        def select_move():
            """Select a random move"""
            # Move to random legal position
            piece_legal_moves = self.board.generate_legal_moves()

            # No legal moves for this piece
            if len(piece_legal_moves) == 0:
                return None

            # Return random legal move
            return random.choice(piece_legal_moves)

        while True:
            piece_to = select_move()
            if piece_to is not None:
                return piece_to

    def step(self, action: shogi.Move):
        """
        Take a step in the environment based on the action.

        Args:
            action (shogi.Move): Action to take (legal move).

        Returns:
            tuple: Tuple containing the next observation, reward, termination status, truncation status, and additional info.
        """
        if action not in self.board.legal_moves:
            raise ValueError("Illegal move")

        self.move += 1
        self.board.push(action)

        reward = 0.0
        terminated = False
        truncated = False

        if self.move >= self.max_moves:
            truncated = True

        # Direct if game is won? reward + 1, terminated = True
        if self.board.is_checkmate():
            reward = 10
            terminated = True
        elif self.board.is_stalemate():
            reward = 2
            terminated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """
        Get the current bitboard of the Shogi board.

        Returns:
            list: List representing the current bitboard of the Shogi board.
        """
        PIECE_SYMBOLS = [
            "",
            "p",
            "l",
            "n",
            "s",
            "g",
            "b",
            "r",
            "k",
            "+p",
            "+l",
            "+n",
            "+s",
            "+b",
            "+r",
        ]
        pieces_in_board = [self.board.piece_at(i) for i in range(81)]
        indices = []

        def print_bitboard(piece, pieces_in_board):
            output = []
            for _, x in enumerate(pieces_in_board):
                if str(x).lower() == piece:
                    output.append(1)
                else:
                    output.append(0)
            return np.reshape(output, (9, 9))

        for piece in PIECE_SYMBOLS:
            if piece == "":
                continue
            indices.append(print_bitboard(piece, pieces_in_board))

        return np.array(indices)

    def render(self):
        """
        Render the current state of the Shogi board.
        """
        print("=" * 25)
        print(self.board)
