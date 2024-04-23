"""Shogi environment for reinforcement learning."""

import random
import shogi

import gymnasium as gym
from gymnasium import spaces
from shogi import Piece


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
        _player_0: Get indices of pieces belonging to player 0.
        _player_1: Get indices of pieces belonging to player 1.
        sample_action: Sample a random legal move for the specified player.
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

    def _player_0(self):
        """
        Get indices of pieces belonging to player 0.

        Returns:
            list: Indices of pieces belonging to player 0.
        """
        pieces = [self.board.piece_at(i) for i in range(81)]

        indices = []
        for i, char in enumerate(pieces):
            if char is not None:
                if char.color == 0:
                    indices.append(i)
        return indices

    def _player_1(self):
        """
        Get indices of pieces belonging to player 1.

        Returns:
            list: Indices of pieces belonging to player 1.
        """
        pieces = [self.board.piece_at(i) for i in range(81)]

        indices = []
        for i, char in enumerate(pieces):
            if char is not None:
                if char.color == 1:
                    indices.append(i)
        return indices

    def sample_action(self, player: int):
        """
        Sample a random legal move for the specified player.

        Args:
            player (int): Player ID (0 or 1).

        Returns:
            shogi.Move: Random legal move for the specified player.
        """
        # Get all piece locations for player
        if player == 0:
            pieces = self._player_0()
        elif player == 1:
            pieces = self._player_1()
        else:
            raise ModuleNotFoundError("Player not found")

        def select_move(pieces: list[Piece]):
            """Select a piece and a random move"""
            # chose random piece
            piece = random.choice(pieces)

            # Move to random legal position
            legal_moves = self.board.generate_legal_moves()
            piece_legal_moves = [
                move for move in legal_moves if move.from_square == piece
            ]

            # No legal moves for this piece
            if len(piece_legal_moves) == 0:
                return None

            # Return random legal move
            return random.choice(piece_legal_moves)

        while True:
            piece_to = select_move(pieces)
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

        if self.move >= 200:
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
        Get the current observation of the Shogi board.

        Returns:
            list: List representing the current state of the Shogi board.
        """
        return [self.board.piece_at(i) for i in range(81)]

    def render(self):
        """
        Render the current state of the Shogi board.
        """
        print("=" * 25)
        print(self.board)
