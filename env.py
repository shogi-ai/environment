"""Shogi environment for reinforcement learning."""

import random
import shogi

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reward_table import PIECE_REWARDS, CHECKMATE, STALEMATE


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
        mask_and_valid_moves: Get the mask and valid moves for the current player.
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
        return self._get_observation()

    def sample_action(self):
        """
        Sample a random legal move for the specified player.

        Returns:
            shogi.Move: Random legal move for the specified player.
        """

        def select_move():
            """Select a random move"""
            # Move to random legal position
            generator = self.board.generate_legal_moves()
            piece_legal_moves = [move for move in generator]

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
        
        state = self._get_observation()

        self.move += 1
        reward = 0.0
        terminated = False
        truncated = False

        piece = self.board.piece_at(action.to_square)
        if piece:
            piece_name = self._get_piece_name(piece.piece_type)
            reward = PIECE_REWARDS[piece_name]

        self.board.push(action)

        if self.move >= self.max_moves:
            truncated = True

        # Direct if game is won? reward + 1, terminated = True
        if self.board.is_checkmate():
            reward = CHECKMATE
            terminated = True
        elif self.board.is_stalemate():
            reward = STALEMATE
            terminated = True

        if reward:
            print(reward)

        return self._get_observation(), reward, terminated, truncated, state, {}

    def _get_observation(self):
        """
        Get the current bitboard of the Shogi board.

        Returns:
            list: List representing the current bitboard of the Shogi board.
        """
        piece_symbols = [
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

        for piece in piece_symbols:
            if piece == "":
                continue
            indices.append(print_bitboard(piece, pieces_in_board))

        return (np.array(indices),)

    def render(self):
        """
        Render the current state of the Shogi board.
        """
        print("=" * 25)
        print(self.board)

    def mask_and_valid_moves(self):
        """
        Get the mask and valid moves for the current player.

        Returns:
            tuple: Tuple containing the mask and valid moves for the current player.
        """
        mask = np.zeros((81, 81))
        valid_moves_dict = {}

        for move in self.board.legal_moves:
            mask[move.from_square, move.to_square] = 1
            index = 81 * (move.from_square) + (move.to_square)
            valid_moves_dict[index] = move

        return mask, valid_moves_dict

    @staticmethod
    def _get_piece_name(piece_type: int) -> str:
        """
        Get piece name based on piece type.

        Args:
            piece_type (int): The piece type you want the name of.

        Return:
            str: The name of the piece
        """
        piece_types = [
            "PAWN",
            "LANCE",
            "KNIGHT",
            "SILVER",
            "GOLD",
            "BISHOP",
            "ROOK",
            "KING",
            "PROM_PAWN",
            "PROM_LANCE",
            "PROM_KNIGHT",
            "PROM_SILVER",
            "PROM_BISHOP",
            "PROM_ROOK",
        ]
        piece = piece_types[piece_type - 1]
        return piece
