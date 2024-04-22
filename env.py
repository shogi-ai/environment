import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import shogi


class ShogiEnv(gym.Env):
    def __init__(self):
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
        self.board = shogi.Board()
        self.move = 0
        return (self._get_observation(),)

    def _player_0(self):
        pieces = [self.board.piece_at(i) for i in range(81)]

        indices = []
        for i, char in enumerate(pieces):
            if char is not None:
                if char.color == 0:
                    indices.append(i)
        return indices

    def _player_1(self):
        pieces = [self.board.piece_at(i) for i in range(81)]

        indices = []
        for i, char in enumerate(pieces):
            if char is not None:
                if char.color == 1:
                    indices.append(i)
        return indices

    def sample_action(self, player: int):
        # Get all piece locations for player
        if player == 0:
            pieces = self._player_0()
        elif player == 1:
            pieces = self._player_1()
        else:
            raise Exception("Player not found")

        def select_move(board, pieces):
            # chose random piece
            piece = random.choice(pieces)

            # Move to random legal position
            legal_moves = board.generate_legal_moves()
            piece_legal_moves = [
                move for move in legal_moves if move.from_square == piece
            ]

            # No legal moves for this piece
            if len(piece_legal_moves) == 0:
                return None

            # Return random legal move
            return random.choice(piece_legal_moves)

        while True:
            piece_to = select_move(self.board, pieces)
            if piece_to is not None:
                return piece_to

    def step(self, action: shogi.Move):
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
        if self.board.is_game_over():
            reward = 1
            terminated = True
        if self.board.is_stalemate():
            reward = 0.5
            terminated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        return self.board.pieces

    def render(self):
        print("=" * 25)
        print(self.board)
