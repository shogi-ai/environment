"""
Agent for playing Shogi using a Deep Q-Network (DQN). Handles model initialization,
action selection, memory management, and training using experience replay.
"""

import os
import random

import numpy as np
import torch

from env import ShogiEnv
from model.deep_q_network import DQN


class ShogiAgent:
    """
    Agent for playing Shogi using a Deep Q-Network (DQN). Handles model initialization,
    action selection, memory management, and training using experience replay.

    Attributes:
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum value for epsilon.
        learning_rate (float): Learning rate for the optimizer.
        target_network (DQN): Target network for stabilizing training.
        optimizer (torch.optim.Optimizer): Optimizer for training the network.
    """

    def __init__(self):
        """
        Initializes the ShogiAgent with parameters, networks, loss function, and optimizer.
        """
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.learning_rate = 1e-03

        self.target_network = DQN()

        self.optimizer = torch.optim.Adam(
            self.target_network.parameters(), lr=self.learning_rate
        )

    def reset(self):
        self.epsilon = 1
        self.epsilon_decay = 0.99

    @staticmethod
    def get_move_index(move):
        """
        Converts a move to an index.

        Args:
            move: A move object with from_square and to_square attributes.

        Returns:
            int: Index representing the move.
        """
        index = 81 * move.from_square + move.to_square
        return index

    def mask_and_valid_moves(self, env: ShogiEnv) -> (np.array, dict):
        """
        Get the mask and valid moves for the current player.

        Returns:
            tuple: Tuple containing the mask and valid moves for the current player.
        """
        mask = np.zeros((81, 81))
        valid_moves_dict = {}

        legal_moves = env.get_legal_moves()

        for move in legal_moves:
            mask[move.from_square, move.to_square] = 1
            index = 81 * move.from_square + move.to_square
            valid_moves_dict[index] = move

        return mask, valid_moves_dict

    def select_action(self, env: ShogiEnv):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            env: Environment object with mask_and_valid_moves and get_state methods.

        Returns:
            tuple: Move index, chosen move, current state, valid moves.
        """
        valid_moves, valid_move_dict = self.mask_and_valid_moves(env)
        current_state = env.get_observation()

        if random.uniform(0, 1) >= self.epsilon:
            valid_moves_tensor = torch.from_numpy(valid_moves).float().unsqueeze(0)
            current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0)
            valid_moves_tensor = valid_moves_tensor.view(
                current_state_tensor.size(0), -1
            )
            policy_values = self.target_network(
                current_state_tensor, valid_moves_tensor
            )
            chosen_move_index = int(policy_values.max(1)[1].view(1, 1))
            chosen_move = valid_move_dict[chosen_move_index]
        else:
            chosen_move = env.sample_action()

        return chosen_move

    def adaptive_e_greedy(self):
        """
        Decays the exploration rate epsilon.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, path: str):
        """
        Saves the model parameters to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.target_network.state_dict(), path)

    def get_model(self, path: str):
        """
        Get the model parameters from the specified path.

        Args:
            path (str): Get the model.
        """
        if os.path.isfile(path):
            model_dict = torch.load(path)
            self.target_network.load_state_dict(model_dict)
