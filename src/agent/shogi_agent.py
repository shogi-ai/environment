"""
Agent for playing Shogi using a Deep Q-Network (DQN). Handles model initialization,
action selection, memory management, and training using experience replay.
"""

import os
import random

import numpy as np
import torch
from shogi import Move

from src.environment.env import ShogiEnv
from src.agent.deep_q_network import DQN
import torch.nn.functional as F

from torch.utils.data import Dataset


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.memory_capacity = 10000

        self.q_network = DQN()
        self.target_network = self.q_network

        self.optimizer = torch.optim.Adam(
            self.target_network.parameters(), lr=self.learning_rate
        )
        self.memory = ReplayMemory(self.memory_capacity)

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

    def select_action(self, env: ShogiEnv) -> (Move, int):
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
            try:
                chosen_move = valid_move_dict[chosen_move_index]
            except Exception:
                chosen_move = env.sample_action()
                chosen_move_index = 81 * chosen_move.from_square + chosen_move.to_square
        else:
            chosen_move = env.sample_action()
            chosen_move_index = 81 * chosen_move.from_square + chosen_move.to_square

        return chosen_move, chosen_move_index

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

    def train_model(self, action, reward, done, current_state, next_state) -> float:
        self.memory.push(current_state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return 0

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(batch[0], dtype=torch.float32)
        action_batch = torch.tensor(batch[1], dtype=torch.int64)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32)
        done_batch = torch.tensor(batch[4], dtype=torch.float32)

        state_action_values = self.q_network(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        next_state_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (
            next_state_values * self.gamma * (1 - done_batch)
        ) + reward_batch

        loss = F.mse_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)
