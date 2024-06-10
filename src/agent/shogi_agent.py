"""
Agent for playing Shogi using a Deep Q-Network (DQN). Handles model initialization,
action selection, memory management, and training using experience replay.
"""

import os
import random

import numpy as np
from shogi import Move
import torch
import torch.nn.functional as F
from torch import nn

from src.environment.env import ShogiEnv
from src.agent.deep_q_network import DQN


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

        self.gamma = 0.5
        self.learning_rate = 1e-03

        self.MEMORY_SIZE = 512
        self.MAX_PRIORITY = 1e06
        self.memory = []
        self.batch_size = 64

        self.q_network = DQN()
        self.target_network = self.q_network

        self.loss_function = nn.MSELoss()
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
        index = 81 * ShogiAgent.get_from_square(move) + move.to_square
        return index
    
    @staticmethod
    def get_from_square(move):
        """
        Converts a move to an index.

        Args:
            move: A move object with from_square and to_square attributes.

        Returns:
            int: Index representing the move.
        """
        # pieces = ["", "p", "l", "n", "s", "g", "b", "r"]
        if(move.from_square == None):
            # from_square max = 81
            return 81 + move.drop_piece_type - 1
            # now from_square max = 88
        return move.from_square

    def mask_and_valid_moves(self, env: ShogiEnv) -> (np.array, dict):
        """
        Get the mask and valid moves for the current player.

        Returns:
            tuple: Tuple containing the mask and valid moves for the current player.
        """
        mask = np.zeros((88, 81))
        valid_moves_dict = {}

        legal_moves = env.get_legal_moves()

        for move in legal_moves:
            mask[self.get_from_square(move), move.to_square] = 1
            valid_moves_dict[self.get_move_index(move)] = move

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
                chosen_move_index = 81 * self.get_from_square(chosen_move) + chosen_move.to_square
        else:
            chosen_move = env.sample_action()
            chosen_move_index = 81 * self.get_from_square(chosen_move) + chosen_move.to_square

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

    def remember(
        self,
        priority,
        action,
        reward,
        done,
        current_state,
        current_state_valid_moves,
        next_state,
        next_state_valid_moves,
    ):
        if len(self.memory) >= self.MEMORY_SIZE:
            min_value = self.MAX_PRIORITY
            min_index = 0

            for i, n in enumerate(self.memory):
                if n[0] < min_value:
                    min_value = n[0]
                    min_index = i

            del self.memory[min_index]

        self.memory.append(
            (
                priority,
                action,
                reward,
                done,
                current_state,
                current_state_valid_moves,
                next_state,
                next_state_valid_moves,
            )
        )

    def train_model(self) -> float:
        """
        Trains the network using experience replay.
        Returns:
            float: The loss value.
        """
        if len(self.memory) < self.batch_size:
            return 0

        # get priorities from the first element in the memory samples tuple
        priorities = [x[0] for x in self.memory]

        # the higher the priority, the more probable the sample will be included in the batch training
        priorities_total = np.sum(priorities)
        weights = priorities / priorities_total

        # extract samples for the batch training
        minibatch_indexes = np.random.choice(
            range(len(self.memory)), size=self.batch_size, replace=False, p=weights
        )
        minibatch = [self.memory[x] for x in minibatch_indexes]

        action_list = []
        reward_list = []
        done_list = []
        current_state_list = []
        current_moves_list = []
        next_state_list = []
        next_moves_list = []
        for (
            priority,
            action,
            reward,
            done,
            current_state,
            current_moves,
            next_state,
            next_moves,
        ) in minibatch:

            # bit state is the 16 bitboards of the state before the move
            current_state_list.append(current_state)

            current_moves = torch.from_numpy(current_moves.flatten())
            current_moves_list.append(current_moves.unsqueeze(0))

            # action is the index of the chosen move (out of 6561)
            action_list.append([action])

            # reward is the reward obtained by making the chosen move
            reward_list.append(reward)

            # done indicates if the game ended after making the chosen move
            done_list.append(done)

            if not done:
                # next_bit_state is the 16 bitboards of the state after the move
                next_state_list.append(next_state)

                # next_state_valid_moves is a tensor containing the indexes of valid moves (out of 6561)
                next_moves = torch.from_numpy(next_moves.flatten())
                next_moves_list.append(next_moves.unsqueeze(0))

        # state_valid_moves and next_state_valid_moves are already tensors, we just need to concat them
        state_valid_move_tensor = torch.cat(current_moves_list, 0)
        next_state_valid_move_tensor = torch.cat(next_moves_list, 0)

        # convert all lists to tensors
        state_tensor = torch.from_numpy(np.array(current_state_list)).float()
        action_list_tensor = torch.from_numpy(np.array(action_list, dtype=np.int64))
        reward_list_tensor = torch.from_numpy(np.array(reward_list)).float()
        next_state_tensor = torch.from_numpy(np.array(next_state_list)).float()

        # create a tensor with
        bool_array = np.array([not x for x in done_list])
        not_done_mask = torch.tensor(bool_array, dtype=torch.bool)

        # compute the expected rewards for each valid move
        policy_action_values = self.q_network(state_tensor, state_valid_move_tensor)

        # get only the expected reward for the chosen move (to calculate loss against the actual reward)
        policy_action_values = policy_action_values.gather(1, action_list_tensor)

        # target values are what we want the network to predict (our actual values in the loss function)
        # target values = reward + max_reward_in_next_state * gamma
        # gamma is the discount factor and tells the agent whether to prefer long term rewards or
        # immediate rewards. 0 = greedy, 1 = long term
        max_reward_in_next_state = torch.zeros(self.batch_size, dtype=torch.double)

        with torch.no_grad():

            # if the state is final (done = True, not_done_mask = False) the max_reward_in_next_state stays 0
            max_reward_in_next_state[not_done_mask] = self.target_network(
                next_state_tensor, next_state_valid_move_tensor
            ).max(1)[0]

        target_action_values = (
                                       max_reward_in_next_state * self.gamma
                               ) + reward_list_tensor
        target_action_values = target_action_values.unsqueeze(1)

        # loss is computed between expected values (predicted) and target values (actual)
        loss = self.loss_function(policy_action_values, target_action_values)

        # Update priorities of samples in memory based on size of error (higher error = higher priority)
        for i in range(self.batch_size):
            predicted_value = policy_action_values[i]
            target_value = target_action_values[i]

            # priority = mean squared error
            priority = (
                F.mse_loss(predicted_value, target_value, reduction="mean")
                .detach()
                .numpy()
            )

            # change priority of sample in memory
            sample = list(self.memory[minibatch_indexes[i]])
            sample[0] = priority
            self.memory[minibatch_indexes[i]] = tuple(sample)

        # clear gradients of all parameters from the previous training step
        self.optimizer.zero_grad()

        # calculate the new gradients of the loss with respect to all the model parameters by traversing the network backwards
        loss.backward()

        # adjust model parameters (weights, biases) according to computed gradients and learning rate
        self.optimizer.step()

        self.target_network = self.q_network

        # return loss so that we can plot loss by training step
        return float(loss)