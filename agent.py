# ######################################################################################################################
# A reinforcement learning agent.
#
# Code adaption of [1] as required by [2].
#
# [1] Udacity, Deep Reinforcement Learning Nanodegree Program,
#     2. Value-Based Methods, Lesson 2 - Deep Q-Networks, 7. Workspace, solution, dqn_agent.py
# [2] Udacity, Deep Reinforcement Learning Nanodegree Program,
#     2. Value-Based Methods, Project - Navigation, 7. Not sure where to start?, Step 3
#
# ######################################################################################################################

import network
import memory

if network.FAST_DEVELOPMENT_MODE:
    import importlib

    importlib.reload(network)
    importlib.reload(memory)
    print("Fast development reload: network")
    print("Fast development reload: memory")

from network import NeuralNetwork
from memory import ReplayMemory

import torch

import numpy as np
import torch.nn.functional as F

from numpy import ndarray
from pathlib import Path
from random import randint, random
from torch import Tensor
from torch.optim import Adam
from typing import Tuple

# GENERAL --------------------------------------------------------------------------------------------------------------

# The used device in {"cuda", "cpu"}.
#
# Note that if you have a GPU which requires at least CUDA 9.0, the usage of the CPU is recommended, because otherwise
# the execution might be unexpectedly slow.
DEVICE = "cpu"

# LEARNING -------------------------------------------------------------------------------------------------------------

# The learning rate.
LEARNING_RATE = 0.001

# The interval the epsilon value of epsilon greediness may be in.
EPSILON_INTERVAL = [0.01, 1]

# The amount of epsilon decay.
EPSILON_DECAY = 0.9999

# The discount factor.
GAMMA = 0.99

# The batch size.
BATCH_SIZE = 64

# The loss function.
LOSS = "mse_loss"

# The soft update rate of target deep Q-network.
TAU = 0.001

# The number of frames per update of the target deep Q-network
FRAMES_PER_UPDATE = 4


class Agent:
    """
    An agent based on double deep Q-learning, inspired by [1].
    """

    def __init__(self, number_sensors: int, number_motors: int):
        """
        Initialize the agent.

        Args:
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        self.__number_motors = number_motors

        device_name = DEVICE if torch.cuda.is_available() else "cpu"
        print("Used device:", device_name)

        print()

        self.__device = torch.device(device_name)

        self.__q_network = NeuralNetwork(number_sensors, number_motors).to(self.__device)
        self.__q_network_target = NeuralNetwork(number_sensors, number_motors).to(self.__device)

        print("Q", self.__q_network)
        print()
        print("Target Q", self.__q_network_target)
        print()

        self.__optimizer = Adam(self.__q_network.parameters(), lr=LEARNING_RATE)

        self.__replay_memory = ReplayMemory(self.__device)

        self.__epsilon = EPSILON_INTERVAL[1]

        self.__step = 0

    def __call__(self, state: ndarray) -> int:
        """
        Let the agent act on the given state based on epsilon greedy policy.

        Args:
            state: The current state of the agent.

        Returns:
            The selected action.
        """
        if random() > self.__epsilon:
            input = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)

            self.__q_network.eval()

            with torch.no_grad():
                output = self.__q_network(input)

            argmax_q = torch.argmax(output)

            action = int(argmax_q.cpu().data.numpy())
        else:
            action = randint(0, self.__number_motors - 1)

        return action

    def learn(self, state: ndarray, action: int, reward: float, next_state: ndarray, done: bool) -> Tuple[float, float]:
        """
        Perform a learning step.

        Args:
            state: The current state.
            action: The action taken in the current state.
            reward: The reward obtained by going from the current to the next state.
            next_state: The next state.
            done: Is the episode done?

        Returns:
            The current epsilon value and the loss.
        """
        self.__replay_memory.add(state, action, reward, next_state, done)

        batch_size = min(BATCH_SIZE, len(self.__replay_memory))
        experiences = self.__replay_memory.extract_random_experiences(batch_size)

        loss = self.__update_q_network(experiences)

        self.__step += 1
        if self.__step % FRAMES_PER_UPDATE == 0:
            self.__soft_update()

        self.__epsilon_decay()

        return self.__epsilon, loss

    def save(self, path: str):
        """
        Save the neural network of the agent.

        Args:
            path: The path to the file where the neural network should be stored, excluded the file ending.
        """
        full_path = Path(path).with_suffix(".pt")

        torch.save(self.__q_network.state_dict(), full_path)

        print("Agent saved in", full_path)

    def load(self, path: str):
        """
        Load the neural network of the agent.

        Note that the loading is asymmetric to the saving, for simplicity, because we do not save the epsilon value.
        Hence, it is not possible to continue training from a loaded model.

        Args:
            path: The path to the file where the neural network should be loaded from, excluded the file ending.
        """
        full_path = Path(path).with_suffix(".pt")

        self.__q_network.load_state_dict(torch.load(full_path))
        self.__epsilon = 0

        print("Agent loaded from", full_path)

    def __epsilon_decay(self):
        """
        Perform epsilon decay.
        """
        self.__epsilon = max(EPSILON_INTERVAL[0], EPSILON_DECAY * self.__epsilon)

    def __update_q_network(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> float:
        """
        Update the deep Q-network.

        Args:
            experiences: The experiences used for the update.

        Returns:
            The loss.
        """
        states, actions, rewards, next_states, dones = experiences

        adjusted_actions = actions.unsqueeze(1)
        adjusted_rewards = rewards.unsqueeze(1)
        adjusted_dones = dones.unsqueeze(1).float()

        # max_a' Q'(s',a')
        self.__q_network_target.eval()
        target_q_value_per_action = self.__q_network_target(next_states).detach()
        target_max_q_value = target_q_value_per_action.max(1)[0].unsqueeze(1)

        # r'+gamma*max_a' Q'(s',a')
        target = adjusted_rewards + (GAMMA * target_max_q_value * (1 - adjusted_dones))

        # Q(s,a)
        self.__q_network.train()
        q_value = self.__q_network(states).gather(1, adjusted_actions)

        loss = eval("F." + LOSS)(q_value, target)

        self.__optimizer.zero_grad()
        loss.backward()

        self.__optimizer.step()

        return loss

    def __soft_update(self):
        """
        Perform a soft update of the target deep Q-network.
        """
        for parameters, parameters_target in zip(self.__q_network.parameters(), self.__q_network_target.parameters()):
            parameters_target.data.copy_((1 - TAU) * parameters_target.data + TAU * parameters.data)
