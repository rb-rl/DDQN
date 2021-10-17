########################################################################################################################
# A neural network.
#
# Code adaption of [1] as required by [2].
#
# [1] Udacity, Deep Reinforcement Learning Nanodegree Program,
#     2. Value-Based Methods, Lesson 2 - Deep Q-Networks, 7. Workspace, solution, model.py
# [2] Udacity, Deep Reinforcement Learning Nanodegree Program,
#     2. Value-Based Methods, Project - Navigation, 7. Not sure where to start?, Step 3
#
########################################################################################################################

from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import relu

# TOPOLOGY -------------------------------------------------------------------------------------------------------------

# The number of hidden layers.
NUMBER_HIDDEN_LAYERS = 3

# The number of hidden neurons per layer.
NUMBER_HIDDEN_NEURONS_PER_LAYER = 64

# The activation function used in all but the last layer.
ACTIVATION_FUNCTION = "relu"

# DEVELOPMENT ----------------------------------------------------------------------------------------------------------

# Is the fast development mode activated which reloads imports?
#
# The advantage of the fast development mode is that one does not have to restart Python from scratch fear each
# development increment which makes the development faster.
FAST_DEVELOPMENT_MODE = True


class NeuralNetwork(Module):
    """
    A neural network based on fully connected layers.
    """

    def __init__(self, number_inputs: int, number_outputs: int):
        """
        Initialize the fully connected layers.

        Args:
            number_inputs: The number of input neurons.
            number_outputs: The number of output neurons.
        """
        super(NeuralNetwork, self).__init__()

        NUMBER_VISIBLE_LAYERS = 2

        self.__number_layers = NUMBER_VISIBLE_LAYERS + NUMBER_HIDDEN_LAYERS

        number_before = number_inputs

        for index in range(self.__number_layers):
            if index < self.__number_layers - 1:
                number_after = NUMBER_HIDDEN_NEURONS_PER_LAYER
            else:
                number_after = number_outputs

            exec("self.__linear_" + str(index) + " = Linear(number_before, number_after)")

            number_before = number_after

    def __call__(self, input: Tensor) -> Tensor:
        """
        Perform a forward propagation on the given input.

        Args:
            input: The input to be forward propagated.

        Returns:
            The output resulting from the forward propagation.
        """
        activations = input

        for index in range(self.__number_layers):
            activations = eval("self.__linear_" + str(index) + "(activations)")
            if index < self.__number_layers - 1:
                activations = eval(ACTIVATION_FUNCTION + "(activations)")

        return activations
