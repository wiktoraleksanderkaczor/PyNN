"""
    Module: neuron.py
    This module contains a declaration and definition of the Neuron class.
"""

from numpy import dot as dot_product
from numpy import array
from weights import random_init, xavier_init


class Neuron:
    def __init__(self, num_prev, weight_function, precision):
        """
            Constructor function for a Neuron.

            Args:
                activation_for_neuron (str): The activation function for the neuron.
                num_prev (int): The number of neurons in the previous layer.
                precision (numpy.dtype): The numpy dtype for the network precision.
                weight_function (str): The weight initialization function for the neuron.
        """
        self.activation_value = precision(0)
        self.sum = precision(0)
        self.bias = precision(0)

        # Storing and initializing weights for each neuron not in the input layer.
        if num_prev:
            if weight_function == "xavier":
                self.weights = xavier_init(num_prev, precision)
            elif weight_function == "random":
                self.weights = random_init(num_prev)
        else:
            self.weights = array([1], dtype=precision)

    def set_sum(self, input_val):
        """
            Setting the neuron sum to a dot product of the input tensor and weights 
            plus the bias.

            Args:
                input_val (numpy.array): The input tensor for the neuron.
        """
        self.sum = dot_product(input_val, self.weights) + self.bias

    def set_sum_raw(self, input_val):
        """
            Setting the neuron sum to the input without preprocessing.

            Args:
                input_val (numpy.array): The input sum for the neuron.
        """
        self.sum = input_val
