"""
    Module: neuron.py
    This module contains a declaration and definition of the Neuron class.
"""

from numpy import dot
from numpy import array
import numpy as np
from weights import random_init, xavier_init


class Neuron:
    def __init__(self, precision):
        """
            Constructor function for a Neuron.

            Args:
                precision (numpy.dtype): The numpy dtype for the network precision.
        """
        self.activation = precision(0)
        self.sum = precision(0)
        self.bias = precision(0)
        self.weights = None

    def set_sum(self, input_val):
        """
            Setting the neuron sum to a dot product of the input tensor and weights 
            plus the bias.

            Args:
                input_val (numpy.array): The input tensor for the neuron.
        """
        self.sum = np.dot(self.weights, input_val) + self.bias
