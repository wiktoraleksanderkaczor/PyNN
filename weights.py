# Module: weights.py
# Definition of some useful weight initialization functions.

import numpy as np
from numba import njit

@njit
def random_init(num_prev_neurons, precision):
    """
        Initializes the weights using a random number generated from a seed.
    
        Args:
            num_prev_neurons (int): The number of neurons in the previous layer.
            precision (numpy.dtype): The numpy dtype for the network precision.

        Returns:
            weights (numpy.array): A 1-dimensional array of the randomly initialized weights for a neuron.
    """
    # Setting seed based on num_prev_neurons.
    np.random.seed(num_prev_neurons)
    # Storing weights for each connection to each neuron in the next layer.
    weights = np.random.rand(num_prev_neurons)
    return weights.astype(precision)

@njit(parallel=True)
def xavier_init(num_prev_neurons, precision):
    """
        Initializes the weights using the xavier weight initialization algorithm.
    
        Args:
            num_prev_neurons (int): The number of neurons in the previous layer.
            precision (numpy.dtype): The numpy dtype for the network precision.

        Returns:
            weights (numpy.array): A 1-dimensional array of the xavier initialized weights for a neuron.
    """
    # Setting seed based on num_next_neurons.
    np.random.seed(num_prev_neurons)
    # Storing weights for each connection to each neuron in the next layer.
    weights = np.random.rand(num_prev_neurons)

    # Shifting weights by xavier algorithm, to manage exploding and
    # vanishing gradients.
    # Note: 2 / num_prev_neurons is supposed to work better for RELU
    shift = np.sqrt(1 / num_prev_neurons)
    weights = np.multiply(weights, shift)
    return weights.astype(precision)
