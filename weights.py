# Module: weights.py
# Definition of some useful weight initialization functions.

import numpy as np
np.random.seed(123)

def random_init(num_prev_neurons, precision):
    """
        Initializes the weights using a random number generated from a seed.
    
        Args:
            num_prev_neurons (int): The number of neurons in the previous layer.
            precision (numpy.dtype): The numpy dtype for the network precision.

        Returns:
            weights (numpy.array): A 1-dimensional array of the randomly initialized weights for a neuron.
    """
    # Storing weights for each connection to each neuron in the next layer.
    weights = np.random.rand(num_prev_neurons)
    return weights.astype(precision)


def xavier_init(num_prev_neurons, precision):
    """
        Initializes the weights using the xavier weight initialization algorithm.
    
        Args:
            num_prev_neurons (int): The number of neurons in the previous layer.
            precision (numpy.dtype): The numpy dtype for the network precision.

        Returns:
            weights (numpy.array): A 1-dimensional array of the xavier initialized weights for a neuron.
    """
    # Setting seed based on number of previous neurons.
    #np.random.seed(num_prev_neurons)
    
    lower = -(1.0 / np.sqrt(num_prev_neurons))
    upper = (1.0 / np.sqrt(num_prev_neurons))

    # Storing weights for each connection to each neuron in the next layer.
    weights = np.random.rand(num_prev_neurons)

    return weights.astype(precision)
