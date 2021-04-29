"""
    Module: optimizer.py
    This module contains a definition of various optimizing/learning as well as helper functions.
"""

import numpy as np
from itertools import repeat

# WORK_IN_PROGRESS
def gradient_descent(model, expected, loss):
    """
        Calculating the gradients for the weights and biases of each neuron. 
    """
    neuron_error = [[] for _ in range(len(model["layers"]))]
    weight_gradients = [[] for _ in range(len(model["layers"]))]
    bias_gradients = [[] for _ in range(len(model["layers"]))]

    # Update rule for weight is weight = weight - (learning_rate * weight_derivative)
    # Update rule for bias is bias = bias - (learning_rate * bias)
    return {"weights": weight_gradients}#, "bias": bias_gradients}