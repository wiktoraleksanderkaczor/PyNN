"""
    Module: cost.py
    This module contains a definition of various loss/cost functions.
"""

import numpy as np


def mean_squared_loss(predicted, actual):
    """
        Calculate the loss using the mean squared cost function for a single example.

        Returns a scalar value because backpropagation doesn't work for optimizing vectors.

        Args:
            predicted (numpy.array): The values predicted for a given example.
            actual (numpy.array): The correct values for a given example.

        Returns:
            float: The mean squared error for a given example.
    """
    mse = (np.square(actual - predicted)).mean(axis=0)
    return mse

def sum_squared_loss(predicted, actual):
    """
        Calculate the loss using the sum squared cost function for a single example.

        Returns a scalar value because backpropagation doesn't work for optimizing vectors.

        Args:
            predicted (numpy.array): The values predicted for a given example.
            actual (numpy.array): The correct values for a given example.

        Returns:
            float: The sum squared error for a given example.
    """
    sse = np.sum(np.square(actual - predicted))
    return sse