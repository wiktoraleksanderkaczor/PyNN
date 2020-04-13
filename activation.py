"""
    Module: activation.py
    This module contains a definition of various activation functions.

    The functions here are marked with the "@njit" decorator, this means that they're compiled to machine code at runtime.
"""
import cmath

from numba import njit
from numpy import absolute


@njit
def sigmoid(x):
    """
        Return the activation after a sigmoid function

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return 1 / (1 + absolute(cmath.exp(-x)))


@njit
def tanh(x):
    """
        Return the activation after a hyperbolic tangent function.

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return cmath.tanh(x).real


@njit
def RELU(x):
    """
        Return the activation after a rectifier linear unit function.

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return x * (x > 0)


@njit
def none(x):
    """
        Return the value passed without any activation function applied.

        Args:
            x (numpy.dtype): The input sum for the function.
        
        Returns:
            The activation value.
    """
    return x
