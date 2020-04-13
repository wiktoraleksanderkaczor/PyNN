"""
    Module: derivative.py
    This module contains a definition of various activation function derivatives.

    The functions here are marked with the "@njit" decorator, this means that they're compiled to machine code at runtime.
"""

from numba import njit


@njit
def sigmoid_derivative(x):
    """
        Return the derivative of a sigmoid function

        Args:
            x (numpy.dtype): The activation value for the derivative function.

        Returns:
            The derivative.
    """
    return x * (1.0 - x)


@njit
def tanh_derivative(x):
    """
        Return the derivative of a sigmoid function

        Args:
            x (numpy.dtype): The activation value for the derivative function.

        Returns:
            The derivative.
    """
    return 1.0 - (x ** 2)


@njit
def relu_derivative(x):
    """
        Return the derivative of a rectifier linear unit function

        Args:
            x (numpy.dtype): The activation value for the derivative function.

        Returns:
            The derivative.
    """
    if x == 1:
        return 1
    else:
        return 0


@njit
def none_derivative(x):
    """
        Return the derivative of no activation function

        Args:
            x (numpy.dtype): The activation value for the derivative function.

        Returns:
            The derivative.
    """
    return 1
