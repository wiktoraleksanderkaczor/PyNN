"""
    Module: activation.py
    This module contains a definition of various activation functions.
    https://github.com/google/jax
"""

import jax.numpy as jnp
from jax import grad, jit


def sigmoid(x):
    """
        Return the activation after a sigmoid function

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return 1 / (1 + jnp.absolute(jnp.exp(-x)))
sigmoid.derivative = jit(grad(sigmoid))


def tanh(x):
    """
        Return the activation after a hyperbolic tangent function.

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return jnp.tanh(x)
tanh.derivative = jit(grad(tanh))


def RELU(x):
    """
        Return the activation after a rectifier linear unit function.

        Args:
            x (numpy.dtype): The input sum for the activation function.

        Returns:
            The activation value.
    """
    return x * (x > 0)
RELU.derivative = jit(grad(RELU))


def Input(x):
    """
        Return the value passed without any activation function applied.

        Args:
            x (numpy.dtype): The input sum for the function.
        
        Returns:
            The activation value.
    """
    return x
Input.derivative = jit(grad(Input))