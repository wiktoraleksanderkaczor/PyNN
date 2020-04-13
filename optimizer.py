"""
    Module: optimizer.py
    This module contains a definition of various optimizing/learning as well as helper functions.
"""

from numpy import zeros
from training import get_activation_values, get_delta_bias_structure, get_delta_structure


# WORK_IN_PROGRESS
# TODO: Sum neuron influence over layer for multi-layer networks with more than one neuron per layer (excluding the input layer).
def gradient_descent(self, learning_rate, loss):
    """
        Calculating the gradients for the weights and biases of each neuron and updating them.

        The weight gradient for hidden layers is the gradient for the connection to the neuron in the previous layer (layer - 1) times the derivative of 
        the activation function for the current neuron times the activation value of the neuron in the previous layer.

        The bias gradient is the same except the derivative of the "activation function" so to speak is just 1. 
    """
    # Get all uncleared activation values.
    activation_values = get_activation_values(self=self)

    # Get a structure to store the weight derivatives/gradients that's currently filled with the actual weights.
    delta = get_delta_structure(self=self)
    # Get a structure to store the bias derivatives/gradients that's currently filled with zeros.
    delta_bias = get_delta_bias_structure(self=self)

    for layer in reversed(range(self.num_layers_index)):
        # Defining some indexing conveniences.
        activations = self.model[self.neuron_activation_layer][layer]
        derivative = self.model[self.neuron_derivative_layer][layer]
        previous_layer = layer + 1
        next_layer = layer - 1

        # Skipping layer 0 because it doesn't have any weights to optimize.
        if layer == 0:
            continue

        # For each neuron in the layer.
        for neuron in range(self.model[self.neuron_num_layer][layer]):
            # Calculating gradients for the output layer.
            if layer == self.num_layers:
                # For each connection in the next layer calculate bias for which the derivative is 1.
                for connection in range(self.model[self.neuron_num_layer][next_layer]):
                    delta_bias[layer][neuron] = loss * 1 * \
                        activation_values[next_layer][connection]

                # For each connection in the next layer.
                for connection in range(self.model[self.neuron_num_layer][next_layer]):
                    # Calculating the gradient for the weight.
                    delta[layer][neuron][connection] = loss * \
                        derivative(activation_values[layer][neuron]) * \
                        activation_values[next_layer][connection]

            # Calculating gradients for the hidden layers.
            else:
                # For each connection in the previous layer calculate bias for which the derivative is 1.
                for connection in range(self.model[self.neuron_num_layer][previous_layer]):
                    delta_bias[layer][neuron] = delta[previous_layer][connection][neuron] * 1 * \
                        activation_values[next_layer][connection]

                # For each connection in the previous layer.
                for connection in range(self.model[self.neuron_num_layer][previous_layer]):
                    # Calculating common elements to avoid code duplication.
                    delta[layer][neuron][connection] = delta[previous_layer][connection][neuron] * \
                        derivative(activation_values[layer][neuron]) * \
                        activation_values[next_layer][connection]

    # Returning the calculated gradients.
    return (delta, delta_bias)


# WORK_IN_PROGRESS
def stochastic_gradient_descent(self, learning_rate, loss, mini_batch):
    pass
