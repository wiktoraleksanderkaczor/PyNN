"""
    Module: optimizer.py
    This module contains a definition of various optimizing/learning as well as helper functions.
"""

import numpy as np
from itertools import repeat

# WORK_IN_PROGRESS
# TODO: Sum neuron influence over layer for multi-layer networks with more than one neuron per layer (excluding the input layer).
def gradient_descent(model, expected, loss, learning_rate=0.01):
    """
        Calculating the gradients for the weights and biases of each neuron and updating them.

        The weight gradient for hidden layers is the gradient for the connection to the neuron in the previous layer (layer - 1) times the derivative of 
        the activation function for the current neuron times the activation value of the neuron in the previous layer.

        The bias gradient is the same except the derivative of the "activation function" so to speak is just 1. 
    """
    weight_gradients = [[] for _ in range(len(model["layers"]))]
    bias_gradients = [[] for _ in range(len(model["layers"]))]

    output_layer = model["layers"][-1]["neurons"]
    node_loss = [np.square(neuron.activation - actual) for neuron, actual in zip(output_layer, expected)]

    num_layers = len(model["layers"]) - 1
    for layer in reversed(model["layers"]):
        # Get current layer number
        layer_num = model["layers"].index(layer)
        derivative = layer["activation"].derivative

        # Get previous layer and number.
        prev_layer_num = layer_num - 1
        if prev_layer_num == -1:
            # Finished
            return
        else:
            prev_layer = model["layers"][prev_layer_num]

        if not layer_num == num_layers:
            next_layer_num = layer_num + 1
            next_layer = model["layers"][next_layer_num]

        # If output layer
        if layer_num == num_layers:
            for neuron in layer["neurons"]:
                which_neuron = layer["neurons"].index(neuron)
                neuron_derivative = derivative(neuron.sum)
                gradients = []
                for prev_neuron in prev_layer["neurons"]:
                    gradients.append(
                        # The loss for this neuron times the derivative of this neuron times the activation of the neuron which the weight is connecting.
                        node_loss[which_neuron] * neuron_derivative * prev_neuron.activation
                    )
                weight_gradients[layer_num].append(
                    np.asarray(gradients)
                )
                """
                # Bias starts at 0, will always be 0.
                bias_gradients[layer_num].append(
                    [neuron.bias * node_loss[which_neuron]]
                )
                """
        else:
            for neuron in layer["neurons"]:
                which_neuron = layer["neurons"].index(neuron)
                neuron_derivative = derivative(neuron.sum)
                gradients = []
                for prev_neuron in prev_layer["neurons"]:
                    gradients.append(
                        # The loss for this neuron times the derivative of this neuron times the activation of the neuron which the weight is connecting.
                        neuron_derivative * prev_neuron.activation
                    )
                weight_gradients[layer_num].append(
                    np.asarray(gradients)
                )

    # Update rule for weight is weight = weight - (learning_rate * weight_derivative)
    # Update rule for bias is bias = bias - (learning_rate * bias)
    return weight_gradients, bias_gradients
