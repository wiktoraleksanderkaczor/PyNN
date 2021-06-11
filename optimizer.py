"""
    Module: optimizer.py
    This module contains a definition of various optimizing/learning as well as helper functions.
"""

import numpy as np
from itertools import repeat


def gradient_descent(model, input_data, loss):
    """
        Calculating the gradients for the weights and biases of each neuron. 
    """
    neuron_error = [[] for _ in range(len(model["layers"]))]
    weight_gradients = [[] for _ in range(len(model["layers"]))]
    bias_gradients = [[] for _ in range(len(model["layers"]))]

    # So the same function for output and hidden can be used.
    neuron_error.append([loss])

    
    for layer_num, layer in reversed(list(enumerate(model["layers"]))):
        prev_layer_neuron_activations = [neuron.activation for neuron in model["layers"][layer_num - 1]["neurons"]] if layer_num != 0 else [neuron.sum for neuron in layer["neurons"]]
        next_layer_influence = np.sum(neuron_error[layer_num+1])

        # Hidden or output layers, the error is substituted by the neuron_error:
        for neuron_num, neuron in enumerate(layer["neurons"]):
            next_layer_influence_derivative = layer["activation"].derivative(neuron.sum) * next_layer_influence 

            if layer_num != 0:
                neuron_weight_error = np.multiply(prev_layer_neuron_activations, next_layer_influence_derivative)
            else:
                # Input layer (only one weight)
                neuron_weight_error = np.multiply(input_data[neuron_num], next_layer_influence_derivative)

            # Only one per neuron
            bias_gradients[layer_num].append(
                # The derivative for bias is 1 and that's the same as multiplying by nothing.
                np.multiply(neuron.activation, next_layer_influence)
            )

            weight_gradients[layer_num].append(neuron_weight_error)
            neuron_error[layer_num].append(np.sum(neuron_weight_error))

    return weight_gradients, bias_gradients


def update_weights_and_bias(model, weight_gradients, bias_gradients, learning_rate=0.01):
    """
        Updating the neuron weights and bias for each layer using the calculated gradients.

        Args:
            model (dict): A python dictionary that contains the model.
            gradients (dict): A python dictionary that contains the weight and bias gradients.

    """
    for layer_num, layer in reversed(list(enumerate(model["layers"]))):
        for neuron_num, neuron in enumerate(layer["neurons"]):
            neuron_weight_gradients = weight_gradients[layer_num][neuron_num]
            neuron_weight_gradients = np.multiply(neuron_weight_gradients, learning_rate)
            neuron.weights = np.subtract(neuron.weights, neuron_weight_gradients)
            neuron.bias = neuron.bias - bias_gradients[layer_num][neuron_num] * learning_rate
