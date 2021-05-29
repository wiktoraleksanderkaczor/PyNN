"""
    Module: optimizer.py
    This module contains a definition of various optimizing/learning as well as helper functions.
"""

import numpy as np
from itertools import repeat

# WORK_IN_PROGRESS
def gradient_descent(model, loss):
    """
        Calculating the gradients for the weights and biases of each neuron. 
    """
    neuron_error = [[] for _ in range(len(model["layers"]))]
    weight_gradients = [[] for _ in range(len(model["layers"]))]
    bias_gradients = [[] for _ in range(len(model["layers"]))]

    # So the same function for output and hidden can be used.
    neuron_error.append([loss])

    
    for layer_num, layer in reversed(list(enumerate(model["layers"]))):
        prev_layer_neurons = model["layers"][layer_num - 1]["neurons"]

        # Hidden or output layers, the error is substituted by the neuron_error:
        if layer != model["layers"][0]:
            summed_influence = sum(neuron_error[layer_num+1])
            for neuron in layer["neurons"]:
                summed_influence_derivative = layer["activation"].derivative(neuron.sum) * summed_influence 

                neuron_weight_error = [
                    neuron_back.activation * summed_influence_derivative for neuron_back in prev_layer_neurons
                ]

                # Only one per neuron
                bias_gradients[layer_num].append(
                    # The derivative for bias is 1 and that's the same as multiplying by nothing.
                    neuron.activation * summed_influence
                )

                weight_gradients[layer_num].append(neuron_weight_error)
                neuron_error[layer_num].append(sum(neuron_weight_error))

        # Input layer (only one weight)
        else:
            summed_influence = sum(neuron_error[layer_num+1])
            for neuron in layer["neurons"]:
                # The neuron sum is essentially the input or the activation of the previous layer 
                neuron_weight_error = [
                    neuron.sum * layer["activation"].derivative(neuron.sum) * summed_influence
                ] 

                # Only one per neuron
                bias_gradients[layer_num].append(
                    # The derivative for bias is 1 and that's the same as multiplying by nothing.
                    neuron.activation * summed_influence
                )

                weight_gradients[layer_num].append(neuron_weight_error)
                neuron_error[layer_num].append(sum(neuron_weight_error))

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
            neuron_weight_gradients = np.array(weight_gradients[layer_num][neuron_num])
            neuron_weight_gradients = np.multiply(neuron_weight_gradients, learning_rate)
            neuron.weights = np.subtract(neuron.weights, neuron_weight_gradients)
            neuron.bias = neuron.bias - bias_gradients[layer_num][neuron_num] * learning_rate
