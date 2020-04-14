"""
    Module: training.py
    This module contains a definition of various training helper functions.
"""

from numpy import zeros, array, ones


def get_activation_values(self):
    """
        Getting the uncleared activation values for each neuron into a 2-dimensional array.
    """
    activation_values = [[] for i in range(self.num_layers_index)]
    for layer in range(self.num_layers_index):
        activation_values[layer] = zeros(
            self.model[self.neuron_num_layer][layer], dtype=self.precision)
        for neuron in range(self.model[self.neuron_num_layer][layer]):
            activation_values[layer][neuron] = self.list_neurons[layer][neuron].activation_value

    return activation_values


def get_delta_structure(self, fill_zeros=False):
    """
        Creating a three-dimensional array to store the gradients for each weight and 
        prefilling it with the weights for each neuron.

        The array is indexed by; layer, neuron, and finally, connection for each neuron in 
        the previous layer.

        Args:
            zeros (boolean): A boolean representing whether to change all the to zeros. 
    """
    delta = [[] for i in range(self.num_layers_index)]
    for layer in range(self.num_layers_index):

        # Skipping layer 0 because it doesn't have any weights to optimize.
        if layer == 0:
            continue

        if fill_zeros:
            delta[layer] = self.rip_layer_weights(layer=layer)
            for neuron in range(len(delta[layer])):
                for weight in range(len(delta[layer][neuron])):
                    delta[layer][neuron][weight] = 0

        else:
            delta[layer] = self.rip_layer_weights(layer=layer)

    return delta


def get_delta_bias_structure(self, fill_zeros=False):
    """
        Return a structure to hold the bias gradients for each neuron. Prefilled with zeros.
    """
    delta = [[] for i in range(self.num_layers_index)]
    for layer in range(self.num_layers_index):
        if layer == 0:
            continue

        if fill_zeros:
            delta[layer] = zeros(
                self.model[self.neuron_num_layer][layer], dtype=self.precision)
        else:
            delta[layer] = self.rip_layer_biases(layer=layer)

    return delta


def update_weights_and_bias(self, delta, delta_bias):
    """
        Updating the neuron weights and bias for each layer using the calculated gradients.

        Args:
            delta (array): A python array that contains the weight gradients for each layer, neuron 
                and connection backwards.
            delta_bias (array): A python array that contains the bias gradients for each layer, 
                neuron and connection backwards.

    """
    for layer in reversed(range(self.num_layers_index)):
        # Skipping layer 0 because it doesn't have any weights to optimize.
        if layer == 0:
            continue
        for neuron in range(len(self.list_neurons[layer])):
            # Updating bias.
            self.list_neurons[layer][neuron].bias -= \
                delta_bias[layer][neuron]

            # Updating weights
            for weight in range(len(self.list_neurons[layer][neuron].weights)):
                self.list_neurons[layer][neuron].weights[weight] -= \
                    delta[layer][neuron][weight]


def apply_learning_rate(delta, delta_bias, learning_rate):
    """
        Updating the weight and bias gradients with the learning rate.

        Args:
            delta (array): The weight gradients from the current iteration.
            delta_bias (array): The bias gradients from the current iteration.   
            learning_rate (float): The learning rate hyper-parameter for calculating the new gradient 
                update.     
    """
    for layer in range(len(delta)):
        for neuron in range(len(delta[layer])):
            delta_bias[layer][neuron] *= learning_rate

            for weight in range(len(delta[layer][neuron])):
                delta[layer][neuron][weight] *= learning_rate

    return delta, delta_bias


def calculate_momentum(momentum, momentum_bias, delta, delta_bias, coefficient):
    """
        Updating the weight gradients with the momentum from the previous iteration.

        Args:
            momentum (array): The momentum-adjusted weight gradients from the previous iteration.
                This is preinitialised to zeros for the first iteration.
            momentum_bias (array): The momentum-adjusted bias gradients from the previous iteration.
                This is preinitialised to zeros for the first iteration.
            delta (array): The weight gradients from the current iteration.
            delta_bias (array): The bias gradients from the current iteration.
            coefficient (float): The percentage value of which to apply momentum for from the previous
                iteration.
    """
    for layer in range(len(momentum)):
        for neuron in range(len(momentum[layer])):
            momentum_bias[layer][neuron] = (
                coefficient * momentum_bias[layer][neuron]) + delta_bias[layer][neuron]

            for weight in range(len(momentum[layer][neuron])):
                momentum[layer][neuron][weight] = (
                    coefficient * momentum[layer][neuron][weight]) + delta[layer][neuron][weight]

    return momentum, momentum_bias
