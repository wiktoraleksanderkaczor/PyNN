"""
    Module: model.py
    This module contains a declaration and definition of useful functions for creating a 
    neural network model as well as training and running it. 
"""

import numpy as np
from neuron import Neuron
from optimizer import gradient_descent, stochastic_gradient_descent
from cost import mean_squared_loss
from training import update_weights_and_bias, get_delta_structure, get_delta_bias_structure, momentum_scaling
from prettytable import PrettyTable


class Model:
    def __init__(self, num_layers, precision):
        """
            Constructor function for a Model.

            Args:
                num_layers (int): The number of layers in the network.
                precision (numpy.dtype): The numpy dtype for the network precision.
        """
        self.num_layers = -1
        self.num_layers_index = 0
        self.input_layer = 0
        self.neuron_num_layer = 0
        self.neuron_activation_layer = 1
        self.neuron_derivative_layer = 2
        self.weight_function_layer = 3
        self.precision = precision
        self.list_neurons = [[] for i in range(num_layers)]
        self.model = np.array(
            [np.zeros(num_layers, dtype=np.int32), [], [], []])

    def add_layer(self, num_neurons, neuron_activation, neuron_derivative, weight_function):
        """
            Adding a layer to the model.

            Args:
                num_neurons (int): The number of neurons in the layer to be added.
                neuron_activation (function): The activation function to use for the layer.
                neuron_derivative (function): The derivative function for the activation for the layer.
                weight_function (str): The weight initialization function to use for the layer. 
        """
        self.num_layers += 1
        self.num_layers_index += 1
        self.model[self.neuron_num_layer][self.num_layers] = num_neurons
        self.model[self.neuron_activation_layer] = \
            np.append(
                self.model[self.neuron_activation_layer], neuron_activation)
        self.model[self.neuron_derivative_layer] = \
            np.append(
                self.model[self.neuron_derivative_layer], neuron_derivative)
        self.model[self.weight_function_layer] = \
            np.append(
                self.model[self.weight_function_layer], weight_function)

    def compile(self):
        """
            A compilation step to use the model defined for creating an actual network.
        """
        for layer in range(self.num_layers_index):
            # Create the layers from the model template.
            number = self.model[self.neuron_num_layer][layer]
            weight_function = self.model[self.weight_function_layer][layer]

            if layer != 0:
                num_prev = self.model[self.neuron_num_layer][layer - 1]
            else:
                num_prev = False

            for neuron in range(number):
                self.list_neurons[layer].append(
                    Neuron(num_prev, weight_function, self.precision))

    def run_once(self, inputs):
        """
            Feeding a given set of inputs to the network and returning the predicted values for the output layer.

            Args:
                inputs (numpy.array): A 1-dimensional array of the inputs to feed to the input layer.

            Returns:
                feed_forward (numpy.array): A 1-dimensional array containing the activation values for the output layer neurons. 
        """
        # Create numpy array to store the predictions of each neuron to feed it forward.
        feed_forward = np.array([], dtype=self.precision)

        # For each layer in the model.
        for layer in range(self.num_layers_index):
            # Create array to store activations for next layer
            tensor_length = self.model[self.neuron_num_layer][layer]

            feed_forward_tensor = np.zeros(tensor_length, dtype=self.precision)
            neuron_num = self.model[self.neuron_num_layer][layer]

            # If the neuron isn't the input layer, set the inputs to the outputs of the
            # previous layer. Otherwise, set them using the "inputs" array.
            if layer != 0:
                for neuron in range(neuron_num):
                    self.list_neurons[layer][neuron].set_sum(feed_forward)
            else:
                for neuron in range(neuron_num):
                    self.list_neurons[layer][neuron].set_sum_raw(
                        inputs[neuron])

            # For each neuron in layer.
            for neuron in range(neuron_num):
                # Triggers the activation for the neuron and sets the neuron activation value to the output.
                feed_forward_tensor[neuron] = self.model[self.neuron_activation_layer][layer](
                    self.list_neurons[layer][neuron].sum)
                self.list_neurons[layer][neuron].activation_value = feed_forward_tensor[neuron]

            feed_forward = feed_forward_tensor

        # Return activation for last layer.
        return feed_forward

    def rip_layer_weights(self, layer):
        """
            Return a 2-dimensional array of weights for a given layer. Ensure this isn't called 
            for layer 0 since it doesn't have any weights.

            Args:
                layer (int): A integer value representing the layer for which the weights will be returned.

            Returns:
                weights (numpy.array): A 2-dimensional array containing the weights of the specified layer. 
                The structure for this array wil be (layer_length x previous_layer_length).
        """

        # Get the current layer length.
        layer_length = self.model[self.neuron_num_layer][layer]

        # Get the previous layer length.
        prev_layer_length = self.model[self.neuron_num_layer][layer - 1]

        # Prepare structure to hold arrays of weights for each neuron in the layer and
        # ensure they are preinitialized because I cannot set array elements with a sequence.
        weights = np.array([np.zeros(prev_layer_length)
                            for i in range(layer_length)], dtype=self.precision)

        # Deep-copy array because otherwise it stores the gradients in the
        # weights and the backpropagation calculation is messed up.
        for neuron in range(layer_length):
            weights[neuron] = np.copy(self.list_neurons[layer][neuron].weights)

        return weights

    def train(self, function, epochs, inputs, actual, max_iter, mini_batch=10, learning_rate=0.01, min_precision=0.001,
              learning_rate_function="none", coefficient=0.9):
        """
            Runs a optimizing algorithm on the neural network using a specified algorithm.

            Args:
                function (str): A string representing the optimizing function to use.
                epochs (int): The number of times that the given example is shown to the network.
                input (numpy.array): A 1-dimensional array of the inputs to feed to the input layer.
                actual (numpy.array): A 1-dimensional array of the expected values for the output layer.
                learning_rate (float): The learning rate hyperparameter for determining how quickly a network can learn.
                max_iter (int): The maximum number of iterations ran on a given example.
                mini_batch (int): The mini batch size for stochastic gradient descent. Warning; Unimplemented.
                min_precision (float): The minimum change in the error of the network for iteration until it exits.
        """

        # Declare a new PrettyTable and define its column names.
        table = PrettyTable()
        table.field_names = ["EPOCH", "EXAMPLE", "ITERATION", "LOSS", "PREDICTED", "AFTER_UPDATE", "CHANGE", "INPUTS"]

        # Repeating training the network on the examples for N epochs.
        for epoch in range(epochs):

            # For each example do...
            for example in range(len(actual)):

                # Create structures for holding the gradient history.
                momentum = get_delta_structure(self=self, zeros=True)
                momentum_bias = get_delta_bias_structure(self=self)

                # Running multiple iterations for gradient descent.
                for iteration in range(max_iter):
                    # I'm only running it once for one example to preserve the set sums.
                    predicted = self.run_once(inputs[example])

                    # Loss for optimizers has to be a scalar so I'm choosing a sum.
                    scalar_loss = mean_squared_loss(predicted, actual[example])

                    # Set the sign for the scalar loss according to the predicted value.
                    if actual[example] > predicted:
                        scalar_loss = -scalar_loss
                    elif actual[example] < predicted:
                        scalar_loss = scalar_loss

                    if function == "GD":
                        # Run gradient descent on single example.
                        delta, delta_bias = gradient_descent(
                            self, learning_rate=learning_rate, loss=scalar_loss)
                    elif function == "SGD":
                        stochastic_gradient_descent(
                            self, learning_rate=learning_rate, loss=scalar_loss, mini_batch=mini_batch)

                    # Adapt the learning rate for the next iteration.
                    if learning_rate_function == "momentum":
                        # Implement learning rate scaling.
                        for layer in range(len(momentum)):
                            for neuron in range(len(momentum[layer])):

                                momentum_bias[layer][neuron] = (coefficient * momentum_bias[layer][neuron]) - \
                                    (learning_rate * delta_bias[layer][neuron])

                                for weight in range(len(momentum[layer][neuron])):

                                    momentum[layer][neuron][weight] = (coefficient * momentum[layer][neuron][weight]) - \
                                        (learning_rate * delta[layer][neuron][weight])
                                    
                        # Updating weights and biases using calculated gradients with momentum.
                        update_weights_and_bias(self=self, delta=momentum,
                                delta_bias=momentum_bias)


                    else:
                        for layer in range(len(delta)):
                            for neuron in range(len(delta[layer])):

                                delta_bias[layer][neuron] *= learning_rate
                                
                                for weight in range(len(delta[layer][neuron])):
                                
                                    delta[layer][neuron][weight] *= learning_rate
                        
                        # Updating weights and biases using calculated gradients.
                        update_weights_and_bias(self=self, delta=delta,
                                delta_bias=delta_bias)

                    # Check how the output changed for this example.
                    predicted_after = self.run_once(inputs[example])

                    # Add a row to the PrettyTable with relevant data.
                    table.add_row([(epoch+1), (example+1), (iteration+1), scalar_loss, predicted, predicted_after, (predicted-predicted_after), str(inputs[example])])
                                        
                    # Exit if the change was lower than the minimum precision.
                    if abs(predicted-predicted_after) < min_precision:
                        break
        # Print the PrettyTable after all computation.
        print(table)