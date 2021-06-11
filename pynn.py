"""
    Module: pynn.py
    This module contains a definition of various PyNN convenience functions for model creation, training and inference. 
"""

import pickle


import numpy as np
from neuron import Neuron
from cost import mean_squared_loss
from optimizer import update_weights_and_bias 
from tqdm import tqdm

class Model:
    def __init__(self, precision, weight_init, optimizer, loss):
        """
            Constructor function for a Model.

            Args:
                precision (numpy.dtype): The numpy dtype for the network precision.
                weight_init (function): The function used for network weight initialisation.
        """
        self.model = {"layers": []}
        self.precision = precision
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.loss = loss        

    def add_layer(self, num_neurons, neuron_activation):
        """
            Adding a layer to the model.

            Args:
                num_neurons (int): The number of neurons in the layer to be added.
                neuron_activation (function): The activation function to use for the layer.
        """

        self.model["layers"].append(
            {"neurons": [Neuron(self.precision) for _ in range(num_neurons)],
            "activation": neuron_activation}
        )

        # Set number of weights to generate for each neuron in layer.
        layer_num = len(self.model["layers"]) - 1
        if layer_num == 0:
            num_prev_neurons = 1
        else:
            # Number of neurons in previous layer
            num_prev_neurons = len(self.model["layers"][layer_num - 1]["neurons"])

        for neuron in self.model["layers"][layer_num]["neurons"]:
            neuron.weights = self.weight_init(
                num_prev_neurons, 
                self.precision)

    def input_layer(self, values):
        input_layer = self.model["layers"][0]["neurons"]
        activation = self.model["layers"][0]["activation"]
        for neuron, value in zip(input_layer, values):
            neuron.set_sum(value)
            neuron.activation = activation(neuron.sum)

        feed_forward = np.array([neuron.activation for neuron in input_layer], dtype=self.precision).flatten()
        return feed_forward

    def step(self, inputs):
        """
            Feeding a given set of inputs to the network and returning the predicted values for the output layer.

            Args:
                inputs (numpy.array): A 1-dimensional array of the inputs to feed to the input layer.

            Returns:
                feed_forward (numpy.array): A 1-dimensional array containing the activation values for the output layer neurons. 
        """
        # Set input layer, trigger and get outputs for feed forward.
        feed_forward = self.input_layer(inputs)

        # For each layer in the model, set inputs, get outputs.
        for layer in self.model["layers"][1:]:
            # Set inputs from previous layer and activate
            for neuron in layer["neurons"]:
                neuron.set_sum(feed_forward)
                neuron.activation = layer["activation"](neuron.sum)
            
            # Get output for new layer.
            feed_forward = np.array([neuron.activation for neuron in layer["neurons"]], dtype=self.precision).flatten()

        # Return activation for last layer.
        return feed_forward


    def train(self, epochs, training_data, learning_rate=0.01, min_precision=0.001):
        """
            Runs a optimizing algorithm on the neural network using a specified algorithm.

            Args:
                epochs (int): The number of times that the dataset is shown to the network.
                data (numpy.array, numpy.array): Two zipped 1-dimensional arrays of the input data and the expected values.
                learning_rate (float): The learning rate hyperparameter for determining how quickly a network can learn.
                min_precision (float): The minimum change in the error of the network for until it exits.
        """
        # Repeating training the network on the examples for N epochs.
        for epoch in range(epochs):
            # For each example do...
            epoch_loss = 0
            tq = tqdm(total=len(training_data))
            for input_data, expected in training_data:
                # Run for given example
                predicted = self.step(input_data)

                # Loss for optimizers
                loss = self.loss(predicted, expected)  
                
                # Add to total epoch loss for tracking.
                epoch_loss += loss

                # Run training on single example.
                weight_gradients, bias_gradients = self.optimizer(self.model, input_data, loss=loss)
                update_weights_and_bias(self.model, weight_gradients, bias_gradients, learning_rate=learning_rate)
                
                tq.update(1)
                tq.set_postfix_str("Loss: {}".format(loss))
            
            tq.close()
                
        # Print statistics
        print("EPOCH: ", epoch)
        print("EPOCH LOSS: ", epoch_loss)

        # Exit condition
        if epoch_loss <= min_precision:
            print("Reached desired precision.")
            return
        else:
            epoch_loss = 0


def save_network(model):
    """
        Save the passed model to a file for later use.

        Args:
            model (object): A ready to use neural network in a python object.
    """
    # Open the file for writing as "open_file"
    with open('./model.obj', 'wb') as open_file:
        # Write out the object to disk.
        pickle.dump(model, open_file)

def load_network():
    """
        Load a neural network from a file for use.

        Returns:
            loaded_object (object): A neural network contained within a python object.
    """
    # Open the file for reading as "open_file"
    with open('./model.obj', 'rb') as open_file:
        # Load the object from file
        loaded_object = pickle.load(open_file)
 
    # After loaded_object is read from file
    print(loaded_object)
    return loaded_object