"""
    Module: pynn.py
    This module contains a definition of various PyNN convenience function.
"""

from model import Model
import pickle
import image

def create_network(num_layers, neurons_layer, activation_layer, derivative_layer, weight_function, precision):
    """
        Create new model using passed parameters

        Args:
            num_layers (int): A integer that specifies the number of layers in the network.
            neurons_layer (list): A list containing the integer number of neurons for each layer
            activation_layer (list): A list containing the activation function objects for each layer
            derivative_layer (list): A list containing the derivative function objects for each layer
            precision (numpy.dtype): The numpy dtype for the network precision.
        
        Returns:
            model (object): A ready to use neural network as defined by the parameters.
    """
    model = Model(num_layers, precision)
    for i in range(num_layers):
        model.add_layer(neurons_layer[i], activation_layer[i], derivative_layer[i], weight_function)
    # Call the model preparation function before returning the object    
    model.compile()
    return model

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