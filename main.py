import pynn
from activation import *
from derivative import *
import numpy as np

# Loading network from file.
#model = pynn.load_network()

# Declaration and definition of model.
# NOTE: IF CHANGING MODEL AT ALL, YOU HAVE TO REGENERATE AND SAVE THE NETWORK.
# THIS MEANS THAT IT'S ONLY GOOD FOR QUICK STARTUPS AND INFERENCE BUT NOT DEVELOPMENT.
model = pynn.create_network(num_layers=5, neurons_layer=[3, 2, 3, 2, 1],
                            activation_layer=[none, tanh, tanh, tanh, tanh], 
                            derivative_layer=[none_derivative, tanh_derivative, tanh_derivative, tanh_derivative, tanh_derivative],
                            weight_function="xavier",
                            precision=np.float32)

"""
# Run without momentum.
model.train(epochs=10, function="GD", inputs=[[1,1,0]]*5,
            actual=np.ones(5), max_iter=5, learning_rate=0.1, min_precision=0.1)
"""

# Run with momentum.
model.train(epochs=10, function="GD", inputs=[[0,0,1]]*5,
            actual=np.zeros(5), max_iter=5, learning_rate=0.1, min_precision=0.1, learning_rate_function="momentum", coefficient=0.9)

# Saving network to file.
pynn.save_network(model)
