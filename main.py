import pynn
from activation import *
from derivative import *
import numpy as np

# Loading network from file.
#model = pynn.load_network()

# Declaration and definition of model.
# NOTE: IF CHANGING MODEL AT ALL, YOU HAVE TO REGENERATE AND SAVE THE NETWORK.
# THIS MEANS THAT IT'S ONLY GOOD FOR QUICK STARTUPS AND INFERENCE BUT NOT DEVELOPMENT.
model = pynn.create_network(num_layers=4, neurons_layer=[3, 2, 3, 1],
                            activation_layer=[none, tanh, tanh, tanh], 
                            derivative_layer=[none_derivative, tanh_derivative, tanh_derivative, tanh_derivative],
                            weight_function="xavier",
                            precision=np.float32)

model.train(epochs=2, function="GD", inputs=[[1,1,0]]*100,
            actual=np.ones(100), max_iter=20, learning_rate=0.1, min_precision=0.001)#, learning_rate_function="momentum")
model.train(epochs=2, function="GD", inputs=[[1,0,0]]*100,
            actual=np.zeros(100), max_iter=20, learning_rate=0.1, min_precision=0.001)

# Saving network to file.
pynn.save_network(model)
