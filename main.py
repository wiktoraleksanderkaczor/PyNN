from pynn import Model
import activation
from weights import xavier_init
from optimizer import gradient_descent
from cost import mean_squared_loss
import numpy as np

from loader import mnist
train_data, train_labels, test_data, test_labels = mnist()

model = Model(precision=np.float32, weight_init=xavier_init, optimizer=gradient_descent, loss=mean_squared_loss)
# First is input.
model.add_layer(784, activation.Input)
model.add_layer(20, activation.tanh)
model.add_layer(10, activation.tanh)

import pandas as pd
label_series = pd.Series(train_labels)
train_labels = np.array(pd.get_dummies(label_series).values.tolist())
data = list(zip(train_data, train_labels))

model.train(epochs=10, training_data=data, max_iter=5, learning_rate=0.1, min_precision=0.1, learning_rate_function="momentum", coefficient=0.9)

# Saving network to file.
pynn.save_network(model)
