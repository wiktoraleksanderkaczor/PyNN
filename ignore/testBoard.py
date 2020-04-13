# For optional X1
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import time

# Optionally, X1, read images and show them in a pyplot.
# img = mpimg.imread("images/image0.png")
# imgplot = plt.imshow(img)
# plt.show(imgplot)

# How to time:
# t1 = time.time()
# t2 = time.time()
# print(t1 - t2)

...

# The webcam can do 640*480. 4:3 aspect ratio
# numPix = 640*480
numPix = 320 * 240

# Temporary performance monitoring:
t1 = time.time()
# THING TO MONITOR HERE.
print(time.time() - t1)


# STOCHASTIC_GRADIENT_DESCENT
# In a “purist” implementation of SGD, your mini-batch size would be set to 1.
# However, we often uses mini-batches that are > 1. Typical values include 32,
# 64, 128, and 256.
# https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/

# You can use:
# mse = ((A - B)**2).mean(axis=ax)
# Or
# mse = (np.square(A - B)).mean(axis=ax)
# with ax=0 the average is performed along the row, for each column, returning an array
# with ax=1 the average is performed along the column, for each row, returning an array
# with ax=None the average is performed element-wise along the array, returning a scalar value


# Old GD function;

# WORK_IN_PROGRESS
# Calculates and updates the bias and weight for a neuron with the loss passed to it for
# the number of examples calculated from the predicted array shape.
def gradient_descent(weight, bias, learning_rate, predicted, actual, loss, max_iterations):
    dW = 0  # Weight gradient, i.e. accumulator for weight
    # dB = 0 #Bias gradient, i.e. accumulator for bias
    num_examples = predicted.shape[0]  # No of training examples
    for i in range(0, max_iterations):
        # Reset gradients
        dW = 0
        # dB = 0
        for j in range(0, num_examples):
            dW = dW - learning_rate * (loss[j] * (predicted[j] - actual[j]))
            # dB = dB - learning_rate * (loss[j] * (predicted[j] - actual[j]))

        dW = dW / num_examples
        # dB = dB / num_examples

        # Update weight and bias
        weight += np.sum(dW)
        # bias += dB

    pass
