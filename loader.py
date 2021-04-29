import numpy as np

# https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
def mnist():
    from mlxtend.data import loadlocal_mnist
    # Getting MNIST dataset.
    train_data, train_labels = loadlocal_mnist(
        images_path='loader_data/train-images-idx3-ubyte', 
        labels_path='loader_data/train-labels-idx1-ubyte')
    print('Dimensions: %s x %s' % (train_data.shape[0], train_data.shape[1]))
    #print('\n1st row', train_data[0])

    print('Digits:  0 1 2 3 4 5 6 7 8 9')
    print('labels: %s' % np.unique(train_labels))
    print('Class distribution: %s' % np.bincount(train_labels))

    test_data, test_labels = loadlocal_mnist(
        images_path='loader_data/train-images-idx3-ubyte', 
        labels_path='loader_data/train-labels-idx1-ubyte')
    print('Dimensions: %s x %s' % (test_data.shape[0], test_data.shape[1]))
    #print('\n1st row', test_data[0])

    print('Digits:  0 1 2 3 4 5 6 7 8 9')
    print('labels: %s' % np.unique(test_labels))
    print('Class distribution: %s' % np.bincount(test_labels))

    return train_data, train_labels, test_data, test_labels
