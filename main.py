from pynn import Model, save_network, load_network
import activation
from weights import xavier_init, random_init
from optimizer import gradient_descent
from cost import mean_squared_loss, sum_squared_loss
import numpy as np

from loader import mnist

def main():
    train_data, train_labels, test_data, test_labels = mnist()

    #model = load_network()
    model = Model(precision=np.float16, weight_init=xavier_init, optimizer=gradient_descent, loss=sum_squared_loss)
    # First is input.
    model.add_layer(784, activation.sigmoid)
    model.add_layer(20, activation.sigmoid)
    model.add_layer(10, activation.sigmoid)

    import pandas as pd
    label_series = pd.Series(train_labels)
    train_labels = np.array(pd.get_dummies(label_series).values.tolist())
    data = np.array(list(zip(train_data, train_labels)))

    model.train(epochs=1, training_data=data, learning_rate=0.01, min_precision=0.1)

    # Saving network to file.
    save_network(model)

if __name__ == '__main__':
    do_profiling = False
    if do_profiling:
        import cProfile
        from pstats import Stats, SortKey
        with cProfile.Profile() as pr:
            main()

        with open('profiling_stats.txt', 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    else:
        main()