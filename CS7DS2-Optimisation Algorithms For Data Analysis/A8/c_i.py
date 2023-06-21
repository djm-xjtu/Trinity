from tensorflow import keras
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from b_i import new_global_search
from a_i import random_search

def get_model_loss(batch_size, alpha, beta1, beta2, epochs):
    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n = 5000
    x_train = x_train[1:n]
    y_train = y_train[1:n]
    # x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
        metrics=["accuracy"])
    y_predicts = model.predict(x_test)
    loss = CategoricalCrossentropy()
    return loss(y_test, y_predicts).numpy()


if __name__ == '__main__':
    n = 5
    data_range = [
        [1, 128],
        [0.001, 0.001],
        [0.9, 0.9],
        [0.99, 0.99],
        [15, 15]
    ]
    global_random_search_x_list, global_random_search_f_list = random_search(get_model_loss, n, data_range, N=30)
    new_global_search_x_list, new_global_search_f_list = new_global_search(get_model_loss, n, data_range, N=12, M=4,
                                                                           itr_times=4)
    global_random_search_x_list_, new_global_search_x_list_ = list(range(len(global_random_search_f_list))), list(range(len(new_global_search_f_list)))
    plt.plot(global_random_search_x_list_, global_random_search_f_list, label='Global Random Search')
    plt.plot(new_global_search_x_list_, new_global_search_f_list, label='New Global Search')
    plt.xlabel('function evaluations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()