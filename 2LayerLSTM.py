import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

sns.set_style("darkgrid")


directory = os.getcwd()

#Load training and test data

# Paths to training label and data .npy files go here
train_data = np.load()
train_labels = np.load()

# Paths to test labels and data .npy files go here
test_data = np.load()
test_labels = np.load()

train_data = keras.utils.normalize(train_data)
test_data = keras.utils.normalize(test_data)

val_data = train_data[:450]
partial_train_data = train_data[450:]

val_labels = train_labels[:450]
partial_train_labels = train_labels[450:]

print(train_data.shape)
print(test_data.shape)

scores = []

def run_training(plot=True):

    verbose = 1
    epochs = 200
    batch_size = 100

    n_timesteps, n_features, n_outputs = train_data.shape[1], train_data.shape[2], train_labels.shape[1]

    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(keras.layers.Dense(n_outputs, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])


    history = model.fit(train_data,
                        train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, val_labels),
                        verbose=verbose)

    _, accuracy = model.evaluate(test_data,
                                 test_labels,
                                 batch_size=batch_size,
                                 verbose=verbose)


    scores.append(accuracy)

    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']


    if plot:
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


n = 5

def run_experiment(n):
    for i in range(n):
        run_training(plot=False)
    print(scores)
    print('M={}'.format(np.mean(scores)), 'STD={}'.format(np.std(scores)))
    print('Min={}'.format(np.min(scores)), 'Max={}'.format(np.max(scores)))


run_experiment(n)
