import numpy as np
import tensorflow as tf

class Data:

    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()


    def prepare_dataset(self, selected_digits):
        # Filter the dataset for the digits 2, 3, 8, and 9
        train_filter = np.isin(self.y_train, selected_digits)
        test_filter = np.isin(self.y_test, selected_digits)

        self.x_train = self.x_train[train_filter]
        self.x_test = self.x_test[test_filter]
        self.y_train = self.y_train[train_filter]
        self.y_test = self.y_test[test_filter]

        # Flatten the 28x28 images into 784-dimensional vectors
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)


    def print_data(self):
        # Print the shapes to verify
        print("Training set shape (images):", self.x_train.shape)
        print("Training set shape (labels):", self.y_train.shape)
        print("Test set shape (images):", self.x_test.shape)
        print("Test set shape (labels):", self.y_test.shape)


    def normalize_data(self, X):
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)

        X_standardized = np.where(std_dev != 0, (X - mean) / std_dev, 0)
        return X_standardized
