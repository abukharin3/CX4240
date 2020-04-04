import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import gc

class LSTM(Model):
    def __init__(self, train, train_labels, test, test_labels):
        super(LSTM, self).__init__()
        # Define training parameters
        self.learning_rate = 0.001
        self.training_steps = 1000
        self.batch_size = 500
        self.display_step = 10
        self.num_epochs = 3
        self.train = train
        self.train_labels = train_labels
        self.test = test
        self.test_labels = test_labels

        self.shuffle()
        self.num_genres = 5
        self.num_units = 32
        # Define hidden layers
        self.fc1 = layers.Dense(len(self.train[0]), activation = tf.nn.relu)
        self.lstm_layer = layers.LSTM(units = self.num_units)
        self.out = layers.Dense(self.num_genres)
        # Define Optimizer
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def call(self, x, is_training = False):
        # Forward pass
        x = self.fc1(x)
        x = self.lstm_layer(x)
        x = self.out(x)
        if not is_training:
            # Give softmax ouput if not training, otherwise keep logits
            x = tf.nn.softmax(x)
        return x

    # Define Cross entropy loss
    def cross_entropy_loss(self, pred, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = pred)
        return tf.reduce_mean(loss)

    def accuracy(self, y_pred, label):
        # Take argmax of prediction
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(label, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def run_optimization(self, x, label):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass
            pred = self.call(x, is_training = True)
            # Compute Loss
            loss = self.cross_entropy_loss(pred, label)

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.trainable_variables

        # Compute Gradients
        gradients = g.gradient(loss, trainable_variables)

        # Update Variables
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train_op(self):
        for e in range(self.num_epochs):
            self.shuffle()
            for b in range(8):
                batch_x = self.train[self.batch_size * b: self.batch_size * (b + 1)]
                batch_y = self.train_labels[self.batch_size * b: self.batch_size * (b + 1)]
                for step in range(self.training_steps):
                    self.run_optimization(batch_x, batch_y)
                    if step % self.display_step == 0:
                        pred = self.call(batch_x, is_training = True)
                        loss = self.cross_entropy_loss(pred, batch_y)
                        acc = self.accuracy(pred, batch_y)
                        test_pred = self.call(self.test, is_training = True)
                        test_acc = self.accuracy(test_pred, self.test_labels)
                        print("step: %i, loss: %f, accuracy: %f, epoch: %f, batch: %f" % (step, loss, acc, e, b))
                        print("Test Accuracy: %f" % (test_acc))

    def shuffle(self):
        train_order = np.random.permutation(len(self.train))
        self.train, self.train_labels = self.train[train_order], self.train_labels[train_order]




# Load Data
data = np.load("wordvecs.npy")
labels = np.load("wordvec_labels.npy")
test = np.load("testvecs.npy")
test_labels = np.load("test_labels.npy")

data = data[:, 0:100, :]
print("End Data Processing")

lstm = LSTM(train = data, train_labels = labels, test = test, test_labels = test_labels)
lstm.train_op()
