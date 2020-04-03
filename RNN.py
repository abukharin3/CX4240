import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import gc

class LSTM(Model):
    def __init__(self, train, train_labels):
        super(LSTM, self).__init__()
        # Define training parameters
        self.learning_rate = 0.01
        self.training_steps = 2000
        self.num_batches = 50
        self.display_step = 1
        self.num_epochs = 3
        self.train = train
        self.train_labels = train_labels

        self.shuffle()
        self.num_genres = 5
        self.num_units = 32
        # Define hidden layers
        self.lstm_layer = layers.LSTM(units = self.num_units)
        self.out = layers.Dense(self.num_genres)
        # Define Optimizer
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def call(self, x, is_training = False):
        # Forward pass
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
            order = self.shuffle()
            for b in range(10):
                batch_x = self.train[order][self.num_batches * b: self.num_batches * (b + 1)]
                batch_y = self.train_labels[order][self.num_batches * b: self.num_batches * (b + 1)]
                for step in range(self.training_steps):
                    self.run_optimization(batch_x, batch_y)
                    if step % self.display_step == 0:
                        pred = self.call(batch_x, is_training = True)
                        loss = self.cross_entropy_loss(pred, batch_y)
                        acc = self.accuracy(pred, batch_y)
                        print("step: %i, loss: %f, accuracy: %f, epoch: %f, batch: %f" % (step, loss, acc, e, b))

    def shuffle(self):
        train_order = np.random.permutation(len(self.train))
        return train_order




# Load Data
data = np.load("wordvecs.npy")
labels = np.load("wordvec_labels.npy")
data = data[:, 0:2000, :]
print("End Data Processing")
lstm = LSTM(train = data, train_labels = labels)
lstm.train_op()
