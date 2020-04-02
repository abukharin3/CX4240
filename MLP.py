import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

class MLP(Model):
    '''
    Multi-layer perceptron for text
    classification
    '''
    def __init__(self, train, train_labels, test, test_labels):
        super(MLP, self).__init__()
        # Define training parameters
        self.learning_rate = 0.01
        self.training_steps = 2000
        self.num_batches = 40
        self.display_step = 100
        self.num_epochs = 3
        self.train = train
        self.train_labels = train_labels
        self.test = test
        self.test_labels = test_labels
        self.shuffle()
        self.num_genres = 5
        # Define hidden layers
        self.fc1 = layers.Dense(len(self.train[0]), activation = tf.nn.relu)
        self.fc2 = layers.Dense(256, activation = tf.nn.relu)
        self.out = layers.Dense(self.num_genres, activation = tf.nn.relu)
        # Define Optimizer
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    # Forward pass
    def call(self, x, is_training = False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # Training expects logits so we do not apply softmax
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
            for b in range(self.num_batches):
                batch_x = self.train[self.num_batches * b: self.num_batches * (b + 1)]
                batch_y = self.train_labels[self.num_batches * b: self.num_batches * (b + 1)]
                for step in range(self.training_steps):
                    self.run_optimization(batch_x, batch_y)
                    if step % self.display_step == 0:
                        pred = self.call(self.train, is_training = True)
                        loss = self.cross_entropy_loss(pred, self.train_labels)
                        acc = self.accuracy(pred, loss)
                        print("step: %i, loss: %f, accuracy: %f, epoch: %f, batch: %f" % (step, loss, acc, e, b))


    def shuffle(self):
        train_order = np.random.permutation(len(self.train))
        self.train, self.train_labels = self.train[train_order], self.train_labels[train_order]
        test_order = np.random.permutation(len(self.test))
        self.test, self.test_labels = self.test[test_order], self.test_labels[test_order]

#Load Data
train_path = r"C:\Users\Alexander\Downloads\reduced_training_songs.npy"
test_path = r"C:\Users\Alexander\Downloads\reduced_test_songs.npy"

test_songs = np.load(test_path, allow_pickle = True)
train_songs = np.load(train_path, allow_pickle = True)

# Separate labels from data
train = []
train_labels = []
for x in train_songs:
    train.append(x[1])
    train_labels.append(x[0])
train = np.array(train)
train_labels = np.array(train_labels)

test_labels = []
test = []
for y in test_songs:
    test.append(y[1])
    test_labels.append(y[0])
test = np.array(test)
test_labels = np.array(test_labels)


print(len(train_labels))
print("End Data Processing")
mlp = MLP(train = train, train_labels = train_labels, test = test, test_labels = test_labels)
mlp.train_op()
