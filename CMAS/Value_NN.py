import numpy as np
import tensorflow as tf


class ValueFunctionWithNN():
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.X = tf.placeholder("float", [None, state_dims])
        self.Y = tf.placeholder("float", [None, 1])
        self.dims = state_dims

        weights = {
            # 4 inputs 10  nodes in h1 layer
            'h1': tf.Variable(tf.zeros([2, 32])),
            # 10 nodes in h2 layer
            'h2': tf.Variable(tf.zeros([32, 32])),
            # 1 ouput label
            'out': tf.Variable(tf.zeros([32, 1]))
        }

        biases = {
            'b1': tf.Variable(tf.zeros([32])),
            'b2': tf.Variable(tf.zeros([32])),
            'out': tf.Variable(tf.zeros([1]))
        }

        # hidden layer 1
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)  # activation

        # hideen layer 2
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)  # activation

        # output layer
        self.Y_hat = tf.matmul(layer_2, weights['out']) + biases['out']

        optimizer = tf.train.AdamOptimizer(learning_rate=.5*.001)
        self.loss_op = tf.losses.mean_squared_error(self.Y, self.Y_hat)

        self.train_op = optimizer.minimize(self.loss_op)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def __call__(self, input):
        x_test = input.reshape((1, self.dims))
        pred = self.sess.run(self.Y_hat, feed_dict={self.X: x_test})
        return np.asscalar(pred)

    def update(self, val, Input):

        Target = np.zeros(1)
        Target[0] = val
        x_train = Input.reshape((1, self.dims))
        y_train = Target.reshape((1, 1))

        self.sess.run(self.train_op, feed_dict={
                      self.X: x_train, self.Y: y_train})

        return None
