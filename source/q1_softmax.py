import numpy as np
import tensorflow as tf
import unittest
import random


def softmax(x):
    """
     Compute the softmax function in tensorflow.

     You might find the tensorflow functions tf.exp, tf.reduce_max,
     tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you
     may not need to use all of these functions). Recall also that many common
     tensorflow operations are sugared (e.g. x * y does a tensor multiplication
     if x and y are both tensors). Make sure to implement the numerical
     stability fixes as in the previous homework!

     Args:
       x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors
            are represented by row-vectors. (For simplicity, no need to
            handle 1-d input as in the previous homework)
     Returns:
       out: tf.Tensor with shape (n_sample, n_features). You need to construct
            this tensor in this problem.
     """

    # ## YOUR CODE HERE
    all_constants = - tf.reduce_max(x, axis=1)
    x = x + tf.expand_dims(all_constants, 1)
    x = tf.exp(x)
    all_sums = tf.reduce_sum(x, 1)
    all_sums = tf.pow(all_sums, -1)
    out = x*tf.expand_dims(all_sums, 1)
    # ## END YOUR CODE

    return out


def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and that
    should be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful.
    (Many solutions are possible, so you may not need to use all of
    these functions).

    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
          functions.

    Args:
      y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
      yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
            probability distribution and should sum to 1.
    Returns:
      out:  tf.Tensor with shape (1,) (Scalar output). You need to construct
            this tensor in the problem.
    """
    # ## YOUR CODE HERE
    y = tf.cast(y, tf.float32)
    yhat = tf.log(yhat)
    out = - tf.reduce_sum(y*yhat)
    out = tf.reshape(out, (1,))
    # ## END YOUR CODE
    return out


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(tf.convert_to_tensor(
        np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(tf.convert_to_tensor(
        np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session():
        test2 = test2.eval()
    assert np.amax(np.fabs(test2 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "Basic (non-exhaustive) softmax tests pass\n"


def test_cross_entropy_loss_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(
        tf.convert_to_tensor(y, dtype=tf.int32),
        tf.convert_to_tensor(yhat, dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    result = -3 * np.log(.5)
    assert np.amax(np.fabs(test1 - result)) <= 1e-6
    print "Basic (non-exhaustive) cross-entropy tests pass\n"


class TestSoftmax(unittest.TestCase):

    def test_upperbound(self):
        for k in range(10):
            y = np.ndarray(shape=(1, 10), dtype=float)
            for i in range(10):
                y[0][i] = random.randint(-100, 100)
        test1 = softmax(tf.convert_to_tensor(y, dtype=tf.float32))
        with tf.Session():
            sum_test = np.sum(test1.eval())
            self.assertTrue(sum_test <= 1.0001,
                            """\n if y = {0} \n, then np.sum(softmax(y))
                            = {1} is bigger than 1"""
                            .format(y, sum_test))

    def test_high_low(self):
        for k in range(10):
            y = np.ndarray(shape=(2, 2), dtype=float)
            y[0][0] = random.randint(-20000, -10000)
            y[0][1] = y[0][0]+1
            y[1][0] = random.randint(10000, 20000)
            y[1][1] = y[1][0]+1
            test1 = softmax(tf.convert_to_tensor(y, dtype=tf.float32))
            with tf.Session():
                test1 = test1.eval()
            self.assertTrue(np.amax(np.fabs(test1 - np.array(
             [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6,
                            """\n if y = {0} \n,
                            then softmax is {1}: problem"""
                            .format(y, test1))


class TestCrossEntropy(unittest.TestCase):

    def test_random(self):
        for k in range(10):
            y = np.array([0, 0, 1])
            yhat = np.ndarray(shape=(1, 3), dtype=float)
            yhat[0][0] = random.randint(1, 10)
            yhat[0][1] = random.randint(10, 20)
            yhat[0][2] = random.randint(30, 50)
            yhat = softmax(tf.convert_to_tensor(yhat, dtype=tf.float32))
            test1 = cross_entropy_loss(tf.convert_to_tensor(y, dtype=tf.int32),
                                       yhat)
            with tf.Session():
                test1 = test1.eval()
                yhat = yhat.eval()
                result = -np.log(yhat[0][2])
            self.assertTrue(np.amax(np.fabs(test1 - result)) <= 1e-6,
                            """\n if y = {0} and yhat = {1} \n,
                            then
                            test1 = {2} and result = {3}""" .format(y,
                                                                    yhat,
                                                                    test1,
                                                                    result))


def my_test():
    print("Running your tests...")
    suite = unittest.TestSuite()
    for method in dir(TestSoftmax):
        if method.startswith("test"):
            suite.addTest(TestSoftmax(method))
    for method in dir(TestCrossEntropy):
        if method.startswith("test"):
            suite.addTest(TestCrossEntropy(method))
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
    my_test()
