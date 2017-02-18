import numpy as np
import tensorflow as tf
import unittest
import random


def xavier_weight_init():
    """
    Returns function that creates random tensor.

    The specified function will take in a shape (tuple or 1-d array) and must
    return a random tensor of the specified shape and must be drawn from the
    Xavier initialization distribution.

    Hint: You might find tf.random_uniform useful.
    """
    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.

        This function will be used as a variable scope initializer.

        https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope
        https://www.tensorflow.org/programmers_guide/variable_scope
        Args:
          shape: Tuple or 1-d array that species dimensions of
                 requested tensor.
        Returns:
          out: tf.Tensor of specified shape sampled from Xavier distribution.
        """
        # ## YOUR CODE HERE
        epsilon = np.sqrt(6.)/np.sqrt(np.sum(shape))
        out = tf.random_uniform(shape,
                                minval=-epsilon,
                                maxval=epsilon,
                                dtype=tf.float32,
                                name='weights')
        # ## END YOUR CODE
        return out
    # Returns defined initializer function.
    return _xavier_initializer


def test_initialization_basic():
    """
    Some simple tests for the initialization.
    """
    print "Running basic tests..."
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape

    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    print "Basic (non-exhaustive) Xavier initialization tests pass\n"


class TestXavier(unittest.TestCase):
    def test_bounds(self):
        for i in range(100):
            random_row = random.randint(1, 100)
            random_collumn = random.randint(1, 100)
            epsilon = np.sqrt(6.)/np.sqrt(np.sum(random_row + random_collumn))
            random_row_obs = random.randint(0, random_row-1)
            random_collumn_obs = random.randint(0, random_collumn-1)
            xavier_initializer = xavier_weight_init()
            shape = (random_row, random_collumn)
            xavier_mat = xavier_initializer(shape)
            with tf.Session():
                test = xavier_mat.eval()[random_row_obs][random_collumn_obs]
            self.assertTrue(test <= epsilon,
                            """bigger than the upper bound,
                            test = {0}, epsilon
                            = {1}"""
                            .format(test, epsilon))
            self.assertTrue(-epsilon <= test,
                            """lower than the lower bound,
                            test = {0}, -epsilon
                            = {1}"""
                            .format(test, -epsilon))


def test_initialization():
    """
    Use this space to test your Xavier initialization code by running:
        python q1_initialization.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    # ## YOUR CODE HERE
    suite = unittest.TestSuite()
    for method in dir(TestXavier):
        if method.startswith("test"):
            suite.addTest(TestXavier(method))
    unittest.TextTestRunner().run(suite)
    # ## END YOUR CODE

if __name__ == "__main__":
    test_initialization_basic()
    test_initialization()
