import os
import getpass
import sys
import time
import inspect

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self,
                 embed_size=50,
                 batch_size=84,
                 label_size=5,
                 hidden_size=100,
                 max_epochs=24,
                 early_stopping=2,
                 dropout=0.991323729933,
                 lr=0.00365884577219,
                 l2=1.7095245617e-05,
                 window_size=3):
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.lr = lr
        self.l2 = l2
        self.window_size = window_size


class NERModel(LanguageModel):
    """Implements a NER (Named Entity Recognition) model.

    This class implements a deep network for named entity recognition. It
    inherits from LanguageModel, which has an add_embedding method
    in addition to the standard Model method.
    """

    def load_data(self, debug=False, search=False):
        """Loads starter word-vectors and train/dev/test data."""
        # Load the starter word vectors
        path_vocab = 'data/ner/vocab.txt'
        path_wordVectors = 'data/ner/wordVectors.txt'
        path_train = 'data/ner/train'
        path_dev = 'data/ner/dev'
        path_test = 'data/ner/test.masked'
        if search:
            currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            path_vocab = currentdir + "/" + path_vocab
            path_wordVectors = currentdir + "/" + path_wordVectors
            path_train = currentdir + "/" + path_train
            path_dev = currentdir + "/" + path_dev
            path_test = currentdir + "/" + path_test
        self.wv, word_to_num, num_to_word = ner.load_wv(
          path_vocab, path_wordVectors)
        tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
        self.num_to_tag = dict(enumerate(tagnames))
        tag_to_num = {v: k for k, v in self.num_to_tag.iteritems()}

        # Load the training set
        docs = du.load_dataset(path_train)
        self.X_train, self.y_train = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_train = self.X_train[:1024]
            self.y_train = self.y_train[:1024]

        # Load the dev set (for tuning hyperparameters)
        docs = du.load_dataset(path_dev)
        self.X_dev, self.y_dev = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_dev = self.X_dev[:1024]
            self.y_dev = self.y_dev[:1024]

        # Load the test set (dummy labels only)
        docs = du.load_dataset(path_test)
        self.X_test, self.y_test = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.
        Note that when "None" is in a placeholder's shape, it's flexible

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                           (None, window_size), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape
                            (None, label_size), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar),
                             type tf.float32

        Add these placeholders to self as the instance variables

          self.input_placeholder
          self.labels_placeholder
          self.dropout_placeholder

        (Don't change the variable names)
        """
        # ## YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=[None,
                                                       self.config.window_size],
                                                name="input_placeholder")
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None,
                                                        self.config.label_size],
                                                 name="labels_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  shape=[],
                                                  name="dropout_value")
        # ## END YOUR CODE

    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        """Creates the feed_dict for softmax classifier.

        A feed_dict takes the form of:

        feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....}
        Hint: The keys for the feed_dict should be a subset of the placeholder
              tensors created in add_placeholders.
        Hint: When label_batch is None,
              don't add a labels entry to the feed_dict.

        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # ## YOUR CODE HERE
        if label_batch is None:
            feed_dict = {self.input_placeholder: input_batch,
                         self.dropout_placeholder: dropout}
        else:
            feed_dict = {self.input_placeholder: input_batch,
                         self.labels_placeholder: label_batch,
                         self.dropout_placeholder: dropout}
        # ## END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Add embedding layer that maps from vocabulary to vectors.

        Creates an embedding tensor (of shape (len(self.wv), embed_size).
        Use the input_placeholder to retrieve the embeddings for words
        in the current batch.

        (Words are discrete entities. They need to be transformed into
        vectors for use in deep-learning. Although we won't do so in
        this problem, in practice it's useful to initialize the
        embedding with pre-trained word-vectors. For this
        problem, using the default initializer is sufficient.)

        Hint: This layer should use the input_placeholder to index into the
              embedding.
        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: See following link to understand what -1 in a shape means.
          https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
        https://www.tensorflow.org/api_docs/python/tf/reshape
        Hint: Check the last slide from the TensorFlow lecture.
        Hint: Here are the dimensions of the variables you will need to create:

          L: (len(self.wv), embed_size)

        Returns:
          window: tf.Tensor of shape (-1, window_size*embed_size)
        """
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
            # ## YOUR CODE HERE
            Linit = tf.constant_initializer(self.wv)
            L = tf.get_variable("L",
                                shape=[len(self.wv), self.config.embed_size],
                                dtype='float32',
                                initializer=Linit)
            window = tf.nn.embedding_lookup(L, self.input_placeholder)
            window = tf.reshape(window,
                                (-1,
                                 self.config.window_size*self.config.embed_size))
        # ## END YOUR CODE
        return window

    def add_model(self, window):
        """Adds the 1-hidden-layer NN.

        Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer,
              and another variable_scope (e.g. "Softmax") for the linear
              transformation preceding the softmax. Make sure to
              use the xavier_weight_init you defined in the previous
              part to initialize weights.
        Hint: Make sure to add in regularization and dropout to this network.
              Regularization should be an addition to the cost function, while
              dropout should be added after both variable scopes.
        Hint: You might consider using a tensorflow Graph Collection (e.g
              "total_loss") to collect the regularization and loss terms
              (which you will add in add_loss_op below).
        Hint: Here are the dimensions of the various variables you will need to
              create

              W:  (window_size*embed_size, hidden_size)
              b1: (hidden_size,)
              U:  (hidden_size, label_size)
              b2: (label_size)

        https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
        Args:
          window: tf.Tensor of shape (-1, window_size*embed_size)
        Returns:
          output: tf.Tensor of shape (batch_size, label_size)
        """
        # ## YOUR CODE HERE
        # shapes
        Wshape = (self.config.window_size*self.config.embed_size,
                  self.config.hidden_size)
        b1shape = (1, self.config.hidden_size)
        Ushape = (self.config.hidden_size, self.config.label_size)
        b2shape = (1, self.config.label_size)

        # initializers
        xavier_initializer = xavier_weight_init()
        Winit = xavier_initializer(Wshape)
        b1init = xavier_initializer(b1shape)
        Uinit = xavier_initializer(Ushape)
        b2init = xavier_initializer(b2shape)

        with tf.variable_scope("Layer"):
            self.W = tf.get_variable("weights",
                                     dtype='float32',
                                     initializer=Winit)
            self.b1 = tf.get_variable("bias",
                                      dtype='float32',
                                      initializer=b1init)
            linear_op = tf.matmul(window, self.W) + self.b1
            first_output = tf.nn.dropout(tf.tanh(linear_op),
                                         self.config.dropout,
                                         name="output")
            tf.add_to_collection("reg", tf.reduce_sum(tf.pow(self.W, 2)))
        with tf.variable_scope("Softmax"):
            self.U = tf.get_variable("weights",
                                     dtype='float32',
                                     initializer=Uinit)
            self.b2 = tf.get_variable("bias",
                                      dtype='float32',
                                      initializer=b2init)
            output = tf.nn.dropout(tf.matmul(first_output, self.U) + self.b2,
                                   self.config.dropout,
                                   name="output")
            tf.add_to_collection("reg", tf.reduce_sum(tf.pow(self.U, 2)))
        # END YOUR CODE
        return output

    def add_loss_op(self, y):
        """Adds cross_entropy_loss ops to the computational graph.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to
              simplify your implementation.
              You might find tf.reduce_mean useful.
        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        # ## YOUR CODE HERE
        pred = self.labels_placeholder
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, pred))
        regularization = self.config.l2*0.5*sum(tf.get_collection("reg"))
        loss = cross_entropy + regularization
        # ## END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients
        to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()`
        call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.AdamOptimizer for this model.
              Calling optimizer.minimize() will return a train_op object.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        # ## YOUR CODE HERE
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        # ## END YOUR CODE
        return train_op

    def __init__(self, config, debug=False, search=False):
        """Constructs the network using the helper functions defined above."""
        self.config = config
        self.load_data(debug, search)
        self.add_placeholders()
        window = self.add_embedding()
        y = self.add_model(window)

        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_prediction = tf.equal(
            tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction,
                                                 'int32'))
        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, input_labels,
                  shuffle=True, verbose=True):
        orig_X, orig_y = input_data, input_labels
        dp = self.config.dropout
        # We're interested in keeping
        # track of the loss and accuracy during training
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        for step, (x, y) in enumerate(
          data_iterator(orig_X,
                        orig_y,
                        batch_size=self.config.batch_size,
                        label_size=self.config.label_size,
                        shuffle=shuffle)):
            feed = self.create_feed_dict(input_batch=x,
                                         dropout=dp,
                                         label_batch=y)
            loss, total_correct, _ = session.run([self.loss,
                                                  self.correct_predictions,
                                                  self.train_op],
                                                 feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)
            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(step,
                                                                total_steps,
                                                                np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

    def predict(self, session, X, y=None):
        """Make predictions from the provided model."""
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        dp = 1
        losses = []
        results = []
        if np.any(y):
            data = data_iterator(X,
                                 y,
                                 batch_size=self.config.batch_size,
                                 label_size=self.config.label_size,
                                 shuffle=False)
        else:
            data = data_iterator(X,
                                 batch_size=self.config.batch_size,
                                 label_size=self.config.label_size,
                                 shuffle=False)
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run(
                                          [self.loss, self.predictions],
                                          feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
        return np.mean(losses), results


def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    # print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)


def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros(
                         (config.label_size, config.label_size),
                         dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion


def save_predictions(predictions, filename):
    """Saves predictions to provided file."""
    with open(filename, "wb") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")


def test_NER(config, save=True, verbose=True, debug=False, search=False):
    """Test NER model implementation.
    You can use this function to test your implementation of the Named Entity
    Recognition network. When debugging, set max_epochs in the Config object to
    1 so you can rapidly iterate.
    """
    initial_time = time.time()
    with tf.Graph().as_default():
        model = NERModel(config, debug, search)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0
            session.run(init)
            for epoch in xrange(config.max_epochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                ###
                train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                        model.y_train)
                val_loss, predictions = model.predict(session,
                                                      model.X_dev,
                                                      model.y_dev)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Validation loss: {}'.format(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if save:
                        if not os.path.exists("./weights"):
                            os.makedirs("./weights")
                        saver.save(session, './weights/ner.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
                ###
                if verbose:
                    confusion = calculate_confusion(config,
                                                    predictions,
                                                    model.y_dev)
                    print_confusion(confusion, model.num_to_tag)
                    print 'Total time: {}'.format(time.time() - start)

            if save:
                saver.restore(session, './weights/ner.weights')
                print 'Test'
                print '=-=-='
                print 'Writing predictions to q2_test.predicted'
                _, predictions = model.predict(session,
                                               model.X_test,
                                               model.y_test)
                save_predictions(predictions, "q2_test.predicted")
    duration = time.time() - initial_time
    return best_val_loss, duration

if __name__ == "__main__":
    config = Config()
    val_loss, duration = test_NER(config, debug=True)
    print("The best val_loss is {0} and the whole training takes {1}(s)".format(val_loss,
                                                               duration))
