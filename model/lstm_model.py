import tensorflow as tf
import logging
from utils.setup_logger import logger
logger = logging.getLogger('app.lstm_model')


class Lstm_model(object):
    def __init__(self, config):
        self._n_steps = config['time_step']
        self._input_size = config['num_feature']
        self._output_size = config['num_classes']
        self._layers = config['lstm_layers']
        self._num_units = 256
        self._learning_rate = config['learning_rate']
        self._batch_size = config['batch_size']
        with tf.name_scope('inputs_layer'):
            self.xs = tf.placeholder(
                tf.float32, [None, self._n_steps, self._input_size], name='xs')
            self.ys = tf.placeholder(
                tf.float32, [None, self._output_size], name='ys')
        with tf.variable_scope('LSTM_cell'):
            self._rnn_outputs = self.add_cell()
        with tf.variable_scope('out_hidden_layer'):
            self.outputs = self.add_output_layer()
        with tf.name_scope('cost'):
            self.loss = self.compute_lost()
        with tf.name_scope('optimizer'):
            self.train_op = self.optimization()
        logger.info('build model success')

    def add_cell(self):
        if isinstance(self._layers[0], dict):
            layers = [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    layer['num_units'], activation=tf.nn.leaky_relu, forget_bias=1.0,
                    state_is_tuple=True, use_peepholes=True
                ),
                layer['keep_prob']
            ) if layer.get('keep_prob') else tf.contrib.rnn.LSTMCell(
                layer['num_units'], activation=tf.nn.leaky_relu, forget_bias=1.0,
                state_is_tuple=True, use_peepholes=True
            ) for layer in self._layers
            ]
            self._num_units = self._layers[-1]['num_units']
        else:
            return [tf.contrib.rnn.LSTMCell(self._num_units, activation=tf.nn.leaky_relu,
                                            forget_bias=1.0, use_peepholes=True,
                                            state_is_tuple=True)
                    for _ in layers]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
        rnn_outputs, states = tf.nn.dynamic_rnn(
            multi_layer_cell, self.xs, dtype=tf.float32)
        return rnn_outputs

    def add_output_layer(self):
        stacked_rnn_outputs = tf.reshape(
            self._rnn_outputs, [-1, self._num_units])
        stacked_outputs = tf.layers.dense(
            stacked_rnn_outputs, self._output_size)
        outputs = tf.reshape(
            stacked_outputs, [-1, self._n_steps, self._output_size])
        outputs = outputs[:, self._n_steps-1, :]
        return outputs

    def compute_lost(self):
        lost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.outputs, labels=self.ys))
        return lost

    def optimization(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self._learning_rate).minimize(self.loss)
        return optimizer
