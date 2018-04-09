import numpy as np

import tensorflow as tf


LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn


class Decoder():

    def __init__(self, encoded_size, n_clusters):
        self.encoded_size = encoded_size

        self.n_clusters = n_clusters

        self.initializer = tf.contrib.layers.xavier_initializer()

    def decode(self, output_attender, contexts_mask):
        with tf.variable_scope("decoder"):

            output_attender_shape = tf.shape(output_attender)

            def get_logits(k):

                Wr = tf.get_variable('Wr' + str(k), [4 * self.encoded_size, 2 * self.encoded_size], dtype=tf.float32,
                                     initializer=self.initializer
                                     )
                Wh = tf.get_variable('Wh' + str(k), [4 * self.encoded_size, 2 * self.encoded_size], dtype=tf.float32,
                                     initializer=self.initializer
                                     )
                Wf = tf.get_variable('Wf' + str(k), [2 * self.encoded_size, 1], dtype=tf.float32,
                                     initializer=self.initializer
                                     )
                br = tf.get_variable('br' + str(k), [2 * self.encoded_size], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
                bf = tf.get_variable('bf' + str(k), [1, ], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())

                wr_e = tf.tile(
                    tf.expand_dims(Wr, axis=[0]),
                    [output_attender_shape[0], 1, 1]
                )
                s_f = tf.tanh(tf.matmul(output_attender, wr_e) + br)

                # f = tf.nn.dropout(f, keep_prob=keep_prob)

                wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [
                    output_attender_shape[0], 1, 1])

                with tf.name_scope('starter_score'):
                    s_score = tf.squeeze(tf.matmul(s_f, wf_e) + bf, axis=[2])

                with tf.name_scope('starter_prob'):
                    s_prob = tf.nn.softmax(s_score)
                    s_prob = tf.multiply(s_prob, contexts_mask)

                Hr_attend = tf.reduce_sum(tf.multiply(
                    output_attender, tf.expand_dims(s_prob, axis=[2])), axis=1)
                e_f = tf.tanh(tf.matmul(output_attender, wr_e) +
                              tf.expand_dims(
                                  tf.matmul(Hr_attend, Wh), axis=[1])
                              + br)

                with tf.name_scope('end_score'):
                    e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])

                with tf.name_scope('end_prob'):
                    e_prob = tf.nn.softmax(e_score)
                    e_prob = tf.multiply(e_prob, contexts_mask)

                return s_score, e_score

            logits = tf.stack(
                [get_logits(_k) for _k in range(self.n_clusters)],
                axis=0
            )

            return logits
