import numpy as np

import tensorflow as tf

from attention_wrapper import BahdanauAttention, AttentionWrapper


LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn


class Decoder():

    def __init__(self, hidden_size, encoded_size):
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size

    def match_lstm(self, vectors, lengths):
        with tf.variable_scope("match_lstm"):
            questions, passages = vectors
            questions_length, passages_length = lengths

            def attention_function(x, state):
                return tf.concat([x, state], axis=-1)

            attention_mechanism_match_lstm = BahdanauAttention(
                self.encoded_size,
                questions,
                memory_sequence_length=questions_length
            )

            cell = LSTMCell(
                self.hidden_size,
                state_is_tuple=True
            )
            lstm_attender = AttentionWrapper(
                cell,
                attention_mechanism_match_lstm,
                output_attention=False,
                attention_input_fn=attention_function
            )

            output_attender_fw, _ = DynamicRNN(
                lstm_attender, passages, dtype=tf.float32
            )

            reverse_encoded_context = tf.reverse_sequence(
                passages, passages_length, batch_axis=0, seq_axis=1
            )

            output_attender_bw, _ = DynamicRNN(
                lstm_attender, reverse_encoded_context, dtype=tf.float32, scope="rnn"
            )
            output_attender_bw = tf.reverse_sequence(
                output_attender_bw, passages_length, batch_axis=0, seq_axis=1
            )

            output_attender = tf.concat(
                [output_attender_fw, output_attender_bw], axis=-1
            )

        return output_attender

    def answer_pointer(self, output_attender, lengths, labels):
        with tf.variable_scope("answer_pointer"):
            _, passages_length = lengths
            labels = tf.unstack(labels, axis=1)

            def input_function(curr_input, context):
                return context

            query_depth_answer_ptr = 2 * self.hidden_size

            attention_mechanism_answer_ptr = BahdanauAttention(
                query_depth_answer_ptr,
                output_attender,
                memory_sequence_length=passages_length
            )

            cell_answer_ptr = LSTMCell(
                self.hidden_size,
                state_is_tuple=True
            )
            answer_ptr_attender = AttentionWrapper(
                cell_answer_ptr,
                attention_mechanism_answer_ptr,
                cell_input_fn=input_function
            )

            logits, _ = tf.nn.static_rnn(
                answer_ptr_attender, labels, dtype=tf.float32
            )

        return logits

    def predict(self, vectors, lengths, questions_representation, labels):
        output_attender = self.match_lstm(vectors, lengths)
        logits = self.answer_pointer(output_attender, lengths, labels)

        return logits
