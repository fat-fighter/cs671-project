import tensorflow as tf


LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn
DynamicBiRNN = tf.nn.bidirectional_dynamic_rnn


class Encoder:

    def __init__(self, hidden_size, bi_directional_encoding=False):
        self.hidden_size = hidden_size
        self.bi_directional_encoding = bi_directional_encoding

    def encode(self, vectors, lengths):
        questions, contexts = vectors
        questions_length, contexts_length = lengths

        if self.bi_directional_encoding:
            with tf.variable_scope("encoder_question"):
                question_lstm_fw_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                question_lstm_bw_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                encoded_questions_tuple, (q_rep, _) = DynamicBiRNN(
                    question_lstm_fw_cell,
                    question_lstm_bw_cell,
                    questions,
                    sequence_length=questions_length,
                    dtype=tf.float32
                )
                encoded_questions = tf.concat(
                    encoded_questions_tuple, axis=2
                )

            with tf.variable_scope("encoder_context"):
                context_lstm_fw_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                context_lstm_bw_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                encoded_contexts_tuple, (p_rep, _) = DynamicBiRNN(
                    context_lstm_fw_cell,
                    context_lstm_bw_cell,
                    contexts,
                    sequence_length=contexts_length,
                    dtype=tf.float32
                )
                encoded_contexts = tf.concat(
                    encoded_contexts_tuple, axis=2
                )

        else:
            with tf.variable_scope("encoder_question"):
                question_lstm_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                encoded_questions, (q_rep, _) = DynamicRNN(
                    question_lstm_cell,
                    questions,
                    questions_length,
                    dtype=tf.float32
                )

            with tf.variable_scope("encoder_context"):
                context_lstm_cell = LSTMCell(
                    self.hidden_size,
                    state_is_tuple=True
                )
                encoded_contexts, (p_rep, _) = DynamicRNN(
                    context_lstm_cell,
                    contexts,
                    contexts_length,
                    dtype=tf.float32
                )

        return encoded_questions, q_rep, encoded_contexts, p_rep
