import tensorflow as tf


LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn


class Encoder:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def encode(self, vectors, lengths):
        questions, contexts = vectors
        questions_length, contexts_length = lengths

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
