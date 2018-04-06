import tensorflow as tf


LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn


class Encoder:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def encode(self, vectors, lengths):
        questions, passages = vectors
        questions_length, passages_length = lengths

        question_lstm_cell = LSTMCell(
            self.hidden_size,
            state_is_tuple=True,
            name="question_lstm_cell"
        )
        encoded_questions, (q_rep, _) = DynamicRNN(
            question_lstm_cell,
            questions,
            questions_length,
            dtype=tf.float32
        )

        context_lstm_cell = LSTMCell(
            self.hidden_size,
            state_is_tuple=True,
            name="context_lstm_cell"
        )
        encoded_passages, (p_rep, _) = DynamicRNN(
            context_lstm_cell,
            passages,
            passages_length,
            dtype=tf.float32
        )

        return encoded_questions, q_rep, encoded_passages, p_rep
