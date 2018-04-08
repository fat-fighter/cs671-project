import tensorflow as tf

from mlstm import MatchLSTMCell
LSTMCell = tf.contrib.rnn.BasicLSTMCell

DynamicRNN = tf.nn.dynamic_rnn
DynamicBiRNN = tf.nn.bidirectional_dynamic_rnn


class Encoder:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def encode(self, vectors, lengths, questions_mask):
        questions, contexts = vectors
        questions_length, contexts_length = lengths

        with tf.variable_scope("encoder_question"):
            question_lstm_fw_cell = LSTMCell(
                self.hidden_size,
                state_is_tuple=True
            )
            question_lstm_bw_cell = LSTMCell(
                self.hidden_size,
                state_is_tuple=True
            )
            encoded_questions_tuple, _ = DynamicBiRNN(
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
            encoded_contexts_tuple, _ = DynamicBiRNN(
                context_lstm_fw_cell,
                context_lstm_bw_cell,
                contexts,
                sequence_length=contexts_length,
                dtype=tf.float32
            )
            encoded_contexts = tf.concat(
                encoded_contexts_tuple, axis=2
            )

        with tf.variable_scope("match_encoding"):
            input_size = 2 * self.hidden_size

            matchlstm_fw_cell = MatchLSTMCell(
                input_size,
                input_size,
                encoded_questions,
                questions_mask
            )
            matchlstm_bw_cell = MatchLSTMCell(
                input_size,
                input_size,
                encoded_questions,
                questions_mask
            )

            output_attender_tuple, _ = tf.nn.bidirectional_dynamic_rnn(
                matchlstm_fw_cell,
                matchlstm_bw_cell,
                encoded_contexts,
                sequence_length=contexts_length,
                dtype=tf.float32
            )

            output_attender = tf.concat(output_attender_tuple, axis=2)

        return output_attender
