import os
import sys
import numpy as np
from tqdm import tqdm

from includes import config
from includes.utils import squad_dataset, pad_sequences
from includes.evaluate import evaluate_model, test, get_answers

from attention_wrapper import BahdanauAttention, AttentionWrapper

import tensorflow as tf


root_dir = os.getcwd()

LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn

CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits


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


class Graph():
    def __init__(self, encoded_size, match_encoded_size, embeddings, n_clusters):
        self.encoded_size = encoded_size
        self.match_encoded_size = match_encoded_size

        self.encoder = Encoder(self.encoded_size)
        self.decoder = Decoder(
            self.match_encoded_size,
            self.encoded_size
        )

        self.embeddings = embeddings

        self.n_clusters = n_clusters

        self.init_placeholders()
        self.init_variables()
        self.init_nodes()

    def init_placeholders(self):
        self.question_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )
        self.context_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )
        self.questions_length = tf.placeholder(
            tf.int32, shape=[None]
        )
        self.passages_length = tf.placeholder(
            tf.int32, shape=[None]
        )
        self.labels = tf.placeholder(
            tf.int32, shape=[None, 2]
        )
        self.clusters = tf.placeholder(
            tf.int32, shape=[None, self.n_clusters]
        )
        self.dropout = tf.placeholder(
            tf.float32, shape=[]
        )

    def init_variables(self):
        word_embeddings = tf.Variable(
            self.embeddings, dtype=tf.float32, trainable=config.train_embeddings
        )
        questions_embedding = tf.nn.embedding_lookup(
            word_embeddings,
            self.question_ids
        )
        passages_embedding = tf.nn.embedding_lookup(
            word_embeddings,
            self.context_ids
        )

        self.questions = tf.nn.dropout(questions_embedding, self.dropout)
        self.passages = tf.nn.dropout(passages_embedding, self.dropout)

    def init_nodes(self):
        self.encoded_questions, \
            self.questions_representation, \
            self.encoded_passages, \
            self.passages_representation = self.encoder.encode(
                (self.questions, self.passages),
                (self.questions_length, self.passages_length)
            )

        self.predictions = self.decoder.predict(
            [self.encoded_questions, self.encoded_passages],
            [self.questions_length, self.passages_length],
            self.questions_representation,
            self.labels
        )
        self.logits = self.predictions

        self.loss = tf.reduce_mean(
            CrossEntropy(
                logits=self.logits[0], labels=self.labels[:, 0]
            ) +
            CrossEntropy(
                logits=self.logits[1], labels=self.labels[:, 1]
            )
        )

        adam_optimizer = tf.train.AdamOptimizer()
        grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

        self.gradients = zip(grads, vars)

        self.train_step = adam_optimizer.apply_gradients(self.gradients)

    def run_epoch(self, train_dataset, epoch, sess, cluster, max_batch_epochs=-1):
        print_dict = {"loss": "inf"}

        one_hot_clusters = np.zeros(self.n_clusters)
        one_hot_clusters[cluster] = 1

        with tqdm(train_data, postfix=print_dict) as pbar:
            pbar.set_description("Epoch %d" % (epoch + 1))
            for i, batch in enumerate(pbar):
                padded_questions, questions_length = pad_sequences(
                    np.array(batch[:, 0]), 0)
                padded_passages, passages_length = pad_sequences(
                    np.array(batch[:, 1]), 0)

                loss, _ = sess.run(
                    [self.loss, self.train_step],
                    feed_dict={
                        self.question_ids: np.array(padded_questions),
                        self.context_ids: np.array(padded_passages),
                        self.questions_length: np.array(questions_length),
                        self.passages_length: np.array(passages_length),
                        self.labels: np.array([np.array(el[2]) for el in batch]),
                        self.dropout: config.train_dropout_val,
                        self.cluster: one_hot_clusters
                    }
                )
                print_dict["loss"] = "%.3f" % loss
                pbar.set_postfix(print_dict)

                if i == max_batch_epochs:
                    return


words_embedding = np.load(config.embed_path)["glove"]


with tf.Session() as sess:
    scores = []

    train_data = squad_dataset(
        question_train,
        context_train,
        answer_train,
        root=root_dir + "/",
        batch_size=config.batch_size
    )

    val_data = squad_dataset(
        question_val,
        context_val,
        answer_val,
        root=root_dir + "/",
        batch_size=1
    )

    graph = Graph(
        config.hidden_state_size,
        config.hidden_state_size,
        words_embedding,
        config.n_clusters
    )

        tf.global_variables_initializer().run(session=sess)

        print "\tcluster: %d, epoch: %d, em: %.3f\n" % (
            cluster, 0,
            evaluate_model(graph, sess, val_data) / float(val_data.__len__())
        )

        for epoch in range(config.num_epochs):
            graph.run_epoch(train_data, epoch, sess, max_batch_epochs=-1)

            scores[-1].append(evaluate_model(graph, sess, val_data))
            print "\tcluster: %d, epoch: %d, em: %.3f\n" % (
                cluster, epoch +
                1, float(scores[-1][-1]) / float(val_data.__len__())
            )
