from includes import config
from includes.utils import pad_sequences

from encoder import Encoder
from decoder import Decoder

import numpy as np
from tqdm import tqdm
import tensorflow as tf


CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits


class Graph():
    def __init__(self, encoded_size, match_encoded_size, embeddings, n_clusters):
        self.encoded_size = encoded_size
        self.match_encoded_size = match_encoded_size

        self.encoder = Encoder(self.encoded_size)
        self.decoder = Decoder(
            self.match_encoded_size,
            self.encoded_size,
            n_clusters
        )

        self.embeddings = embeddings

        self.n_clusters = n_clusters

        self.init_placeholders()
        self.init_variables()
        self.init_nodes()

    def init_placeholders(self):
        self.questions_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )
        self.contexts_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )

        self.questions_length = tf.placeholder(
            tf.int32, shape=[None]
        )
        self.contexts_length = tf.placeholder(
            tf.int32, shape=[None]
        )

        self.answers = tf.placeholder(
            tf.int32, shape=[None, 2]
        )

        self.labels = tf.placeholder(
            tf.float32, shape=[None, self.n_clusters]
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
            self.questions_ids
        )
        contexts_embedding = tf.nn.embedding_lookup(
            word_embeddings,
            self.contexts_ids
        )

        self.questions = tf.nn.dropout(questions_embedding, self.dropout)
        self.contexts = tf.nn.dropout(contexts_embedding, self.dropout)

    def init_nodes(self):
        self.encoded_questions, \
            self.questions_representation, \
            self.encoded_contexts, \
            self.contexts_representation = self.encoder.encode(
                (self.questions, self.contexts),
                (self.questions_length, self.contexts_length)
            )

        self.predictions = self.decoder.predict(
            [self.encoded_questions, self.encoded_contexts],
            [self.questions_length, self.contexts_length],
            self.questions_representation,
            self.answers
        )

        labels_shape = tf.shape(self.labels)
        predictions_shape = tf.shape(self.predictions)

        self.labels_broadcasted = tf.tile(
            tf.reshape(
                tf.transpose(self.labels), [6, 1, labels_shape[0], 1]
            ), tf.stack(
                [1, 2, 1, predictions_shape[3]]
            )
        )

        self.logits = tf.reduce_sum(
            tf.multiply(
                self.labels_broadcasted, self.predictions
            ), axis=0
        )

        self.loss = tf.reduce_mean(
            CrossEntropy(
                logits=self.logits[0], labels=self.answers[:, 0]
            ) +
            CrossEntropy(
                logits=self.logits[1], labels=self.answers[:, 1]
            )
        )

        adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.train_step = adam_optimizer.minimize(self.loss)

        # grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))
        # self.gradients = zip(grads, vars)
        # self.train_step = adam_optimizer.apply_gradients(self.gradients)

    def run_epoch(self, train_dataset, epoch, sess, max_batch_epochs=-1):
        print_dict = {"loss": "inf"}

        with tqdm(train_dataset, postfix=print_dict) as pbar:
            pbar.set_description("Epoch %d" % (epoch + 1))
            for i, batch in enumerate(pbar):
                if i == max_batch_epochs:
                    return

                questions_padded, questions_length = pad_sequences(
                    np.array(batch[:, 0]), 0
                )
                contexts_padded, contexts_length = pad_sequences(
                    np.array(batch[:, 1]), 0
                )

                labels = np.zeros(
                    (len(batch), self.n_clusters), dtype=np.float32
                )
                for i, el in enumerate(batch):
                    labels[i, el[3]] = 1

                loss, _ = sess.run(
                    [self.loss, self.train_step],
                    feed_dict={
                        self.questions_ids: np.array(questions_padded),
                        self.questions_length: np.array(questions_length),
                        self.contexts_ids: np.array(contexts_padded),
                        self.contexts_length: np.array(contexts_length),
                        self.answers: np.array([np.array(el[2]) for el in batch]),
                        self.labels: labels,
                        self.dropout: config.train_dropout_val
                    }
                )
                print_dict["loss"] = "%.3f" % loss
                pbar.set_postfix(print_dict)
