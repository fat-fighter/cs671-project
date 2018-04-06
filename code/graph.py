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
        self.spans = tf.placeholder(
            tf.int32, shape=[None, 2]
        )
        self.labels = tf.placeholder(
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
            self.spans
        )
        self.logits = self.predictions

        self.loss = tf.reduce_mean(
            CrossEntropy(
                logits=self.logits[0], labels=self.spans[:, 0]
            ) +
            CrossEntropy(
                logits=self.logits[1], labels=self.spans[:, 1]
            )
        )

        adam_optimizer = tf.train.AdamOptimizer()
        grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

        self.gradients = zip(grads, vars)

        self.train_step = adam_optimizer.apply_gradients(self.gradients)

    def run_epoch(self, train_dataset, epoch, sess, max_batch_epochs=-1):
        print_dict = {"loss": "inf"}

        with tqdm(train_dataset, postfix=print_dict) as pbar:
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
                        self.spans: np.array([np.array(el[2]) for el in batch]),
                        self.dropout: config.train_dropout_val
                    }
                )
                print_dict["loss"] = "%.3f" % loss
                pbar.set_postfix(print_dict)

                if i == max_batch_epochs:
                    return
