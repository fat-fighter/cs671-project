import os
import sys
import numpy as np
from tqdm import tqdm

from includes import config
from includes.utils import squad_dataset, evaluate

from graph import Graph
from encoder import Encoder
from decoder import Decoder

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams = mpl.rc_params_from_file("includes/matplotlibrc")

root_dir = os.getcwd()

words_embedding = np.load(config.embed_path)["glove"]

sess = tf.Session()

encoder = Encoder(
    config.encoding_size
)
decoder = Decoder(
    config.encoding_size,
    config.n_clusters
)

graph = Graph(
    words_embedding,
    encoder,
    decoder
)

init = graph.init_model(sess)

train_data = squad_dataset(
    config.questions_train,
    config.contexts_train,
    config.answers_train,
    config.labels_train,
    root=root_dir + "/",
    batch_size=config.batch_size
)

val_data = squad_dataset(
    config.questions_val,
    config.contexts_val,
    config.answers_val,
    config.labels_val,
    root=root_dir + "/",
    batch_size=config.val_batch_size
)


def print_score(epoch, score):
    print "\nepoch: %d, f1: %.4f, em: %.4f, em@1: %.4f, em@2: %.4f\n" % (
        epoch, score[1], score[0][0], score[0][1][0], score[0][1][1]
    )


w1 = w2 = 1.0

losses = []
if os.path.exists(config.loss_path):
    losses = list(np.load(config.loss_path))

best_em = 0
scores = []
if os.path.exists(config.scores_path):
    scores = list(np.load(config.scores_path))

    best_em = np.max([score[0][1] for score in scores])

if not init:
    losses = []
    scores = []

    scores.append(
        evaluate(graph, sess, val_data, "evaluating ... epoch: 0")
    )

    best_em = scores[-1][0][1]

print_score(0, scores[-1])

for epoch in range(config.num_epochs):
    w1 = 1.0 / scores[-1][0][1][0]
    w2 = 1.0 / scores[-1][0][1][1]

    w = w1 + w2

    w1 = 2 * w1 / w
    w2 = 2 * w2 / w

    losses.append(graph.run_epoch(
        train_data, epoch, sess, max_batch_epochs=-1, w1=w1, w2=w2)
    )

    scores.append(
        evaluate(graph, sess, val_data,
                 "evaluating ... epoch: %d" % (epoch + 1))
    )
    print_score(epoch + 1, scores[-1])

    if scores[-1][0][0] >= best_em:
        graph.save_model(sess)
        best_em = scores[-1][0][0]

        np.save("data/plots/loss.npy", np.array(losses))
        np.save("data/plots/scores.npy", np.array(scores))
