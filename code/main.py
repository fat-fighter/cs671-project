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

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams = mpl.rc_params_from_file("includes/matplotlibrc")

root_dir = os.getcwd()

words_embedding = np.load(config.embed_path)["glove"]

sess = tf.Session()

encoder = Encoder(
    config.encoding_size,
    config.dropout_keep_prob
)
decoder = Decoder(
    config.encoding_size,
    config.n_clusters,
    config.dropout_keep_prob
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


losses = []
if os.path.exists(config.loss_path):
    losses = list(np.load(config.loss_path))

scores = []
if os.path.exists(config.scores_path):
    scores = list(np.load(config.scores_path))

best_em = np.max([score[0][1] for score in scores]) or 0

if not init:
    scores.append(
        evaluate(graph, sess, val_data, "evaluating ... epoch: 0")
    )
    print_score(0, scores[-1])
else:
    score = evaluate(graph, sess, val_data, "evaluating ... epoch: 0")
    print_score(0, score)

for epoch in range(config.num_epochs)[:1]:

    losses.append(graph.run_epoch(
        train_data, epoch, sess, max_batch_epochs=-1)
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
