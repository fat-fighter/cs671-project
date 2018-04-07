import os
import sys
import numpy as np
from tqdm import tqdm

from includes import config
from includes.utils import squad_dataset, pad_sequences
from includes.evaluate import evaluate_model, test, get_answers

from graph import Graph

import tensorflow as tf


root_dir = os.getcwd()


words_embedding = np.load(config.embed_path)["glove"]


with tf.Session() as sess:

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
        batch_size=500
    )

    graph = Graph(
        config.hidden_state_size,
        config.hidden_state_size,
        words_embedding,
        config.n_clusters
    )

    graph.init_model(sess)

    scores = []
    best_em = 0

    for epoch in range(config.num_epochs):

        graph.run_epoch(train_data, epoch, sess, max_batch_epochs=-1)

        scores.append(evaluate_model(graph, sess, val_data))
        print "Epoch %d EM Score: %.4f" % (
            epoch + 1, scores[-1]
        )

        if scores[-1] >= best_em:
            graph.save_model(sess)
            best_em = scores[-1]
