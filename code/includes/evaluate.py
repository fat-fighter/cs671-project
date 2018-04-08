import numpy as np

from includes import config
from includes.utils import squad_dataset, pad_sequences


def test(graph, sess, valid):
    q, c, a, l = valid

    labels = np.zeros(
        (len(l), config.n_clusters), dtype=np.float32
    )
    if config.clustering:
        for i, _l in enumerate(l):
            labels[i, _l] = 1
    else:
        labels[:, 0] = 1

    padded_questions, questions_length = pad_sequences(q, 0)
    padded_passages, passages_length = pad_sequences(c, 0)

    input_feed = {
        graph.questions_ids: np.array(padded_questions),
        graph.contexts_ids: np.array(padded_passages),
        graph.questions_length: np.array(questions_length),
        graph.contexts_length: np.array(passages_length),
        graph.answers: np.array(a),
        graph.labels: labels,
        graph.dropout: 1
    }

    output_feed = [graph.logits]

    outputs = sess.run(output_feed, input_feed)

    return outputs[0][0], outputs[0][1]


def get_answers(graph, sess, dataset):
    yp, yp2 = test(graph, sess, dataset)

    def func(y1, y2):
        max_ans = -999999
        a_s, a_e = 0, 0
        num_classes = len(y1)
        for i in xrange(num_classes):
            for j in xrange(15):
                if i+j >= num_classes:
                    break

                curr_a_s = y1[i]
                curr_a_e = y2[i+j]
                if (curr_a_e+curr_a_s) > max_ans:
                    max_ans = curr_a_e + curr_a_s
                    a_s = i
                    a_e = i+j

        return (a_s, a_e)

    a_s, a_e = [], []
    for i in xrange(yp.shape[0]):
        _a_s, _a_e = func(yp[i], yp2[i])
        a_s.append(_a_s)
        a_e.append(_a_e)

    return (np.array(a_s), np.array(a_e))


def evaluate_model(graph, sess, dataset):

    em_score = 0
    length = 0
    em_1 = 0
    em_2 = 0

    for batch in dataset:
        q, c, a, l = zip(*batch)

        sample = len(q)
        a_s, a_o = get_answers(graph, sess, [q, c, a, l])

        answers = np.hstack(
            [a_s.reshape([sample, -1]), a_o.reshape([sample, -1])]
        )
        gold_answers = np.array([_a for _a in a])

        match = (answers == gold_answers)
        _em_1, _em_2 = np.sum(match, axis=0)

        em_1 += _em_1
        em_2 += _em_2

        em_score += np.sum(np.sum(match, axis=1) == 2, axis=0)
        length += sample

    em_1 /= float(length)
    em_2 /= float(length)

    em_score /= float(length)

    return em_score, em_1, em_2
