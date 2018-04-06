import numpy as np

from includes import config
from includes.utils import squad_dataset, pad_sequences


def test(graph, session, valid):
    q, c, a = valid

    # at test time we do not perform dropout.
    padded_questions, questions_length = pad_sequences(q, 0)
    padded_passages, passages_length = pad_sequences(c, 0)

    input_feed = {
        graph.question_ids: np.array(padded_questions),
        graph.context_ids: np.array(padded_passages),
        graph.questions_length: np.array(questions_length),
        graph.passages_length: np.array(passages_length),
        graph.labels: np.array(a),
        graph.dropout: config.train_dropout_val
    }

    output_feed = [graph.logits]

    outputs = session.run(output_feed, input_feed)

    return outputs[0][0], outputs[0][1]


def get_answers(graph, session, dataset):
    yp, yp2 = test(graph, session, dataset)

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


def evaluate_model(graph, session, dataset):

    q, c, a = zip(*[_row[0] for _row in dataset])

    sample = len(dataset)
    a_s, a_o = get_answers(graph, session, [q, c, a])
    answers = np.hstack(
        [a_s.reshape([sample, -1]), a_o.reshape([sample, -1])])
    gold_answers = np.array([a[0][2] for a in dataset])

    em_score = 0
    em_1 = 0
    em_2 = 0
    for i in xrange(sample):
        gold_s, gold_e = gold_answers[i]
        s, e = answers[i]
        if (s == gold_s):
            em_1 += 1.0
        if (e == gold_e):
            em_2 += 1.0
        if (s == gold_s and e == gold_e):
            em_score += 1.0

    em_1 /= float(len(answers))
    em_2 /= float(len(answers))
    print("\nExact match on 1st token: %5.4f | Exact match on 2nd token: %5.4f\n" % (
        em_1, em_2))

    return em_score
