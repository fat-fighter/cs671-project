import os
import numpy as np


class squad_dataset(object):
    def __init__(self, question_file, context_file, answer_file, label_file, root="", batch_size=1, split=True):
        self.question_file = root + question_file
        self.context_file = root + context_file
        self.answer_file = root + answer_file
        self.label_file = root + label_file

        self.batch_size = batch_size

        self.length = None
        self.split = split

    def __iter_file(self, filename):
        with open(filename) as f:
            for line in f:
                if self.split:
                    line = line.strip().split(" ")
                    line = map(lambda tok: int(tok), line)
                yield line

    def __iter__(self):
        question_file_iter = self.__iter_file(self.question_file)
        context_file_iter = self.__iter_file(self.context_file)
        answer_file_iter = self.__iter_file(self.answer_file)
        label_file_iter = self.__iter_file(self.label_file)

        batch = []
        for question, context, answer, label in zip(question_file_iter, context_file_iter, answer_file_iter, label_file_iter):
            batch.append((question, context, answer, label))
            if len(batch) == self.batch_size:
                yield np.array(batch)
                batch = []
        yield np.array(batch)

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def pad_sequences(sequences, max_length):
    sequences_padded, sequences_length = [], []

    for sequence in sequences:
        sequences_length.append(min(len(sequence), max_length))

        sequence = list(sequence)
        sequence = sequence[:max_length] + [0] * \
            max(max_length - len(sequence), 0)
        sequences_padded.append(sequence)

    return np.array(sequences_padded), np.array(sequences_length)


def masks(lengths, max_length):
    masks = []
    for length in lengths:
        masks.append(
            [1.0] * length + [0.0] * max(max_length - length, 0)
        )
    return np.array(masks)


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
                 a modified z-score (based on the median absolute deviation)
                 greater than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """

    if len(points.shape) == 1:
        points = points[:, None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_answers(s, e):
    def func(y1, y2):
        max_ans = -999999
        a_s, a_e = 0, 0
        num_classes = len(y1)
        for i in range(num_classes):
            for j in range(15):
                if i+j >= num_classes:
                    break

                curr_a_s = y1[i]
                curr_a_e = y2[i+j]
                if (curr_a_e + curr_a_s) > max_ans:
                    max_ans = curr_a_e + curr_a_s
                    a_s = i
                    a_e = i+j

        return (a_s, a_e)

    answers = []
    for i in range(s.shape[0]):
        _a_s, _a_e = func(s[i], e[i])
        answers.append([_a_s, _a_e])

    return answers


def evaluate(graph, sess, val_data, msg):
    predicted, ground = graph.predict(sess, val_data, msg)

    em_1, em_2 = np.mean(predicted == ground, axis=0)
    em = np.mean(np.sum(predicted == ground, axis=1) / 2)

    common_tokens = np.maximum(
        np.min(
            [predicted[:, 1], ground[:, 1]], axis=0
        ) - np.max(
            [predicted[:, 0], ground[:, 0]], axis=0
        ) + 1,
        0
    )

    precision = 1.0 * common_tokens / (predicted[:, 1] - predicted[:, 0] + 1)
    recall = 1.0 * common_tokens / (ground[:, 1] - ground[:, 0] + 1)

    f1 = np.mean(
        2 * precision * recall /
        np.where(
            common_tokens == 0,
            1,
            precision + recall
        )
    )

    return (em, (em_1, em_2)), f1
