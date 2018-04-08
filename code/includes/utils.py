import numpy as np


class squad_dataset(object):
    def __init__(self, question_file, context_file, answer_file, label_file, root="", batch_size=1):
        self.question_file = root + question_file
        self.context_file = root + context_file
        self.answer_file = root + answer_file
        self.label_file = root + label_file

        self.batch_size = batch_size

        self.length = None

    def __iter_file(self, filename):
        with open(filename) as f:
            for line in f:
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
