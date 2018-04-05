import numpy as np


class squad_dataset(object):
	def __init__(self, question_file, context_file, answer_file, root="", batch_size=1):
		self.question_file = root + question_file
		self.context_file = root + context_file
		self.answer_file = root + answer_file

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
		answer_file_iter = self.__iter_file(self.answer_file)
		context_file_iter = self.__iter_file(self.context_file)

		batch = []
		for question, context, answer in zip(question_file_iter, context_file_iter, answer_file_iter):
			batch.append((question, context, answer))
			if len(batch) == self.batch_size:
				yield np.array(batch)
				batch = []

	def __len__(self):
		"""
		Iterates once over the corpus to set and store length
		"""
		if self.length is None:
			self.length = 0
			for _ in self:
				self.length += 1

		return self.length


def pad_sequences(sequences, token):
	"""
	Args:
		sequences	:: list		:: a generator of lists or tuples
		token		:: numeric	:: the number to pad the sequence with
	Returns:
		a list of list where each sublist has the same length
	"""

	max_length = max([len(x) for x in sequences])

	sequences_padded, sequences_length = [], []

	for sequence in sequences:
		sequence = list(sequence)
		sequence = sequence[:max_length] + [token] * max(max_length - len(sequence), 0)
		sequences_padded += [sequence]
		sequences_length += [min(len(sequence), max_length)]

	return np.array(sequences_padded), np.array(sequences_length)
