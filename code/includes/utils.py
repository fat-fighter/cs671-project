class squad_dataset(object):
    def __init__(self, question_file, context_file, answer_file, root=""):
        self.question_file = root + question_file
        self.context_file = root + context_file
        self.answer_file = root + answer_file

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

        for question, context, answer in zip(question_file_iter, context_file_iter, answer_file_iter):
            yield (question, context, answer)

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
