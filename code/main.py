import os
import sys
import numpy as np
from tqdm import tqdm

from includes import config
from includes.utils import squad_dataset, pad_sequences

from attention_wrapper import BahdanauAttention, AttentionWrapper

import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_vocab(vocab_path):
    if os.path.exists(vocab_path):
        
        with open(vocab_path, mode="rb") as f:
            vocab = dict([
                (line.strip(), index)
                for index, line in enumerate(f.readlines())
            ])
        
        return vocab
    
    else:
        raise IOError("File %s not found.", vocab_path)

root_dir = os.getcwd()

LSTMCell = tf.contrib.rnn.BasicLSTMCell
DynamicRNN = tf.nn.dynamic_rnn

CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits

class Encoder:
    """
    LSTM preprocessing  layer to encode the question
    and passage representations using a single layer
    of LSTM
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        
    def encode(self, vectors, lengths):
        """
        vectors  ::  tuple  ::  Word vectors of Question and Passage
        lengths  ::  tuple  ::  Word vectors of Question and Passage
        """
        questions, passages = vectors
        questions_length, passages_length = lengths
        
        question_lstm_cell = LSTMCell(
            self.hidden_size,
            state_is_tuple=True,
            name="question_lstm_cell"
        )
        encoded_questions, (q_rep, _) = DynamicRNN(
            question_lstm_cell,
            questions,
            questions_length,
            dtype=tf.float32
        )

        passage_lstm_cell = LSTMCell(
            self.hidden_size,
            state_is_tuple=True,
            name="passage_lstm_cell"
        )
        encoded_passages, (p_rep, _) =  DynamicRNN(
            passage_lstm_cell,
            passages,
            passages_length,
            dtype=tf.float32
        )
            
        return encoded_questions, q_rep, encoded_passages, p_rep

class MatchEncoder():
    """
    Match-LSTM layer to encode the question
    representation in order to get a hidden
    representation of the question and the passage
    """
    
    def __init__(self, hidden_size, encoded_size):
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
    
    def run_match_lstm(self, vectors, lengths):
        questions, passages = vectors
        questions_length, passages_length = lengths
        
        def attention_function(x, state):
            return tf.concat([x, state], axis=-1)
        
        attention_mechanism_match_lstm = BahdanauAttention(
            self.encoded_size,
            questions,
            memory_sequence_length=questions_length
        )
        
        cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_size, state_is_tuple=True
        )
        lstm_attender = AttentionWrapper(
            cell,
            attention_mechanism_match_lstm,
            output_attention=False,
            attention_input_fn=attention_function
        )

        reverse_encoded_passage = tf.reverse_sequence(passages, passages_length, batch_axis=0, seq_axis=1)

        output_attender_fw, _ = tf.nn.dynamic_rnn(
            lstm_attender, passages, dtype=tf.float32
        )
        output_attender_bw, _ = tf.nn.dynamic_rnn(
            lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope="rnn")

        output_attender_bw = tf.reverse_sequence(output_attender_bw, passages_length, batch_axis=0, seq_axis=1)
        
        output_attender = tf.concat(
            [output_attender_fw, output_attender_bw], axis=-1
        )
        return output_attender

    def run_answer_pointer(self, output_attender, lengths, labels):
        questions_length, passages_length = lengths
        labels = tf.unstack(labels, axis=1)

        def input_function(curr_input, passage):
            return passage
        
        query_depth_answer_ptr = output_attender.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attender"):
            attention_mechanism_answer_ptr = BahdanauAttention(
                query_depth_answer_ptr,
                output_attender,
                memory_sequence_length=passages_length
            )
            
            cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            answer_ptr_attender = AttentionWrapper(
                cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn=input_function)
            logits, _ = tf.nn.static_rnn(
                answer_ptr_attender, labels, dtype=tf.float32)

        return logits
    
    def predict(self, vectors, lengths, questions_representation, labels):
        output_attender = self.run_match_lstm(vectors, lengths)
        logits = self.run_answer_pointer(output_attender, lengths, labels)
        
        return logits
        

class Graph():
    def __init__(self, encoded_size, match_encoded_size, embeddings):
        self.encoded_size = encoded_size
        self.match_encoded_size = match_encoded_size
        
        self.encoder = Encoder(self.encoded_size)
        self.model = MatchEncoder(self.match_encoded_size, self.encoded_size)
        
        self.embeddings = embeddings
        
        self.init_placeholders()
        self.init_variables()
        self.init_nodes()
    
    def init_placeholders(self):
        self.question_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )
        self.passage_ids = tf.placeholder(
            tf.int32, shape=[None, None]
        )
        self.questions_length = tf.placeholder(
            tf.int32, shape=[None]
        )
        self.passages_length = tf.placeholder(
            tf.int32, shape=[None]
        )
        self.labels = tf.placeholder(
            tf.int32, shape=[None, 2]
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
            self.passage_ids
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

        self.logits = self.model.predict(
            [self.encoded_questions, self.encoded_passages],
            [self.questions_length, self.passages_length],
            self.questions_representation,
            self.labels
        )
        
        self.loss = tf.reduce_mean(
            CrossEntropy(
                logits=self.logits[0], labels=self.labels[:, 0]
            ) + \
            CrossEntropy(
                logits=self.logits[1], labels=self.labels[:, 1]
            )
        )
        
        adam_optimizer = tf.train.AdamOptimizer()
        grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

        self.gradients = zip(grads, vars)

        self.train_step = adam_optimizer.apply_gradients(self.gradients)
        
    def test(self, session, valid):
        q, c, a = valid

        # at test time we do not perform dropout.
        padded_questions, questions_length = pad_sequences(q, 0)
        padded_passages, passages_length = pad_sequences(c, 0)
        
        input_feed={
            self.question_ids: np.array(padded_questions),
            self.passage_ids: np.array(padded_passages),
            self.questions_length: np.array(questions_length),
            self.passages_length: np.array(passages_length),
            self.labels: np.array(a),
            self.dropout: config.train_dropout_val
        }

        output_feed = [self.logits]

        outputs = session.run(output_feed, input_feed)

        return outputs[0][0], outputs[0][1]
        
    def answer(self, session, dataset):
        yp, yp2 = self.test(session, dataset)
        
        def func(y1, y2):
            max_ans = -999999
            a_s, a_e= 0,0
            num_classes = len(y1)
            for i in xrange(num_classes):
                for j in xrange(15):
                    if i+j >= num_classes:
                        break

                    curr_a_s = y1[i];
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


    def evaluate_model(self, session, dataset):
        
        q, c, a = zip(*[_row[0] for _row in dataset])

        sample = len(dataset)
        a_s, a_o = self.answer(session, [q, c, a])
        answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample,-1])])
        gold_answers = np.array([a[0][2] for a in dataset])

        em_score = 0
        em_1 = 0
        em_2 = 0
        for i in xrange(sample):
            gold_s, gold_e = gold_answers[i]
            s, e = answers[i]
            if (s==gold_s): em_1 += 1.0
            if (e==gold_e): em_2 += 1.0
            if (s == gold_s and e == gold_e):
                em_score += 1.0

        em_1 /= float(len(answers))
        em_2 /= float(len(answers))
        print("\nExact match on 1st token: %5.4f | Exact match on 2nd token: %5.4f\n" %(em_1, em_2))

        em_score /= float(len(answers))

        return em_score

    def train(self, train_dataset, val_dataset):
        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            
            print_dict = {"loss": "inf"}
            for epoch in range(config.num_epochs):
                with tqdm(train_data, postfix=print_dict) as pbar:
                    pbar.set_description("Epoch %d" % (epoch + 1))

                    index = 0
                    for batch in pbar:
                        padded_questions, questions_length = pad_sequences(np.array(batch[:, 0]), 0)
                        padded_passages, passages_length = pad_sequences(np.array(batch[:, 1]), 0)

                        loss, _ = sess.run(
                            [self.loss, self.train_step],
                            feed_dict={
                                self.question_ids: np.array(padded_questions),
                                self.passage_ids: np.array(padded_passages),
                                self.questions_length: np.array(questions_length),
                                self.passages_length: np.array(passages_length),
                                self.labels: np.array([np.array(el[2]) for el in batch]),
                                self.dropout: config.train_dropout_val
                            }
                        )
                        print_dict["loss"] = "%.3f" % loss
                        pbar.set_postfix(print_dict)
                        if index == 5:
                            break
                        index += 1
                        
                em = self.evaluate_model(sess, val_dataset)
                print("\n#-----------Exact match on val set: %5.4f #-----------\n" %em)

words_embedding = np.load(config.embed_path)["glove"]

train_data = squad_dataset(
    config.question_train,
    config.context_train,
    config.answer_train,
    root=root_dir + "/",
    batch_size=config.batch_size
)

val_data = squad_dataset(
    config.question_val,
    config.context_val,
    config.answer_val,
    root=root_dir + "/",
    batch_size=1
)

graph = Graph(config.hidden_state_size, config.hidden_state_size, words_embedding)
graph.train(train_data, val_data)
