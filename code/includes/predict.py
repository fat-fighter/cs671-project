import io
import os
import json
import nltk
from includes import config

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from graph import Graph

from preprocessing.squad_processes import data_from_json, maybe_download, squad_base_url, invert_map, tokenize, token_idx_map


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']

            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, 2))
                               for w in context_tokens]
                qustion_ids = [str(vocab.get(w, 2))
                               for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    maybe_download(squad_base_url, dev_filename, prefix)
    nltk.download("punkt")

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(
        dev_data, 'dev', vocab)

    def normalize(dat):
        return map(lambda tok: map(int, tok.split()), dat)

    context_data = normalize(context_data)
    question_data = normalize(question_data)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, uuid_data, rev_vocab):
    answers = {}

    q, c, a = dataset
    num_points = len(a)
    sample_size = 1000

    answers_canonical = []
    num_iters = (num_points - 1) / sample_size + 1

    print num_iters
    print num_points
    print sample_size

    for i in xrange(num_iters):
        curr_slice_st = i * sample_size
        curr_slice_en = min((i + 1) * sample_size, num_points)

        slice_sz = curr_slice_en - curr_slice_st

        q_curr = q[curr_slice_st:curr_slice_en]
        c_curr = c[curr_slice_st:curr_slice_en]
        a_curr = a[curr_slice_st:curr_slice_en]

        s, e = model.answer(sess, [q_curr, c_curr, a_curr])

        for j in xrange(slice_sz):
            st_idx = s[j]
            en_idx = e[j]
            curr_context = c[curr_slice_st+j]
            curr_uuid = uuid_data[curr_slice_st+j]

            curr_ans = ""
            for idx in xrange(st_idx, en_idx+1):
                curr_tok = curr_context[idx]
                curr_ans += " %s" % (rev_vocab[curr_tok])

            answers[curr_uuid] = curr_ans
            answers_canonical.append((s, e))

    return answers, answers_canonical


def run_func():
    vocab, rev_vocab = initialize_vocab(config.vocab_path)

    dev_path = "data/squad/train-v1.1.json"
    dev_dirname = os.path.dirname(os.path.abspath(dev_path))
    dev_filename = os.path.basename(dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(
        dev_dirname, dev_filename, vocab)

    ques_len = len(question_data)
    answers = [[0, 0] for _ in xrange(ques_len)]

    dataset = [question_data, context_data, answers]

    words_embedding = np.load(config.embed_path)["glove"]

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

    with tf.Session() as sess:
        graph.init_model(sess)
        answers, _ = generate_answers(
            sess, graph, dataset, question_uuid_data, rev_vocab)

        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(
                unicode(
                    json.dumps(answers, ensure_ascii=False)
                )
            )


# if __name__ == "__main__":

#     run_func()
