# CLUSTERING

min_df = 2
max_df = 1.0

max_tfidf_vocab = 300

kmeans_n_init = 10
kmeans_max_iter = 10000

max_question_len = 10

clustering = "question"
n_clusters = 5 if clustering == "context" else 6


# TRAINING

num_epochs = 15
batch_size = 32

max_gradient_norm = -1

embedding_size = 300
hidden_state_size = 150

data_dir = "data/squad/"
train_dir = "model"

vocab_path = data_dir + "/vocab.dat"
embed_path = data_dir + "/glove.npz"

train_dropout_val = 1.0

train_embeddings = False


def get_paths(mode):
    questions = data_dir + "/%s.ids.questions" % mode
    contexts = data_dir + "/%s.ids.contexts" % mode
    answers = data_dir + "/%s.spans" % mode
    labels = data_dir + "/%s.labels" % mode

    return questions, contexts, answers, labels


questions_train, contexts_train, answers_train, labels_train = \
    get_paths("train")

questions_val, contexts_val, answers_val, labels_val = \
    get_paths("val")
