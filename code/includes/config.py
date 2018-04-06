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

num_epochs = 5
batch_size = 32

max_gradient_norm = -1

embedding_size = 300
hidden_state_size = 150

data_dir = "data/squad/"

vocab_path = data_dir + "/vocab.dat"
embed_path = data_dir + "/glove.trimmed.300.npz"

train_dropout_val = 1.0

train_embeddings = False


def get_paths(mode, cluster=None):
    ddir = data_dir
    if cluster != None:
        ddir += "clusters/k" + str(cluster)

    question = ddir + "/%s.ids.question" % mode
    context = ddir + "/%s.ids.context" % mode
    answer = ddir + "/%s.span" % mode

    return question, context, answer


question_train, context_train, answer_train = get_paths("train")
question_val, context_val, answer_val = get_paths("val")
