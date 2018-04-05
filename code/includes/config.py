num_epochs = 5
batch_size = 32

max_gradient_norm = -1

embedding_size = 300
hidden_state_size = 150

data_dir = "data/squad"

vocab_path = data_dir + "/vocab.dat"
embed_path = data_dir + "/glove.trimmed.300.npz"

train_dropout_val = 1.0

train_embeddings = False


def get_paths(mode):
    question = data_dir + "/%s.ids.question" % mode
    context = data_dir + "/%s.ids.context" % mode
    answer = data_dir + "/%s.span" % mode

    return question, context, answer


question_train, context_train, answer_train = get_paths("train")
question_val, context_val, answer_val = get_paths("val")
