import progressbar
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import *
from neg_loss import *
from utils import *
import random
import torch
import torch.utils

CONTEXT_SIZE = 4  # window size = 2 * CONTEXT_SIZE + 1
EMBEDDING_DIM = 300
NUM_EPOCHS = 10
NEGATIVE_SAMPLING = True
TOP_WORDS_NUM = 5000
EVAL_SAMPLE_NUM = 20

filename = "sample_text.txt"
print("Parsing text and loading training data...")
textdata = TextData(filename, CONTEXT_SIZE, TOP_WORDS_NUM, model_type="skipgram", subsampling=False,
                    sampling_rate=0.001)

losses = []
if NEGATIVE_SAMPLING:
    loss_function = NEGLoss(textdata.get_ix_to_word(), textdata.get_word_count())
else:
    loss_function = nn.NLLLoss()
model = SkipGram(len(textdata.get_word_count()), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)

eval_sample = random.sample([_ for _ in range(min(TOP_WORDS_NUM, len(textdata.get_ix_to_word())))], EVAL_SAMPLE_NUM)
eval_sample = torch.tensor(eval_sample, dtype=torch.long)


def eval_model(topN=10):
    sample_words_embedding = model.embeddings(eval_sample)
    embeddings = model.embeddings.weight.data
    sample_words_score = torch.mm(sample_words_embedding, embeddings.t())
    ix_to_word = textdata.get_ix_to_word()
    for index, word_idx in enumerate(eval_sample):
        similar_word_idx = torch.argsort(sample_words_score[index])[:topN]
        similar_word = []
        for idx in similar_word_idx:
            similar_word.append(ix_to_word[idx.item()])
        print("word [{}] similar words are: {}".format(ix_to_word[word_idx.item()], similar_word))


batch_size = 20
encode_text_len = len(textdata.get_encode_data())
print("Starting training")
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    print("Beginning epoch %d" % epoch)
    iter_sum = int(encode_text_len / batch_size)
    for iter_i in range(1, iter_sum + 1):
        context_var, train_label = textdata.get_batch(batch_size)
        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, train_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if iter_i % 2 == 0:
            print("epoch {}, training iter {}/{}, average loss is {}".format(epoch, iter_i, iter_sum,
                                                                             total_loss / iter_i))
        if iter_i % 2 == 0:
            model.eval()
            eval_model(10)
            model.train()
    print("Epoch %d Loss: %.5f" % (epoch, total_loss / iter_sum))
    losses.append(total_loss)

# Visualize embeddings
if EMBEDDING_DIM == 2:
    indices = np.random.choice(np.arange(len(textdata.get_encode_data())), size=50, replace=False)
    word_to_ix = textdata.get_word_to_ix()
    for ind in indices:
        word = list(textdata.get_word_count().keys())[ind]
        input = torch.tensor(word_to_ix[word], dtype=torch.long)
        vec = model.embeddings(input).data[0]
        x, y = vec[0], vec[1]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.savefig("w2v.png")
