import numpy as np
import torch
import torch.utils
import collections

from nltk.tokenize import word_tokenize


class TextData:
    def __init__(self, filename, context_size, n_words=5000, model_type="skipgram", subsampling=False,
                 sampling_rate=0.001):
        self.filename = filename
        self.context_size = context_size
        self.n_words = n_words
        self.model_type = model_type
        self.subsampling = subsampling
        self.sampling_rate = sampling_rate
        self.global_index = 0
        self.__load_data__()

    def __load_data__(self):
        with open(self.filename, "rb") as file:
            processed_text = word_tokenize(file.read().decode("utf-8").strip())
            processed_text, vocab, word_to_ix, ix_to_word = self.__gather_word_freqs__(processed_text)
            word_idxs = []
            for word in processed_text:
                word_idx = word_to_ix.get(word, 0)
                word_idxs.append(word_idx)
            self.raw_data = processed_text
            self.encode_data = word_idxs
            self.word_count = vocab
            self.word_to_ix = word_to_ix
            self.ix_to_word = ix_to_word

    def get_raw_data(self):
        return self.raw_data

    def get_encode_data(self):
        return self.encode_data

    def get_word_count(self):
        return self.word_count

    def get_word_to_ix(self):
        return self.word_to_ix

    def get_ix_to_word(self):
        return self.ix_to_word

    def __gather_word_freqs__(self, split_text):
        count = [['UNK', -1]]
        count.extend(collections.Counter(split_text).most_common(self.n_words - 1))
        vocab = dict()
        ix_to_word = dict()
        word_to_ix = dict()
        total = 0.0
        for word, num in count:
            ix_to_word[len(word_to_ix)] = word
            word_to_ix[word] = len(word_to_ix)
            vocab[word] = 0
            word_idx = word_to_ix.get(word, 0)
            if word_idx != 0:
                vocab[word] = num
        for word in split_text:
            word_idx = word_to_ix.get(word, 0)
            if word_idx == 0:
                vocab[ix_to_word[word_idx]] += 1
        for word in vocab:
            total += vocab[word]
        if self.subsampling:
            for i, word in enumerate(split_text):
                word_idx = word_to_ix.get(word, 0)
                val = np.sqrt(self.sampling_rate * total / vocab[ix_to_word[word_idx]])
                prob = val * (1 + val)
                sampling = np.random.sample()
                if sampling <= prob:
                    del [split_text[i]]
                    i -= 1
                    vocab[ix_to_word[word_idx]] -= 1
        return split_text, vocab, word_to_ix, ix_to_word

    def get_batch(self, batch_size):
        batch = np.ndarray(shape=(batch_size * 2 * self.context_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size * 2 * self.context_size), dtype=np.int32)
        span = 2 * self.context_size + 1  # [ context_size target context_size ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if self.global_index + span > len(self.encode_data):
            self.global_index = 0
        buffer.extend(self.encode_data[self.global_index:self.global_index + span])
        self.global_index += span
        for i in range(batch_size):
            context_words = [w for w in range(span) if w != self.context_size]
            # words_to_use = random.sample(context_words, num_skips)
            if self.model_type == "skipgram":
                for j, context in enumerate(context_words):
                    batch[i * 2 * self.context_size + j] = buffer[self.context_size]
                    labels[i * 2 * self.context_size + j] = buffer[context]
            elif self.model_type == "cbow":
                for j, context in enumerate(context_words):
                    batch[i * 2 * self.context_size + j] = buffer[context]
                    labels[i * 2 * self.context_size + j] = buffer[self.context_size]
            else:
                raise ValueError("Inappropriate argument value for model_type - either `skipgram` or `cbow`.")
                break
            if self.global_index == len(self.encode_data):
                buffer.extend(self.encode_data[0:span])
                self.global_index = span
            else:
                buffer.append(self.encode_data[self.global_index])
                self.global_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.global_index = (self.global_index + len(self.encode_data) - span) % len(self.encode_data)
        batch = torch.tensor(batch, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return batch, labels


def test(filename, context_size, n_words=5000, model_type="skipgram", subsampling=False, sampling_rate=0.001):
    textdata = TextData(filename=filename, context_size=context_size, model_type=model_type, subsampling=subsampling,
                        sampling_rate=sampling_rate, n_words=n_words)
    return textdata


if __name__ == "__main__":
    textdata = test("text8", 2, 5000, "skipgram", True, 0.001)
    print("processed_text len is {}".format(len(textdata.get_raw_data())))
    print("encode_text len is {}".format(len(textdata.get_encode_data())))
    print("vocab len is {}".format(len(textdata.get_word_count())))
    print("word_to_ix len is {}".format(len(textdata.get_word_to_ix())))
    print("ix_to_word len is {}".format(len(textdata.ix_to_word)))
    train_data, train_label = textdata.get_batch(20)
    print("training_data len is {}".format(len(train_data)))
    print("frist 10 training data is {}".format(train_data[:10]))
    print("frist 10 training label is {}".format(train_label[:10]))
