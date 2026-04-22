import collections
import random
import re
import torch
from torch import nn
import lib.d2l as d2l
from utils import print_val_loss


class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=20, reserved_tokens=[]):
        # Flatten a 2D list if needed.
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # Count token frequencies.
        # {' ': 32775, 'e': 17838, ... }
        counter = collections.Counter(tokens)

        # [(' ', 32755), ('e', 17838), ...]
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens.
        # [' ', '<unk>', 'a', 'b',..., 'z']
        # [value] + [value2] represents list concatenation in python.
        self.idx_to_token = list(
            sorted(
                set(['<unk>'] + reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq])
            )
        )

        # {' ': 0, '<unk>': 1, 'a': 2, ..., 'z': 27}
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab


data = TimeMachine()
raw_text = data._download()
text = data._preprocess(raw_text)
tokens = data._tokenize(text)

# corpus, vocab = data.build(raw_text)
# print(len(corpus), len(vocab))
# print(corpus[0:50])

# words = text.split()
# vocab = Vocab(words)
# print(vocab.token_freqs[:10])
# print(len(vocab))

#
# vocab = Vocab(tokens)
# indices = vocab[tokens[:10]]
# print('indices:', indices)
# print('words:', vocab.to_tokens(indices))

# Corpus is an array of indices.
# corpus, vocab = data.build(raw_text)
# print(len(corpus), len(vocab))
# print(corpus[0:50])

words = text.split()
vocab = Vocab(words)
#
freqs = [freq for token, freq in vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
#          xscale='log', yscale='log')
# d2l.plt.show()

bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])
#
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
# print(trigram_vocab.token_freqs[:10])
#
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
# d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
#          ylabel='frequency: n(x)', xscale='log', yscale='log',
#          legend=['unigram', 'bigram', 'trigram'])
# d2l.plt.show()
#

# Zipfian's Law: frequency n_i of the ith most frequent word is:
#   n_i is proportional to 1 / i^a
#   or
#   log(n_i) = -alog(i) + c,
# where a is the exponent that characterizes the distribution and c is a constant.


# print(freqs)            # [2261, 1267, 1245, 1155, 816, 695, 552, 541, 443, 440, 437, 354, 281, 270, 243, 221, 216, 204]
# print(bigram_freqs)     # [309, 169, 130, 112, 109, 102, 99, 85, 78, 73, 68, 67, 62, 61, 61, 60, 51,...]
# print(trigram_freqs)    # [59, 30, 24, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 12, 11...]

results = {}
a_options = [1.3, 1.35, 1.4, 1.45]
for a in a_options:
    results[a] = []
    for i in bigram_freqs[0:20]:
        n_i = freqs[i]
        rhs = 1 / i**a
        results[a].append((n_i, rhs))
for a in a_options:
    print(results[a])


"""
Exercises

1. Tokenize text into words and vary the min_freq arg value. Describe changes to size of resulting vocab.
    min_freq=0: 4580
    min_freq=1: 4580
    min_freq=2: 2183
    min_freq=3: 1420
    min_freq=4: 1044
    min_freq=5: 825
    min_freq=10: 400
    min_freq=20: 219
    
    Seems like when you double min_freq, the vocab size is halved.
    Something like min_freq_t1 / min_freq_t2 ~= vocab_t2 / vocab_t1, where t2 >= t1
    
    I.E. (approx) half the words are used twice, a third of the words are used 3 times,
    a fourth or the words are used 4 times, etc.
    
2. Estimate the exponent of Zipfian distribution for unigrams, bigrams, and trigrams in this corpus.
    unigrams: around 1.65
    bigrams: around 1.3
"""