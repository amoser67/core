import collections
import re
import torch
import lib.d2l as d2l


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
    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        super(TimeMachine, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = torch.tensor([corpus[i:i + num_steps + 1] for i in range(len(corpus) - num_steps)])
        self.X = array[:, :-1]
        self.Y = array[:, 1:]

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

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


data = d2l.TimeMachine(batch_size=2, num_steps=10)
# for X, Y in data.train_dataloader():
#     print('X:', X, '\nY:', Y)
#     break
# X: tensor([[21,  6,  5,  0,  9, 10, 14, 20,  6, 13], [ 0,  2, 13, 14, 16, 20, 21,  0, 22, 15]])
# Y: tensor([[ 6,  5,  0,  9, 10, 14, 20,  6, 13,  7], [ 2, 13, 14, 16, 20, 21,  0, 22, 15, 10]]


"""
Exercises

1. Suppose there are 100,000 words in the training set. How much word frequency and multi-word adjacent frequency
does a four-gram need to store?

    - Odd wording, unclear what it is asking.
    - There are 100,000 - (4 - 1) four-grams.
    - P(4-gram) = P(x_1) * P(x_2 | x_1) * P(x_3 | x_1, x_2) * P(x_4 | x_1, x_2, x_3)
    - For a given four-gram
        - P(x_1) = n(x_1) / 100,000
        - P(x_2 | x_1) = n(x_1, x_2) / n(x_1)
        - P(x_3 | x_2, x_1) = n(x_1, x_2, x_3) / n(x_1, x_2)
        - P(x_4 | x_3, x_2, x_1) = n(x_1, x_2, x_3, x_4) / n(x_1, x_2, x_3)
    - For the next four gram
        - P(x_2) = n(x_2) / 100,000
        - P(x_3 | x_2) = n(x_2, x_3) / n(x_2)
        - ...
    - So each four-gram needs n(x_1), n(x_1, x_2), n(x_1, x_2, x_3), n(x_1, x_2, x_3, x_4)


2. How would you model a dialogue?

    - Might be best to separate them. Loses some context though, so hard to say.
    

4. Consider the method for discarding a uniformly random number of the first few tokens at the beginning of each epoch.
    a. Does it really lead to a perfectly uniform distribution over the sequences on the document?
        - Not necessarily. The number of epochs would need to be divisible by the number of possible d values.
        
5. How would you handle sequences that must be full sentences when using minibatch sampling.
    - problem is that the length of sentences varies a lot and it doesn't seem practical to construct a sentence vocab. 

"""