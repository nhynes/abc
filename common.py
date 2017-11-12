"""Utility functions for training neural networks."""

import pickle
from contextlib import contextmanager


RUN_DIR = 'run'
LOG_FILE = 'log.txt'
OPTS_FILE = 'opts.pkl'
STATE_FILE = 'state.pth'

PHASES = ('g_ml', 'd_ml', 'adv')
G_ML, D_ML, ADV = PHASES
LABEL_GEN, LABEL_REAL = 0, 1

EXTRA_VOCAB = ['PAD', 'UNK', '<s>', '</s>']
PAD, UNK, BOS, EOS = EXTRA_VOCAB


@contextmanager
def rand_state(th, rand_state):
    """Pushes and pops a random state.
    th: torch or torch.cuda
    rand_state: an integer or tensor returned by `get_rng_state`
    """
    orig_rand_state = th.get_rng_state()
    if isinstance(rand_state, int):
        th.manual_seed(rand_state)  # this is a slow operation!
        rand_state = th.get_rng_state()
    th.set_rng_state(rand_state)
    yield rand_state
    th.set_rng_state(orig_rand_state)


def unpickle(path_pkl):
    """Loads the contents of a pickle file."""
    with open(path_pkl, 'rb') as f_pkl:
        return pickle.load(f_pkl)


def load_txt(path_txt):
    """Loads a text file."""
    with open(path_txt) as f_txt:
        return [line.rstrip() for line in f_txt]


class Vocab(object):
    """Represents a token2index and index2token map."""

    def __init__(self, tok_counts, unk_tok=None):
        """Constructs a Vocab ADT."""
        self.tok_counts = tok_counts
        self.w2i = {w: i for i, (w, _) in enumerate(self.tok_counts)}

        self.unk_tok = unk_tok
        if unk_tok is not None:
            assert unk_tok in self.w2i
            self.unk_idx = self.w2i[unk_tok]

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < len(self):
                return self.tok_counts[index][0]
            elif self.unk_tok:
                return self.unk_idx
            else:
                raise IndexError(f'No token in position {index}!')
        elif isinstance(index, str):
            if index in self.w2i:
                return self.w2i[index]
            elif self.unk_tok:
                return self.unk_idx
            else:
                raise KeyError(f'{index} not in vocab!')
        else:
            raise ValueError('Index to Vocab must be string or int.')

    def add_extra_vocab(self, extra_vocab):
        """Returns a new Vocab with extra tokens prepended."""
        extra_tok_counts = [(w, float('inf')) for w in extra_vocab]
        return Vocab(extra_tok_counts + self.tok_counts,
                     unk_tok=self.unk_tok)

    def set_unk_tok(self, unk_tok):
        """Sets the token/index to return when looking up an OOV token."""
        return Vocab(self.tok_counts, unk_tok=unk_tok)

    def truncate(self, size):
        """Returns a new Vocab containing the top `size` tokens."""
        return Vocab(self.tok_counts[:size], unk_tok=self.unk_tok)

    def __len__(self):
        return len(self.tok_counts)
