"""A Dataset that loads a natural language corpus."""

import os

import torch
import torch.utils.data

from common import EXTRA_VOCAB, UNK, BOS, EOS
import common


class NLDataset(torch.utils.data.Dataset):
    """Loads the data."""

    def __init__(self, data_dir, vocab_size, seqlen, part, **unused_kwargs):
        super(NLDataset, self).__init__()

        self.seqlen = seqlen
        self.part = part

        self.vocab = (common.unpickle(os.path.join(data_dir, 'vocab.pkl'))
                      .add_extra_vocab(EXTRA_VOCAB)
                      .truncate(vocab_size).set_unk_tok(UNK))

        self.qs = common.unpickle(os.path.join(data_dir, part + '.pkl'))

    def __getitem__(self, index):
        q = self.qs[index]
        toks = q.split(' ')[:self.seqlen-1]

        qtoks = torch.LongTensor(self.seqlen + 1).zero_()
        qtoks[0] = self.vocab[BOS]
        for i, tok in enumerate(toks, 1):
            qtoks[i] = self.vocab[tok]
        qtoks[len(toks)] = self.vocab[EOS]

        return qtoks, 1

    def __len__(self):
        return len(self.qs)

    def decode(self, toks_vec):
        """Turns a vector of token indices into a string."""
        return ' '.join([self.vocab[idx] for idx in toks_vec if idx > 0])


def create(*args, **kwargs):
    """Returns a NLDataset."""
    return NLDataset(*args, **kwargs)


def test_dataset():
    """Tests the NLDataset."""

    # pylint: disable=unused-variable
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'qa')
    part = 'test'
    vocab_size = 25000
    seqlen = 21
    debug = True

    ds = NLDataset(**locals())
    toks, labels = ds[0]
    print(toks)
    print(ds.decode(toks))

    for i in torch.randperm(len(ds)):
        toks, labels = ds[i]
        assert (toks >= 0).all() and (toks < vocab_size).all()
