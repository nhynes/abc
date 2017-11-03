"""The Discriminator."""
import functools

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as nnf


def _l2_reg(mod, l=1e-4):
    def _reg(var, grad):
        return grad + l*var
    mod.weight.register_hook(functools.partial(_reg, mod.weight))
    mod.bias.register_hook(functools.partial(_reg, mod.bias))
    return mod


class Highway(nn.Module):
    def __init__(self, in_features, activation=nn.ReLU(True)):
        super(Highway, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )

        self.tx = nn.Sequential(
            nn.Linear(in_features, in_features),
            activation,
        )

    def forward(self, x):
        g = self.gate(x)
        return g * self.tx(x) + (1 - g) * x


class Discriminator(nn.Module):
    """A CNN token discriminator."""

    def __init__(self, vocab_size, word_emb_dim,
                 filter_widths, num_filters, dropout, **kwargs):
        super(Discriminator, self).__init__()

        assert len(filter_widths) == len(num_filters)

        pad_idx = None if kwargs.get('synth') else 0
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim,
                                     padding_idx=pad_idx)

        cnn_layers = []
        for kw, c in zip(filter_widths, num_filters):
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(word_emb_dim, c, kw),
                nn.ReLU(True),
            ))
        self.cnn_layers = nn.ModuleList(cnn_layers)

        emb_dim = sum(num_filters)
        self.cls = nn.Sequential(
            Highway(emb_dim),
            nn.Dropout(dropout),
            _l2_reg(nn.Linear(emb_dim, 2)),
            nn.LogSoftmax(),
       )

    def forward(self, toks):
        """
        toks: N*T
        """
        if isinstance(toks, (list, tuple)):
            toks = torch.cat(toks, -1)

        embs = self.word_emb(toks).transpose(1, 2)  # N*d_wemb*T

        max_acts = []  # num_layers*[N*c]
        for layer in self.cnn_layers:
            max_acts.append(layer(embs).max(-1)[0])
        max_acts = torch.cat(max_acts, -1)  # N*sum(num_filters)

        preds = self.cls(max_acts)

        return preds


def create(d_word_emb_dim, **opts):
    """Creates a token discriminator."""
    return Discriminator(word_emb_dim=d_word_emb_dim, **vars(opts))


def test_discriminator():
    batch_size = 3
    vocab_size = 32
    word_emb_dim = 100
    filter_widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    dropout = 0.25
    debug = True

    d = Discriminator(**locals())

    preds = d(Variable(torch.LongTensor(batch_size, 20).fill_(1)))
    assert preds.size(0) == batch_size
    assert preds.size(1) == 2
