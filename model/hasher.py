"""A Hasher for locality sensitive hashing."""

import torch
from torch import nn
from torch.nn import functional as nnf
from torch.autograd import Variable

import environ
from .bottles import BottledLinear


TYPES = ('ae',)
AE, = TYPES


class RNNEncoder(nn.Module):
    """Encodes tokens to binary codes."""

    def __init__(self, code_dim, vocab_size, word_emb_dim, rnn_dim,
                 num_layers, **kwargs):
        super(RNNEncoder, self).__init__()

        self.code_dim = code_dim

        padding_idx = None if kwargs.get('env') == environ.SYNTH else 0
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim,
                                     padding_idx=padding_idx)

        self.enc = nn.LSTM(word_emb_dim, rnn_dim, num_layers=num_layers,
                           bidirectional=True)
        self.precoder = nn.Linear(rnn_dim*2, code_dim*2)

    def forward(self, toks):
        """
        toks: N*T
        """
        wembs = self.word_emb(toks).transpose(0, 1) # T*N*d_wemb
        sembs, _ = self.enc(wembs)
        precodes = self.precoder(sembs[-1])  # N*code_dim_x2
        # maybe self-attention, max-over-time, etc. would work better?

        precodes = precodes.view(-1, 2)
        flat_codes = nnf.gumbel_softmax(precodes, hard=True)[:, 0].contiguous()

        return flat_codes.view(-1, self.code_dim)


class RNNDecoder(nn.Module):
    """Decodes codes to token log-probs."""

    def __init__(self, code_dim, vocab_size, seqlen, rnn_dim, **unused_kwargs):
        super(RNNDecoder, self).__init__()

        self.seqlen = seqlen

        self.dec = nn.LSTM(code_dim, rnn_dim)
        self.word_dec = BottledLinear(rnn_dim, vocab_size)

    def forward(self, codes, *unused_kwargs):
        """
        codes: N*code_dim
        """
        rep_codes = codes[None].expand(self.seqlen, *codes.size())
        tok_embs, _ = self.dec(rep_codes)
        tok_logits = self.word_dec(tok_embs)
        return nnf.log_softmax(tok_logits, dim=2)


class AEHasher(nn.Module):
    """An autoencoder-based hasher."""

    def __init__(self, **kwargs):
        super(AEHasher, self).__init__()

        self.encoder = RNNEncoder(**kwargs)
        self.decoder = RNNDecoder(**kwargs)

    def forward(self, toks, **unused_kwargs):
        """
        toks: N*T

        In training mode, return log-probs of reconstructed tokens.
        In evaluate mode, return binary codes
        """
        codes = self.encoder(toks)
        if self.training:
            return self.decoder(codes)
        return codes


def create(g_word_emb_dim, num_gen_layers, **opts):
    """Creates a token generator."""
    return AEHasher(word_emb_dim=g_word_emb_dim,
                    num_layers=num_gen_layers,
                    **opts)


def test_ae_hasher():
    """Tests the AEHashser."""
    # pylint: disable=too-many-locals,unused-variable
    import common

    batch_size = 4
    code_dim = 3
    seqlen = 4
    vocab_size = 32
    word_emb_dim = 8
    rnn_dim = 12
    num_layers = 1
    debug = True

    hasher = AEHasher(**locals())

    toks = Variable(torch.LongTensor(batch_size, 1).fill_(1))

    hasher.train()
    tok_log_probs = hasher(toks)
    assert tok_log_probs.size(1) == seqlen
    assert tok_log_probs.size(2) == vocab_size

    hasher.eval()
    hash_code = hasher(toks)
    assert hash_code.size(1) == code_dim
    assert ((hash_code == 0) + (hash_code == 1)).all()
