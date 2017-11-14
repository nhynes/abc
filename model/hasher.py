"""A Hasher for locality sensitive hashing."""

import torch
from torch import nn
from torch.nn import functional as nnf
from torch.autograd import Variable

import environ
from .bottles import BottledLinear
from .utils import Apply


TYPES = ('ae',)
AE, = TYPES


class _RNNEncoder(nn.Module):
    """Encodes tokens to binary codes."""

    def __init__(self, code_len, tok_emb_dim, rnn_dim, num_layers, **kwargs):
        super(_RNNEncoder, self).__init__()

        self.code_len = code_len

        self.enc = nn.LSTM(tok_emb_dim, rnn_dim, num_layers=num_layers,
                           bidirectional=True)
        self.precoder = nn.Linear(rnn_dim*2, code_len*2)

    def forward(self, tok_embs):
        """
        tok_embs: T*N*tok_emb_dim
        """
        seq_embs, _ = self.enc(tok_embs)
        precodes = self.precoder(seq_embs[-1])  # N*code_len_x2
        # maybe self-attention, max-over-time, etc. would work better?

        precodes = precodes.view(-1, 2)
        flat_codes = nnf.gumbel_softmax(precodes, hard=True)[:, 0].contiguous()
        # flat_codes = precodes.max(1)[1].float()  # doesn't train very well!
        # flat_codes = (flat_codes * 2) - 1  # {0, 1} -> {-1, 1}
        # flat_codes = nnf.tanh(precodes)

        return flat_codes.view(-1, self.code_len)


class _RNNDecoder(nn.Module):
    """Decodes codes to token log-probs."""

    def __init__(self, code_len, tok_emb_dim, seqlen, rnn_dim, vocab_size,
                 **unused_kwargs):
        super(_RNNDecoder, self).__init__()

        self.seqlen = seqlen
        self.emb_dim = code_len #+ tok_emb_dim

        self.dec = nn.LSTM(self.emb_dim, rnn_dim, bidirectional=True)
        self.tok_dec = BottledLinear(rnn_dim*2, vocab_size)

    def forward(self, tok_embs, codes, *unused_kwargs):
        """
        tok_embs: (T+1)*N*tok_emb_dim; +1 for init toks
        codes: N*code_len
        """
        batch_size, code_dim = codes.size()

        rep_codes = codes[None].expand(self.seqlen, *codes.size())
        # tok_codes = torch.cat((tok_embs[:-1], rep_codes), dim=-1)
        tok_codes = rep_codes
        tok_embs, _ = self.dec(tok_codes)
        tok_logits = self.tok_dec(tok_embs)
        return nnf.log_softmax(tok_logits, dim=2)


class AEHasher(nn.Module):
    """An autoencoder-based hasher."""

    def __init__(self, code_len, num_hash_buckets, **kwargs):
        super(AEHasher, self).__init__()

        tok_emb_dim = kwargs['tok_emb_dim']
        vocab_size = kwargs['vocab_size']
        padding_idx = None if kwargs.get('env') == environ.SYNTH else 0
        self.tok_emb = nn.Embedding(vocab_size, tok_emb_dim,
                                     padding_idx=padding_idx)

        self.encoder = _RNNEncoder(code_len, **kwargs)
        self.decoder = _RNNDecoder(code_len, **kwargs)

        hash_code_len = int(torch.np.ceil(torch.np.log2(num_hash_buckets)))
        self.proj = nn.Sequential()
        if hash_code_len != code_len:
            self.proj = nn.Sequential(
                nn.Linear(code_len, hash_code_len, bias=False),
                Apply(torch.sign))

    def forward(self, toks, **unused_kwargs):
        """
        In training mode, return log-probs of reconstructed tokens.
            toks: N*(T+1); the first timestep is init toks

        In evaluate mode, return binary codes
            toks: N*T; no init toks
        """
        tok_embs = self.tok_emb(toks).transpose(0, 1)
        codes = self.encoder(tok_embs[self.training:])
        if self.training:
            return self.decoder(tok_embs, codes)
        return self.proj(codes).detach()


def create(g_tok_emb_dim, num_gen_layers, **opts):
    """Creates a token generator."""
    return AEHasher(tok_emb_dim=g_tok_emb_dim,
                    num_layers=num_gen_layers,
                    **opts)


def test_ae_hasher():
    """Tests the AEHashser."""
    # pylint: disable=too-many-locals,unused-variable
    import common

    batch_size = 4
    code_len = 3
    num_hash_buckets = 8
    seqlen = 4
    vocab_size = 32
    tok_emb_dim = 8
    rnn_dim = 12
    num_layers = 1
    debug = True

    hasher = AEHasher(**locals())

    toks = Variable(torch.LongTensor(batch_size, seqlen).random_(vocab_size))

    hasher.train()
    tok_log_probs = hasher(toks)
    assert tok_log_probs.size(1) == seqlen
    assert tok_log_probs.size(2) == vocab_size

    hasher.eval()
    hash_code = hasher(toks)
    assert hash_code.size(1) == torch.np.log2(num_hash_buckets)
    assert (hash_code.sign() == hash_code).all()
