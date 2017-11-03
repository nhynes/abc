"""The Generators."""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as nnf

import environ
from .bottles import BottledLinear, BottledLogSoftmax

TYPES = ('rnn', 'cnn')
RNN, CNN = TYPES


class RNNGenerator(nn.Module):
    """An RNN token generator."""

    def __init__(self, vocab_size, word_emb_dim, gen_dim, num_layers,
                 **kwargs):
        super(RNNGenerator, self).__init__()

        padding_idx = None if kwargs.get('env') == environ.SYNTH else 0
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim,
                                     padding_idx=padding_idx)
        self.gen_rnn = nn.LSTM(word_emb_dim, gen_dim, num_layers)
        self.word_dec = nn.Sequential(
            BottledLinear(gen_dim, vocab_size),
            BottledLogSoftmax(),
        )

    def forward(self, toks, prev_state=None, **unused_kwargs):
        """
        toks: N*T
        """
        wembs = self.word_emb(toks).transpose(0, 1) # T*N*d_wemb
        tok_embs, next_state = self.gen_rnn(wembs, prev_state)
        tok_probs = self.word_dec(tok_embs)         # T*N*vocab_size
        return tok_probs, next_state

    def rollout(self, init_state, ro_steps, return_first_state=False):
        """
        init_state: (N*t : toks, _ : hidden state); t < seqlen
        ro_steps: roll out this many steps

        This method does not modify the global rand state.

        Returns:
            a list of (T-t) samples of size N*1,
            T*N*V tensor of word log-probs,
        """

        if isinstance(init_state, Variable):
            init_state = (init_state, None)
        gen_toks, gen_state = init_state

        gen_seqs = []
        gen_probs = []
        for i in range(ro_steps):
            tok_probs, gen_state = self(gen_toks, gen_state)
            tok_probs = tok_probs[-1].exp()                      # N*V
            gen_toks = torch.multinomial(tok_probs, 1).detach()  # N*1
            gen_seqs.append(gen_toks)
            gen_probs.append(tok_probs)
            if i == 0 and return_first_state:
                th = torch.cuda if gen_toks.is_cuda else torch
                first_state = (gen_state, th.get_rng_state())
        # gen_seqs = torch.cat(gen_seqs, -1)

        if return_first_state:
            return gen_seqs, gen_probs, first_state
        return gen_seqs, gen_probs


class CNNGenerator(nn.Module):
    """A CNN token generator (only used to generate whole sequences)."""

    def __init__(self, vocab_size, word_emb_dim, gen_dim, seqlen, **kwargs):
        super(CNNGenerator, self).__init__()

        self.word_emb_dim = word_emb_dim

        kw = 5
        gen_len = 1
        pad = 1  # padding
        layer_specs = []
        while True:
            stride = 1 + (len(layer_specs) < 3)
            new_len = (gen_len - 1)*stride + kw - 2*pad
            if new_len > seqlen:
                break
            layer_specs.append((kw, stride, pad))
            gen_len = new_len
        remainder = seqlen - gen_len
        if remainder > 0:
            layer_specs.append((remainder+1, 1, 0))

        layers = []
        d_step = (gen_dim - word_emb_dim) // len(layer_specs)
        d = word_emb_dim
        for i, (kw, stride, pad) in enumerate(layer_specs, 1):
            d_out = d + d_step if i < len(layer_specs) else gen_dim
            layers.append(nn.ConvTranspose1d(
                d, d_out, kw, stride=stride, padding=pad))
            layers.append(nn.ReLU(inplace=True))
            d = d_out
        layers.append(nn.Conv1d(gen_dim, vocab_size, 1))
        self.gen = nn.Sequential(*layers)

    def forward(self, init, **unused_kwargs):
        """ init: N*d """
        word_embs = self.gen(init.unsqueeze(-1)).unsqueeze(-1)
        word_log_probs = nnf.log_softmax(word_embs).squeeze(-1)  # N*V*T
        return word_log_probs.permute(2, 0, 1)  # T*N*V

    def rollout(self, init_state, ro_steps):
        """Just generate a totally random sequence.

        init_state: N*1
        """
        if isinstance(init_state, Variable):
            init_state = (init_state, None)
        gen_toks, gen_state = init_state

        th = torch.cuda if gen_toks.is_cuda else torch
        init = Variable(th.FloatTensor(
            gen_toks.size(0), self.word_emb_dim).normal_(), volatile=True)

        tok_log_probs = self(init)
        tok_probs = tok_log_probs.exp()
        flat_tok_probs = tok_probs.view(-1, tok_probs.size(-1))  # (T*N)*V
        flat_gen_toks = torch.multinomial(flat_tok_probs, 1)
        gen_toks = flat_gen_toks.view(tok_probs.size()[:-1]).t()
        return gen_toks, tok_log_probs


def create(opts):
    """Creates a token generator."""
    gen_type = getattr(opts, 'gen_type', RNN)
    gen_cls = CNNGenerator if gen_type == CNN else RNNGenerator
    return gen_cls(word_emb_dim=opts.g_word_emb_dim,
                   num_layers=opts.num_gen_layers, **vars(opts))


def test_rnn_generator():
    import common

    batch_size = 4
    vocab_size = 32
    word_emb_dim = 8
    gen_dim = 12
    num_layers = 1
    debug = True

    gen = RNNGenerator(**locals())

    toks = Variable(torch.LongTensor(batch_size, 1).fill_(1))

    gen_probs, gen_state = gen(toks)
    gen_toks = torch.multinomial(gen_probs.exp(), 1)
    gen_probs, gen_state = gen(toks=gen_toks, prev_state=gen_state)

    # test basic rollout
    init_toks = Variable(torch.LongTensor(batch_size, 4).fill_(1))
    ro_seqs, _ = gen.rollout(init_toks, 4, 0)
    assert len(ro_seqs) == 4

    # test reproducability
    init_rand_state = torch.get_rng_state()
    with common.rand_state(torch, 42) as rand_state:
        ro1, _ = gen.rollout(init_toks, 8)
    with common.rand_state(torch, rand_state):
        ro2, _ = gen.rollout(init_toks, 8)
    assert all((t1.data == t2.data).all() for t1, t2 in zip(ro1, ro2))
    assert (torch.get_rng_state() == init_rand_state).all()

    # test continuation
    rand_toks = Variable(torch.LongTensor(batch_size, 2).random_(vocab_size))
    ro_seqs, _, (ro_hid, ro_rng) = gen.rollout(rand_toks, 2,
                                               return_first_state=True)
    with common.rand_state(torch, ro_rng):
        next_ro, _ = gen.rollout((ro_seqs[0], ro_hid), 1)
    assert (ro_seqs[1].data == next_ro[0].data).all()

def test_cnn_generator():
    batch_size = 4
    vocab_size = 32
    word_emb_dim = 8
    gen_dim = 12
    seqlen = 20
    debug = True

    gen = CNNGenerator(**locals())

    init = Variable(torch.randn(batch_size, word_emb_dim))

    tok_log_probs = gen(init)
    assert tok_log_probs.size(0) == seqlen
    assert tok_log_probs.size(1) == batch_size
    assert tok_log_probs.size(2) == vocab_size
    assert torch.np.allclose(tok_log_probs.exp().sum(-1).data.numpy(), 1)

    gen_toks, _ = gen.rollout(init, seqlen)
    assert gen_toks.size(0) == batch_size
    assert gen_toks.size(1) == seqlen
