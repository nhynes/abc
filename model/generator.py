"""The Generators."""
import itertools

import torch
from torch import nn
from torch.autograd import Variable

import environ
from .bottles import BottledLinear, BottledLogSoftmax

TYPES = ('rnn', 'rrnn')
RNN, RRNN = TYPES


class RNNGenerator(nn.Module):
    """An RNN token generator."""

    def __init__(self, vocab_size, word_emb_dim, gen_dim, num_layers,
                 rrnn=False, **kwargs):
        super(RNNGenerator, self).__init__()

        padding_idx = None if kwargs.get('env') == environ.SYNTH else 0
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim,
                                     padding_idx=padding_idx)

        if rrnn:
            gen_dim *= 10
        self.gen = nn.LSTM(word_emb_dim, gen_dim, num_layers=num_layers)
        self.word_dec = BottledLinear(gen_dim, vocab_size)

    def forward(self, toks, prev_state=None, temperature=1, **unused_kwargs):
        """
        toks: N*T
        """
        wembs = self.word_emb(toks).transpose(0, 1) # T*N*d_wemb
        tok_embs, next_state = self.gen(wembs, prev_state)
        logits = self.word_dec(tok_embs)  # T*N*vocab_size
        if temperature != 1:
            logits /= temperature
        tok_probs = BottledLogSoftmax()(logits)
        return tok_probs, next_state

    def rollout(self, init_state, ro_steps, return_first_state=False,
                temperature=1):
        """
        init_state:
            toks: N*T or (toks, prev_hidden state)
        ro_steps: roll out this many steps
        temperature: positive scalar parameter that raises entropy

        This method does not modify the global random state.

        Returns:
            a list of T samples of size N*1,
            a list of T word log-probs of size N*V
        """

        if isinstance(init_state, Variable):
            init_state = (init_state, None)
        gen_toks, gen_state = init_state

        gen_seqs = []
        gen_log_probs = []
        for i in range(ro_steps):
            tok_log_probs, gen_state = self(
                gen_toks, gen_state, temperature=temperature)
            tok_log_probs = tok_log_probs[-1]  # N*V
            gen_toks = torch.multinomial(tok_log_probs.exp(), 1).detach()  # N*1
            gen_seqs.append(gen_toks)
            gen_log_probs.append(tok_log_probs)
            if i == 0 and return_first_state:
                th = torch.cuda if gen_toks.is_cuda else torch
                first_state = (gen_state, th.get_rng_state())

        if return_first_state:
            return gen_seqs, gen_log_probs, first_state
        return gen_seqs, gen_log_probs

    def parameters(self, dx2=False):
        """
        Returns an iterator over module parameters.
        If dx2=True, only yield parameters that are twice differentiable.
        """
        if not dx2:
            return super(RNNGenerator, self).parameters()
        return itertools.chain(*[
            m.parameters() for m in self.children()
            if m != self.word_emb])


def create(g_word_emb_dim, num_gen_layers, gen_type=RNN, **opts):
    """Creates a token generator."""
    return RNNGenerator(word_emb_dim=g_word_emb_dim,
                        num_layers=num_gen_layers,
                        rrnn=(gen_type == RRNN), **opts)


def test_rnn_generator():
    """Tests the RNNGenerator."""
    # pylint: disable=too-many-locals,unused-variable
    import common

    batch_size = 4
    seqlen = 4
    vocab_size = 32
    word_emb_dim = 8
    gen_dim = 12
    num_layers = 1
    rrnn = True
    debug = True

    gen = RNNGenerator(**locals())

    toks = Variable(torch.LongTensor(batch_size, 1).fill_(1))

    gen_probs, gen_state = gen(toks)
    gen_toks = torch.multinomial(gen_probs.exp(), 1).detach()
    gen_probs, gen_state = gen(toks=gen_toks, prev_state=gen_state)

    # test basic rollout
    init_toks = Variable(torch.LongTensor(batch_size, seqlen).fill_(1))
    ro_seqs, ro_log_probs = gen.rollout(init_toks, seqlen, 0)
    assert len(ro_seqs) == seqlen and len(ro_log_probs) == seqlen
    assert torch.np.allclose(
        torch.stack(ro_log_probs).data.exp().sum(-1).numpy(), 1)

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

    # test double-backward
    sum(gen_probs).sum().backward(create_graph=True)
    sum(p.grad.norm() for p in gen.parameters(dx2=True)).backward()
