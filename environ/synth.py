import argparse
import os

import common
import model
from .environment import Environment

class SynthEnvironment(Environment):
    """Functions for training a model on a synthetic dataset."""

    @classmethod
    def get_opt_parser(cls):
        """Returns an `ArgumentParser` that parses env-specific opts."""
        parser = super(SynthEnvironment, cls).get_opt_parser()
        parser.add_argument(
            '--oracle-type', default=model.generator.RNN,
            choices=model.generator.TYPES)
        parser.set_defaults(
            seqlen=20,
            vocab_size=5000,
            g_word_emb_dim=34,
            d_word_emb_dim=64,
            gen_dim=32,
            dropout=0.25,
            num_gen_layers=1,
            lr_g=0.001,
            lr_d=0.001,
            )
        return parser

    def __init__(self, opts):
        """Creates a SynthEnvironment."""
        super(SynthEnvironment, self).__init__(opts)

        self.oracle = self._create_oracle()
        print('here')
        exit()

    def _create_oracle(self):
        """Returns a randomly initialized generator."""
        with common.rand_state(self.opts.seed):
            oracle = model.generator.create(
                gen_type=opts.oracle_type, **vars(self.opts))
            for param in oracle.parameters():
                nn.init.normal(param, std=1)
        return oracle

    def compute_oracle_nll(self, toks, return_probs=False):
        """
        oracle: a Generator
        toks: [N]*T
        """
        init_toks = Variable(
            torch.LongTensor(toks[0].size(0), 1).fill_(1).type_as(toks[0].data))
        gen_probs, _ = self.oracle(torch.cat([init_toks] + toks, 1))
        gen_probs = gen_probs[:-1]
        flat_gen_probs = gen_probs.view(-1, gen_probs.size(-1))
        flat_toks = torch.cat(toks).squeeze(1)
        nll = nnf.nll_loss(flat_gen_probs, flat_toks).data[0]
        if return_probs:
            return nll, gen_probs
        return nll
