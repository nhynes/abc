import argparse
import os

import common
import model
from .environment import Environment

class SynthEnvironment(Environment):
    def __init__(self, opts):
        self.opts = opts
        self.oracle = self.create_oracle(opts)
        print('here')
        exit()

    def _create_oracle(self):
        """Returns a randomly initialized generator."""
        with common.rand_state(self.opts.seed):
            oracle = model.generator.create(self.opts)
            if opts.gen_type == model.generator.RNN:
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
