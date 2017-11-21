"""A class for training a SeqGAN model on synthetic data."""

import logging

import torch
from torch import nn
from torch.nn import functional as nnf

import common
from common import LABEL_REAL
import model

from .environment import Environment

class SynthEnvironment(Environment):
    """Functions for training a model on a synthetic dataset."""

    _EVAL_METRIC = 'nll'

    @classmethod
    def get_opt_parser(cls):
        """Returns an `ArgumentParser` that parses env-specific opts."""
        parser = super(SynthEnvironment, cls).get_opt_parser()
        parser.add_argument(
            '--oracle-type', default=model.generator.RNN,
            choices=model.generator.TYPES)
        parser.add_argument('--oracle-dim', default=128, type=int)
        parser.add_argument('--num-gen-samps', default=100000, type=int)
        parser.set_defaults(
            seqlen=20,
            vocab_size=5000,
            g_tok_emb_dim=32,
            d_tok_emb_dim=32,
            pretrain_g_epochs=50,  # try 20 when using pretrained w2v
            pretrain_d_epochs=10,
            train_hasher_epochs=25,
            adv_train_iters=750,
            rnn_dim=32,
            code_len=6,
            dropout=0.25,
            num_gen_layers=1,
            batch_size=64,
            lr_g=0.01,
            lr_d=0.001,
            lr_hasher=0.002,
            )
        return parser

    def __init__(self, opts):
        """Creates a SynthEnvironment."""
        super(SynthEnvironment, self).__init__(opts)

        self.ro_init_toks.data.zero_()
        self.opts.padding_idx = self.opts.eos_idx = None

        self.oracle = self._create_oracle().cuda()
        oracle_checksum = sum(p.data.sum() for p in self.oracle.parameters())
        logging.debug(f'#oracle: {oracle_checksum:.3f}')

        self.train_dataset = self._create_gen_dataset(self.oracle, LABEL_REAL)
        self.test_dataset = self._create_gen_dataset(
            self.oracle, LABEL_REAL,
            num_samples=len(self.ro_init_toks)*5, seed=-1)

        if self.opts.load_w2v:
            oracle_w2v = model.utils.Apply(self.oracle.tok_emb, detach=True)
            for net in (self.g, self.d):
                net.tok_emb = oracle_w2v
            if opts.exploration_bonus:
                self.hasher.encoder.tok_emb = oracle_w2v

    def _create_oracle(self):
        """Returns a randomly initialized generator."""
        with common.rand_state(torch, self.opts.seed):
            opt_vars = vars(self.opts)
            opt_vars.pop('rnn_dim')
            oracle = model.generator.create(
                gen_type=self.opts.oracle_type,
                rnn_dim=self.opts.oracle_dim,
                **opt_vars)
            for param in oracle.parameters():
                nn.init.normal(param, std=1)
        return oracle

    def _compute_eval_metric(self, num_samples=256):
        test_nll = 0
        num_test_batches = max(num_samples // len(self.init_toks), 1)
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                test_nll += self.compute_oracle_nll(gen_seqs)
        test_nll /= num_test_batches
        return test_nll

    def compute_oracle_nll(self, toks, return_probs=False):
        """
        toks: [N]*T
        """
        toks = torch.cat([self.init_toks] + toks).view(len(toks), -1)  # T*N
        log_probs = self.oracle(toks.t())[:-1]  # T*N*V
        flat_log_probs = log_probs.view(-1, log_probs.size(-1))  # (T*N)*V
        nll = nnf.nll_loss(flat_log_probs, toks.view(-1)).data[0]
        if return_probs:
            return nll, log_probs
        return nll
