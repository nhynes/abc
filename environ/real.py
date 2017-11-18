"""An Environment for use with a natural language dataset."""

import os
import logging

import torch

import common
import dataset
from .environment import Environment

class NLEnvironment(Environment):
    """Functions for training a model on the NL dataset."""

    _EVAL_METRIC = 'val'

    @classmethod
    def get_opt_parser(cls):
        """Returns an `ArgumentParser` that parses env-specific opts."""
        parser = super(NLEnvironment, cls).get_opt_parser()
        parser.add_argument(
            '--data-dir', default='data/qa', type=os.path.abspath)
        parser.set_defaults(
            seqlen=22,
            vocab_size=20000,
            g_tok_emb_dim=32,
            d_tok_emb_dim=32,
            rnn_dim=64,
            num_gen_layers=2,
            num_gen_samps=None,
            pretrain_g_epochs=10,
            pretrain_d_epochs=10,
            train_hasher_epochs=7,
            adv_train_iters=750,
            code_len=11,
            dropout=0.25,
            lr_g=0.001,
            lr_d=0.001,
            lr_hasher=0.002,
            )
        return parser

    def __init__(self, opts):
        """Creates a NLEnvironment."""
        super(NLEnvironment, self).__init__(opts)

        self.train_dataset = dataset.NLDataset(part='train', **vars(opts))
        self.test_dataset = dataset.NLDataset(part='val', **vars(opts))

        self.ro_init_toks.data.fill_(self.train_dataset.vocab[common.BOS])

    def _compute_eval_metric(self):
        test_loader = self._create_dataloader(self.test_dataset)
        val_loss = sum(self._forward_g_ml(batch, volatile=True).data[0]
                       for batch in test_loader) / len(test_loader)

        gen_toks, _ = self.g.rollout(self.init_toks[:5], self.opts.seqlen)
        for tok_vec in torch.cat(gen_toks, -1).data:
            logging.debug(self.test_dataset.decode(tok_vec))

        return val_loss
