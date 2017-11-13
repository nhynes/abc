"""An Environment for use with the QA dataset."""

import os
import logging

import torch
from torch.nn import functional as nnf
from torch.autograd import Variable

import common
import dataset
from .environment import Environment

class QAEnvironment(Environment):
    """Functions for training a model on the QA dataset."""

    @classmethod
    def get_opt_parser(cls):
        """Returns an `ArgumentParser` that parses env-specific opts."""
        parser = super(QAEnvironment, cls).get_opt_parser()
        parser.add_argument(
            '--data-dir', default='data/qa', type=os.path.abspath)
        parser.set_defaults(
            seqlen=22,
            vocab_size=20000,
            g_word_emb_dim=64,
            d_word_emb_dim=64,
            rnn_dim=512,
            num_gen_layers=2,
            lr_g=0.001,
            lr_d=0.001,
            )
        return parser

    def __init__(self, opts):
        """Creates a QAEnvironment."""
        super(QAEnvironment, self).__init__(opts)

        self.train_dataset = dataset.QADataset(part='train', **vars(opts))
        self.val_dataset = dataset.QADataset(part='val', **vars(opts))

        self.ro_init_toks.data.fill_(self.train_dataset.vocab[common.BOS])

    def pretrain_g(self):
        """Pretrains G using maximum-likelihood on the QA dataset."""

        logger = logging.getLogger()

        train_loader = self._create_dataloader(self.train_dataset)
        val_loader = self._create_dataloader(self.val_dataset)

        for epoch in range(1, self.opts.pretrain_g_epochs + 1):
            train_loss = 0
            for batch in train_loader:
                loss = self._forward_g_pretrain(batch)
                train_loss += loss.data[0]

                self.optim_g.zero_grad()
                loss.backward()
                self.optim_g.step()
            train_loss /= len(train_loader)

            val_loss = 0
            for batch in val_loader:
                val_loss += _forward_batch(batch, volatile=True).data[0]
            val_loss /= len(val_loader)

            logger.info(
                f'[{epoch}] loss: train={train_loss:.3f} val={val_loss:.3f}')

            gen_toks, _ = self.g.rollout(self.init_toks[:5], self.opts.seqlen)
            for tok_vec in torch.cat(gen_toks, -1).data:
                logger.debug(self.train_dataset.decode(tok_vec))

    def pretrain_d(self):
        pass

    def train_adv(self):
        pass
