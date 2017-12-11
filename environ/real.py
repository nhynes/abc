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
            pretrain_g_epochs=10,
            pretrain_d_epochs=10,
            train_hasher_epochs=15,
            adv_train_iters=750,
            code_len=11,
            dropout=0.25,
            batch_size=256,
            lr_g=0.001,
            lr_d=0.001,
            lr_hasher=0.002,
            hasher_ent_reg=0.3,
            log_freq=20,
            )
        return parser

    def __init__(self, opts):
        """Creates a NLEnvironment."""
        if opts.load_w2v:
            w2v = torch.from_numpy(torch.np.load(opts.load_w2v))
            w2v[0] = 0
            opts.g_tok_emb_dim = opts.d_tok_emb_dim = w2v.shape[1]

        super(NLEnvironment, self).__init__(opts)

        if opts.load_w2v:
            def _grad_mask(grad):
                masked_grad = grad.clone()
                masked_grad[len(common.EXTRA_VOCAB):] = 0
                return masked_grad

            def _set_w2v(emb):
                tok_embs = emb.weight
                tok_embs.data.copy_(w2v)
                tok_embs.register_hook(_grad_mask)

            for net in (self.g, self.d):
                _set_w2v(net.tok_emb)
            if opts.exploration_bonus:
                _set_w2v(self.hasher.tok_emb)

        self.train_dataset = dataset.NLDataset(part='train', **vars(opts))
        self.test_dataset = dataset.NLDataset(part='val', **vars(opts))

        self.ro_init_toks.data.fill_(self.train_dataset.vocab[common.BOS])
        self.opts.padding_idx = self.train_dataset.vocab[common.PAD]
        self.opts.eos_idx = self.train_dataset.vocab[common.EOS]

    def _compute_eval_metric(self):
        test_loader = self._create_dataloader(self.test_dataset)
        val_loss = sum(self._forward_g_ml(batch, volatile=True)[0].data[0]
                       for batch in test_loader) / len(test_loader)

        init_toks_volatile = self.init_toks.volatile
        self.init_toks.volatile = True
        gen_toks, _ = self.g.rollout(self.init_toks[:5], self.opts.seqlen)
        for tok_vec in torch.cat(gen_toks, -1).data:
            logging.debug(self.test_dataset.decode(tok_vec))
        logging.debug('\n---')
        self.init_toks.volatile = init_toks_volatile

        return val_loss
