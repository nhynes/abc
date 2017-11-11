"""A class for training a SeqGAN model."""

import argparse
import logging

import torch
from torch.nn import functional as nnf
from torch.autograd import Variable

import model


class Environment(object):
    """A base class for training a SeqGAN model."""

    _STATEFUL = ('g', 'd', 'optim_g', 'optim_d')

    def __init__(self, opts):
        """Creates an Environment."""

        self.opts = opts

        self.g = model.generator.create(**vars(opts)).cuda()
        self.d = model.discriminator.create(**vars(opts)).cuda()
        self.tgt_g = model.generator.create(**vars(opts)).cuda()

        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=opts.lr_g)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=opts.lr_d)

        num_inits = max(opts.num_rollouts, 1) * opts.batch_size
        self.ro_init_toks = Variable(torch.cuda.LongTensor(num_inits, 1))
        self.init_toks = self.ro_init_toks[:opts.batch_size].detach()

        self._labels = torch.cuda.LongTensor(self.opts.batch_size)
        # self._inv_idx = torch.arange(opts.batch_size-1, -1, -1).long().cuda()

    @classmethod
    def get_opt_parser(cls):
        """Returns an `ArgumentParser` that parses env-specific opts."""
        parser = argparse.ArgumentParser(add_help=False)

        # general
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--debug', action='store_true')

        # data
        parser.add_argument('--num-gen-samps', default=10000, type=int)
        parser.add_argument('--nworkers', default=4, type=int)

        # model
        parser.add_argument('--d-type', choices=model.discriminator.TYPES,
                            default=model.discriminator.RNN)
        parser.add_argument('--vocab-size', type=int)
        parser.add_argument('--g-word-emb-dim', type=int)
        parser.add_argument('--d-word-emb-dim', type=int)
        parser.add_argument('--gen-dim', type=int)
        parser.add_argument('--num-gen-layers', default=1, type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--seqlen', type=int)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--num-filters',
                            default=[100] + [200]*4 + [100]*5 + [160]*2,
                            nargs='+', type=int)
        parser.add_argument('--filter-widths',
                            default=list(range(1, 11)) + [15, 20],
                            nargs='+', type=int)

        # training
        parser.add_argument('--lr-g', type=float)
        parser.add_argument('--lr-d', type=float)
        parser.add_argument('--pretrain-g-epochs', type=int)
        parser.add_argument('--pretrain-d-epochs', type=int)
        parser.add_argument('--adv-train-iters', type=int)
        parser.add_argument('--adv-g-iters', default=2, type=int)
        parser.add_argument('--num-rollouts', default=8, type=int)
        parser.add_argument('--discount', default=0.95, type=float)
        parser.add_argument('--g-ent-reg', default=1e-3, type=float)
        parser.add_argument('--d-ent-reg', default=1e-2, type=float)
        parser.add_argument('--temperature', default=1.2, type=float)

        # output
        parser.add_argument('--dispfreq', default=10, type=int)

        return parser

    @property
    def state(self):
        return {item: getattr(self, item).state_dict()
                for item in self._STATEFUL}

    @state.setter
    def state(self, state):
        for item_name, item_state in state.items():
            item = getattr(self, item_name, None)
            if item is None:
                logging.warn(f'WARNING: missing {item_name}')
                continue  # don't load missing modules/optimizers
            if (isinstance(item, torch.optim.Optimizer) and
                not item_state['state']):
                continue  # ignore unstepped optimizers
            try:
                item.load_state_dict(item_state)
            except (RuntimeError, KeyError, ValueError):
                logging.warn(f'WARNING: could not load state for {item_name}')
        self.optim_g.param_groups[0]['lr'] = self.opts.lr_g
        self.optim_d.param_groups[0]['lr'] = self.opts.lr_d

    def pretrain_g(self):
        """Pretrains G on a to-be-implemented dataset."""
        raise NotImplementedError()

    def pretrain_d(self):
        """Pretrains D on a to-be-implemented dataset and pretrained G."""
        raise NotImplementedError()

    def train_adv(self):
        """Adversarially train G against D."""
        raise NotImplementedError()

    def get_tok_log_probs(self, gen, toks):
        """
        Returns log probabilities of tokens under a generative model.

        Args:
            gen: the generative model
            toks: N*T

        Returns:
            tok_log_probs: T*N*V
        """
        tok_log_probs, _ = gen(torch.cat((self.init_toks, toks), 1))
        return tok_log_probs[:-1]

    def compute_acc(self, probs, label):
        """Computes the accuracy given prob Variable and and label."""
        self._labels.fill_(label)
        preds = probs.data.max(1)[1]
        return (preds == self._labels).float().mean()

    def _create_dataloader(self, dataset):
        dl_opts = {'batch_size': self.opts.batch_size,
                   'num_workers': self.opts.nworkers,
                   'pin_memory': True,
                   'shuffle': True}
        return torch.utils.data.DataLoader(dataset, **dl_opts)

    def _forward_g_pretrain(self, batch, volatile=False):
        toks, _ = batch
        toks = Variable(
            toks.view(-1, toks.size(-1)), volatile=volatile).cuda()
        flat_tgts = toks[:, 1:].t().contiguous().view(-1)

        gen_log_probs, _ = self.g(toks[:, :-1])
        flat_gen_log_probs = gen_log_probs.view(-1, gen_log_probs.size(-1))
        return nnf.nll_loss(flat_gen_log_probs, flat_tgts), gen_log_probs

    def _forward_d(self, batch, volatile=False, has_init=True):
        toks, labels = batch
        toks = Variable(toks.view(-1, toks.size(-1)), volatile=volatile).cuda()
        labels = Variable(labels.view(-1), volatile=volatile).cuda()

        d_log_probs = self.d(toks[:, has_init:])
        return nnf.nll_loss(d_log_probs, labels), d_log_probs

    @staticmethod
    def _gather_act_probs(acts, probs):
        """
        Returns probs for each action: (N*T)

        acts: N*T
        probs: T*N*V
        """
        return probs.transpose(0, 1).gather(-1, acts.unsqueeze(-1)).squeeze(-1)
