"""A class for training a SeqGAN model."""

import argparse
import logging

import torch
from torch.nn import functional as nnf
from torch.autograd import Variable

import model
from dataset import samplers, ReplayBuffer


class Environment(object):
    """A base class for training a SeqGAN model."""

    _STATEFUL = ('g', 'd', 'optim_g', 'optim_d')

    def __init__(self, opts):
        """Creates an Environment."""

        self.opts = opts

        self.g = model.generator.create(**vars(opts)).cuda()
        self.d = model.discriminator.create(**vars(opts)).cuda()

        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=opts.lr_g)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=opts.lr_d)

        if self.opts.exploration_bonus:
            self.hasher = model.hasher.create(**vars(self.opts)).cuda()
            self.optim_hasher = torch.optim.Adam(self.hasher.parameters(),
                                                 lr=self.opts.lr_hasher)
            self.state_counts = torch.LongTensor(2**self.opts.code_dim).zero_()

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
        parser.add_argument('--exploration-bonus', action='store_true')
        parser.add_argument('--code-dim', default=32, type=int)

        # training
        parser.add_argument('--lr-g', type=float)
        parser.add_argument('--lr-d', type=float)
        parser.add_argument('--lr-hasher', type=float)
        parser.add_argument('--pretrain-g-epochs', type=int)
        parser.add_argument('--pretrain-d-epochs', type=int)
        parser.add_argument('--train-hasher-epochs', type=int)
        parser.add_argument('--adv-train-iters', type=int)
        parser.add_argument('--adv-g-iters', default=2, type=int)
        parser.add_argument('--num-rollouts', default=8, type=int)
        parser.add_argument('--discount', default=0.95, type=float)
        parser.add_argument('--g-ent-reg', default=1e-3, type=float)
        parser.add_argument('--d-ent-reg', default=1e-2, type=float)
        parser.add_argument('--temperature', default=1, type=float)

        return parser

    @property
    def state(self):
        """Returns a dict containing the state of this Environment."""
        return {item: getattr(self, item).state_dict()
                for item in self._STATEFUL}

    @state.setter
    def state(self, state):
        for item_name, item_state in state.items():
            item = getattr(self, item_name, None)
            if item is None:
                logging.warning(f'WARNING: missing {item_name}')
                continue  # don't load missing modules/optimizers
            if (isinstance(item, torch.optim.Optimizer) and
                    not item_state['state']):
                continue  # ignore unstepped optimizers
            try:
                item.load_state_dict(item_state)
            except (RuntimeError, KeyError, ValueError):
                logging.warning(f'WARNING: could not load {item_name} state')
        self.optim_g.param_groups[0]['lr'] = self.opts.lr_g
        self.optim_d.param_groups[0]['lr'] = self.opts.lr_d

    def train_hasher(self):
        """Train a model that produces binary codes for LSH."""
        raise NotImplementedError()

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

    def _create_dataloader(self, src_dataset, cycle=False):
        dl_opts = {'batch_size': self.opts.batch_size,
                   'num_workers': self.opts.nworkers,
                   'pin_memory': True,
                   'shuffle': not cycle}
        if cycle:
            dl_opts['sampler'] = samplers.InfiniteRandomSampler(src_dataset)
        return torch.utils.data.DataLoader(src_dataset, **dl_opts)

    def _create_replay_buffer(self, max_history, label):
        replay_buffer = ReplayBuffer(max_history, label)
        sampler = samplers.ReplayBufferSampler(replay_buffer,
                                               self.opts.batch_size)
        loader = torch.utils.data.DataLoader(replay_buffer,
                                             batch_sampler=sampler,
                                             num_workers=0,  # FIXME
                                             pin_memory=True)
        return replay_buffer, loader

    def _forward_seq2seq(self, fwd_fn, batch, volatile=False):
        toks, _ = batch
        toks = Variable(
            toks.view(-1, toks.size(-1)), volatile=volatile).cuda()
        flat_tgts = toks[:, 1:].t().contiguous().view(-1)

        gen_log_probs = fwd_fn(toks[:, :-1])
        flat_gen_log_probs = gen_log_probs.view(-1, gen_log_probs.size(-1))
        loss = nnf.nll_loss(flat_gen_log_probs, flat_tgts)
        return loss, gen_log_probs

    def _forward_hasher_train(self, batch):
        return self._forward_seq2seq(self.hasher, batch)[0]

    def _forward_g_pretrain(self, batch):
        return self._forward_seq2seq(lambda toks: self.g(toks)[0], batch)

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

    @staticmethod
    def _get_entropy(log_probs, discount_rate=None):
        # assumes distributions are along the last dimension
        infos = log_probs.exp() * log_probs
        entropy = entropy_undiscounted = -infos.sum(-1).mean()
        if discount_rate and discount_rate != 1:
            sz = [log_probs.size(0)] + [1]*(log_probs.ndimension() - 1)
            discount = log_probs.data.new(*sz).fill_(1)
            discount[1:] *= discount_rate
            discount.cumprod(0, out=discount)
            infos = infos * Variable(discount)
            entropy = -infos.sum(-1).mean()
        return entropy, entropy_undiscounted
