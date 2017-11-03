import argparse

from torch.autograd import Variable
import torch

import model

class Environment(object):
    def __init__(self, opts):
        self.opts = opts

        self.g = model.generator.create(**vars(opts)).cuda()
        self.d = model.discriminator.create(**vars(opts)).cuda()

        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=opts.lr_g)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=opts.lr_d)

        self.init_toks = Variable(torch.LongTensor(opts.batch_size, 1).cuda())

    @classmethod
    def get_opt_parser(cls):
        parser = argparse.ArgumentParser(add_help=False)

        # general
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--debug', action='store_true')

        # data
        parser.add_argument('--num-gen-samps', default=10000, type=int)
        parser.add_argument('--nworkers', default=4, type=int)

        # model
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
        parser.add_argument('--pretrain-g-epochs', default=50, type=int)
        parser.add_argument('--pretrain-d-epochs', default=150, type=int)
        parser.add_argument('--adv-train-iters', default=150, type=int)
        parser.add_argument('--adv-g-iters', default=150, type=int)
        parser.add_argument('--adv-d-iters', default=5, type=int)
        parser.add_argument('--adv-d-epochs', default=3, type=int)
        parser.add_argument('--num-rollouts', default=16, type=int)

        # output
        parser.add_argument('--dispfreq', default=10, type=int)

        return parser

    def compute_acc(self, probs, labels):
        """Computes the accuracy given prob and labels Variables."""
        preds = probs.max(1)[1]
        return ((preds == labels).sum().data[0] / len(labels))

    @property
    def state(self):
        return (self.g.state_dict(), self.d.state_dict(),
                self.optim_g.state_dict(), self.optim_d.state_dict())

    @state.setter
    def state(self, state):
        g_state, d_state, optim_g_state, optim_d_state = state
        self.g.load_state_dict(g_state)
        self.d.load_state_dict(d_state)
        self.optim_g.load_state_dict(optim_g_state)
        self.optim_d.load_state_dict(optim_d_state)

    def pretrain_g(self):
        raise NotImplementedError()

    def pretrain_d(self):
        raise NotImplementedError()

    def train_adv(self):
        raise NotImplementedError()

    def _create_dataloader(self, dataset):
        dl_opts = {'batch_size': self.opts.batch_size,
                   'num_workers': self.opts.nworkers,
                   'pin_memory': True,
                   'shuffle': True}
        return torch.utils.data.DataLoader(dataset, **dl_opts)
