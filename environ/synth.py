"""A class for training a SeqGAN model on synthetic data."""

import itertools
import logging

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as nnf

import common
from common import LABEL_GEN, LABEL_REAL
import dataset
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
        parser.add_argument('--grad-reg', action='store_true')
        parser.set_defaults(
            seqlen=20,
            vocab_size=5000,
            g_word_emb_dim=34,
            d_word_emb_dim=64,
            pretrain_g_epochs=50,
            pretrain_d_epochs=10,
            adv_train_iters=750,
            gen_dim=32,
            dropout=0.25,
            num_gen_layers=1,
            lr_g=0.01,
            lr_d=0.0001,
            )
        return parser

    def __init__(self, opts):
        """Creates a SynthEnvironment."""
        if opts.grad_reg:
            torch.backends.cudnn.enabled = False

        super(SynthEnvironment, self).__init__(opts)

        self.ro_init_toks.data.zero_()

        self.oracle = self._create_oracle().cuda()
        self.oracle_dataset = self._create_dataset(self.oracle, LABEL_REAL)

        with common.rand_state(torch.cuda, -1):
            self.oracle_test_toks, _ = self.oracle.rollout(self.init_toks,
                                                           self.opts.seqlen)

    def _create_oracle(self):
        """Returns a randomly initialized generator."""
        with common.rand_state(torch, self.opts.seed):
            oracle = model.generator.create(
                gen_type=self.opts.oracle_type, **vars(self.opts))
            for param in oracle.parameters():
                nn.init.normal(param, std=1)
        return oracle

    def _create_dataset(self, gen, label, num_samples=None, seed=None):
        num_samples = num_samples or self.opts.num_gen_samps
        seed = seed or self.opts.seed
        return dataset.GenDataset(generator=gen,
                                  label=label,
                                  seqlen=self.opts.seqlen,
                                  seed=seed,
                                  gen_init_toks=self.init_toks,
                                  num_samples=num_samples)

    def compute_oracle_nll(self, toks, return_probs=False):
        """
        oracle: a Generator
        toks: [N]*T
        """
        gen_probs, _ = self.oracle(torch.cat([self.init_toks] + toks, 1))
        gen_probs = gen_probs[:-1]
        flat_gen_probs = gen_probs.view(-1, gen_probs.size(-1))
        flat_toks = torch.cat(toks).squeeze(1)
        nll = nnf.nll_loss(flat_gen_probs, flat_toks).data[0]
        if return_probs:
            return nll, gen_probs
        return nll

    def compute_acc(self, probs, label):
        """Computes the accuracy given prob Variable and and label."""
        self._labels.fill_(label)
        preds = probs.data.max(1)[1]
        return (preds == self._labels).float().mean()

    def _compute_test_nll(self, num_samples=256):
        test_nll = 0
        num_test_batches = num_samples // len(self.init_toks)
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                test_nll += self.compute_oracle_nll(gen_seqs)
        test_nll /= num_test_batches
        return test_nll

    def _compute_test_acc(self, num_samples=256):
        acc_gen = acc_oracle = 0
        num_test_batches = num_samples // len(self.init_toks)
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                acc_gen += self.compute_acc(self.d(gen_seqs), LABEL_GEN)
                acc_oracle += self.compute_acc(
                    self.d(self.oracle_test_toks), LABEL_REAL)
        acc_gen /= num_test_batches
        acc_oracle /= num_test_batches
        return acc_gen, acc_oracle

    def pretrain_g(self):
        """Pretrains G using maximum-likelihood on a synthetic dataset."""

        dataloader = self._create_dataloader(self.oracle_dataset)

        logging.info(f'[0] nll: {self._compute_test_nll():.3f}')
        for epoch in range(1, self.opts.pretrain_g_epochs + 1):
            train_loss = 0
            for batch in dataloader:
                loss = self._forward_g_pretrain(batch)
                train_loss += loss.data[0]

                self.optim_g.zero_grad()
                loss.backward()
                self.optim_g.step()
            train_loss /= len(dataloader)

            oracle_nll = self._compute_test_nll()
            logging.info(
                f'[{epoch}] loss: {train_loss:.3f}  nll: {oracle_nll:.3f}')

    def pretrain_d(self):
        """Pretrains D using pretrained G."""

        for epoch in range(1, self.opts.pretrain_d_epochs+1):
            gen_dataset = self._create_dataset(self.g, LABEL_GEN,
                                               seed=self.opts.seed+epoch)
            dataloader = self._create_dataloader(torch.utils.data.ConcatDataset(
                (self.oracle_dataset, gen_dataset)))

            train_loss = 0
            epoch_gnorm = 0
            for batch in dataloader:
                loss = self._forward_d(batch)
                train_loss += loss.data[0]

                self.optim_d.zero_grad()
                loss.backward()
                gnorm = sum(
                    (p.grad**2).sum() for p in self.d.parameters(dx2=True))
                epoch_gnorm += gnorm.data[0]
                # (gnorm * 0.01).backward()
                nn.utils.clip_grad_norm(self.d.parameters(dx2=True), 1)
                self.optim_d.step()
            train_loss /= len(dataloader)
            epoch_gnorm /= len(dataloader)

            acc_gen, acc_oracle = self._compute_test_acc()
            logging.info(f'[{epoch}] loss: {train_loss:.3f}  '
                         f'acc: oracle={acc_oracle:.2f}  gen={acc_gen:.2f}  '
                         f'gnorm: {epoch_gnorm:.2f}')

    def _get_qs(self, gen_seqs):
        qs = Variable(self._qs.zero_())

        qs[-1] = self.d(gen_seqs)[:, LABEL_REAL].exp()

        if self.opts.num_rollouts == 0:
            return qs.detach()

        rep_gen_seqs = gen_seqs.repeat(self.opts.num_rollouts, 1)

        ro_rng = torch.cuda.get_rng_state()
        _, ro_hid = self.g_ro(self.ro_init_toks)
        for n in range(1, self.opts.seqlen):
            ro_state = (rep_gen_seqs[:, n-1].unsqueeze(-1), ro_hid)
            ro_seqs, _, (ro_hid, ro_rng) = self.g_ro.rollout(
                ro_state, self.opts.seqlen - n, return_first_state=True)
            full_ro = torch.cat([rep_gen_seqs[:, :n]] + ro_seqs, -1)
            assert full_ro.size(1) == self.opts.seqlen

            q = self.d(full_ro)[:, LABEL_REAL].exp()
            # LABEL_G gives cost, LABEL_REAL gives reward
            qs[n-1] = q.view(self.opts.num_rollouts, -1).mean(0)

            torch.cuda.set_rng_state(ro_rng)

        # qs -= qs.mean()#(0, keepdim=True)
        # qs = qs[:, self._inv_idx].cumsum(1)[:, self._inv_idx]
        qs -= qs.mean(0, keepdim=True)
        return qs.detach()

    @staticmethod
    def _get_grad_norm(mod):
        return sum((p.grad**2).sum() for p in mod.parameters(dx2=True))

    def train_adv(self):
        """Adversarially train G against D."""

        # for optimizer in (self.optim_g, self.optim_d):
        #     optimizer.param_groups[0]['lr'] *= 0.1
        # self.optim_g.param_groups[0]['lr'] *= 0.1
        # self.optim_d.param_groups[0]['lr'] *= 0.1
        self.g_ro = self.g
        for epoch in range(1, self.opts.adv_train_iters+1):
            # self.g_ro.load_state_dict(self.g.state_dict())

            # train G
            gen_seqs, gen_probs = self.g.rollout(self.init_toks,
                                                 self.opts.seqlen)
            gen_seqs = torch.cat(gen_seqs, -1)
            gen_probs = torch.stack(gen_probs)  # T*N

            qs = self._get_qs(gen_seqs)  # T*N

            gen_seq_probs = gen_probs.gather(  # T*N*V -> T*N
                -1, gen_seqs.t().unsqueeze(-1)).squeeze(-1)
            loss_g = -(qs * gen_seq_probs).sum(0).mean()

            # train D
            half_batch = self.opts.batch_size // 2
            oracle_batch = [self.oracle_dataset[i]
                            for i in torch.randperm(
                                len(self.oracle_dataset))[:half_batch]]
            oracle_toks, _ = zip(*oracle_batch)
            oracle_toks = torch.stack(oracle_toks)[:, 1:].cuda()

            gen_seqs, _ = self.g.rollout(
                self.init_toks[:half_batch], self.opts.seqlen)
            gen_seqs = torch.cat(gen_seqs, -1).data

            batch_toks = torch.cat((oracle_toks, gen_seqs), 0)
            self._labels[:half_batch] = LABEL_REAL
            self._labels[half_batch:] = LABEL_GEN

            loss_d = self._forward_d((batch_toks, self._labels), has_init=False)

            self.optim_g.zero_grad()
            self.optim_d.zero_grad()
            loss_g.backward(create_graph=self.opts.grad_reg)
            loss_d.backward(create_graph=self.opts.grad_reg)
            gnormg = self._get_grad_norm(self.g)
            gnormd = self._get_grad_norm(self.d)
            gnorm = gnormg + gnormd
            # gnorm = sum(map(self._get_grad_norm, (self.d, self.g)))
            print(gnormg.data[0], gnormd.data[0])
            # gnorm = sum((p.grad**2).sum() for p in itertools.chain(*[
            #     m.parameters(dx2=True) for m in (self.d, self.g)]))
            if self.opts.grad_reg:
                (gnorm * 1000.0).backward()
            self.optim_g.step()
            self.optim_d.step()

            acc_gen, acc_oracle = self._compute_test_acc()
            test_nll = self._compute_test_nll()
            logging.info(
                f'[{epoch}] nll: {test_nll:.3f}  '
                f'acc: oracle={acc_oracle:.2f} gen={acc_gen:.2f}  '
                f'gnorm: {gnorm.data[0]:.2f}')
