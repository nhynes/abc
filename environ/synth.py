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
        parser.add_argument('--grad-reg', default=0, type=float)
        parser.add_argument('--use-oracle-w2v', action='store_true')
        parser.set_defaults(
            num_gen_samps=100000,
            seqlen=20,
            vocab_size=5000,
            g_word_emb_dim=32,
            d_word_emb_dim=32,
            pretrain_g_epochs=50,  # try 10 when using oracle w2v
            pretrain_d_epochs=10,
            adv_train_iters=750,
            gen_dim=32,
            dropout=0.25,
            num_gen_layers=1,
            lr_g=0.01,
            lr_d=0.001,
            )
        return parser

    def __init__(self, opts):
        """Creates a SynthEnvironment."""
        torch.nn._functions.rnn.force_unfused = opts.grad_reg

        super(SynthEnvironment, self).__init__(opts)

        self.ro_init_toks.data.zero_()

        self.oracle = self._create_oracle().cuda()
        self.oracle_dataset = self._create_dataset(self.oracle, LABEL_REAL)

        if self.opts.use_oracle_w2v:
            for net in (self.g, self.d, self.tgt_g):
                net.word_emb = model.utils.DontTrain(self.oracle.word_emb)

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
                                  gen_init_toks=self.ro_init_toks,
                                  num_samples=num_samples)

    def compute_oracle_nll(self, toks, return_probs=False):
        """
        oracle: a Generator
        toks: [N]*T
        """
        if isinstance(toks, list):
            toks = torch.cat(toks).view(len(toks), -1)  # T*N
        gen_log_probs = self.get_tok_log_probs(self.oracle, toks.t())
        flat_log_probs = gen_log_probs.view(-1, gen_log_probs.size(-1))  # T*N*V
        nll = nnf.nll_loss(flat_log_probs, toks.view(-1)).data[0]
        if return_probs:
            return nll, gen_log_probs
        return nll

    def _compute_test_nll(self, num_samples=256):
        test_nll = 0
        num_test_batches = max(num_samples // len(self.init_toks), 1)
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                test_nll += self.compute_oracle_nll(gen_seqs)
        test_nll /= num_test_batches
        return test_nll

    def _compute_test_acc(self, num_samples=256):
        acc_gen = acc_oracle = 0
        num_test_batches = max(num_samples // len(self.init_toks), 1)
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                acc_gen += self.compute_acc(self.d(gen_seqs), LABEL_GEN)
                acc_oracle += self.compute_acc(
                    self.d(self.oracle_test_toks), LABEL_REAL)
        acc_gen /= num_test_batches
        acc_oracle /= num_test_batches
        return acc_gen, acc_oracle

    @staticmethod
    def _get_entropy(log_probs, discount_rate=None):
        # assumes distributions are along the last dimension
        infos = log_probs.exp() * log_probs
        if discount_rate:
            entropy_undiscounted = -infos.sum(-1).mean()
            sz = [log_probs.size(0)] + [1]*(log_probs.ndimension() - 1)
            discount = log_probs.data.new(*sz).fill_(1)
            discount[1:] *= discount_rate
            discount.cumprod(0, out=discount)
            infos = infos * Variable(discount)
            return -infos.sum(-1).mean(), entropy_undiscounted
        return -infos.sum(-1).mean()

    def pretrain_g(self):
        """Pretrains G using maximum-likelihood on a synthetic dataset."""

        dataloader = self._create_dataloader(self.oracle_dataset)

        logging.info(f'[0] nll: {self._compute_test_nll():.3f}')
        for epoch in range(1, self.opts.pretrain_g_epochs + 1):
            train_loss = entropy = 0
            for batch in dataloader:
                loss, gen_log_probs = self._forward_g_pretrain(batch)
                entropy += self._get_entropy(gen_log_probs).data[0]
                train_loss += loss.data[0]

                self.optim_g.zero_grad()
                loss.backward()
                self.optim_g.step()
            train_loss /= len(dataloader)
            entropy /= len(dataloader)

            oracle_nll = self._compute_test_nll()
            logging.info(
                f'[{epoch}] loss: {train_loss:.3f}  nll: {oracle_nll:.3f}  '
                f'H: {entropy:.2f}')

    def pretrain_d(self):
        """Pretrains D using pretrained G."""

        for epoch in range(1, self.opts.pretrain_d_epochs+1):
            gen_dataset = self._create_dataset(self.g, LABEL_GEN,
                                               seed=self.opts.seed+epoch)
            dataloader = self._create_dataloader(torch.utils.data.ConcatDataset(
                (self.oracle_dataset, gen_dataset)))

            train_loss = 0
            gnorm = 0
            for batch in dataloader:
                loss, pred_log_probs = self._forward_d(batch)

                train_loss += loss.data[0]

                self.optim_d.zero_grad()
                loss.backward()
                gnorm += sum(
                    (p.grad.data**2).sum() for p in self.d.parameters(dx2=True))
                self.optim_d.step()
            train_loss /= len(dataloader)
            gnorm /= len(dataloader)

            acc_gen, acc_oracle = self._compute_test_acc()
            logging.info(f'[{epoch}] loss: {train_loss:.3f}  '
                         f'acc: oracle={acc_oracle:.2f}  gen={acc_gen:.2f}  '
                         f'gnorm: {gnorm:.2f}')

    def _get_qs(self, g_ro, rep_gen_seqs):
        qs = Variable(torch.cuda.FloatTensor(
            self.opts.seqlen, self.opts.batch_size).zero_())

        qs[-1] = self.d(rep_gen_seqs[:qs.size(1)])[:, LABEL_REAL]

        if self.opts.num_rollouts == 0:
            qs.data[:-1] = qs.data[None, -1].expand(qs.size(0) - 1, qs.size(1))
            qs.data.exp_()
            return qs.t().detach()

        ro_rng = torch.cuda.get_rng_state()
        _, ro_hid = g_ro(self.ro_init_toks)
        for n in range(1, self.opts.seqlen):
            # ro_seqs, _  = g_ro.rollout(rep_gen_seqs[:,:n], self.opts.seqlen-n)

            torch.cuda.set_rng_state(ro_rng)
            ro_state = (rep_gen_seqs[:, n-1].unsqueeze(-1), ro_hid)
            ro_seqs, _, (ro_hid, ro_rng) = g_ro.rollout(
                ro_state, self.opts.seqlen - n, return_first_state=True)
            full_ro = torch.cat([rep_gen_seqs[:, :n]] + ro_seqs, -1)
            assert full_ro.size(1) == self.opts.seqlen

            q = self.d(full_ro).view(self.opts.num_rollouts, -1, 2)
            # LABEL_G gives cost, LABEL_REAL gives reward
            qs[n-1] = q.mean(0)[:, LABEL_REAL]

        qs.data.exp_()
        return qs.t().detach()

    def _get_advantages(self, gen_seqs):
        rep_gen_seqs = gen_seqs.repeat(self.opts.num_rollouts, 1)
        qs_g = self._get_qs(self.g, rep_gen_seqs)

        advs = qs_g

        # advs = advs[:, self._inv_idx].cumsum(1)[:, self._inv_idx]  # adv to go
        advs -= advs.mean()
        advs /= advs.std()
        return advs.detach()

    @staticmethod
    def _get_grad_norm(mod):
        return sum((p.grad**2).sum() for p in mod.parameters(dx2=True))

    def _ds_iter(self, dataset, batch_size):
        batch_idxs_it = iter(())
        while True:
            try:
                yield dataset[next(batch_idxs_it)]
            except StopIteration:
                batch_idxs = torch.randperm(len(dataset)).split(batch_size)
                if len(dataset) % batch_size:
                    batch_idxs = batch_idxs[:-1]
                batch_idxs_it = iter(batch_idxs)

    def train_adv(self):
        """Adversarially train G against D."""

        self.optim_g.param_groups[0]['lr'] *= 0.1

        half_batch = self.opts.batch_size // 2
        oracle_ds_it = self._ds_iter(self.oracle_dataset, self.opts.batch_size)

        replay_buf = dataset.ReplayBuffer(100)

        for epoch in range(1, self.opts.adv_train_iters+1):
            self.optim_g.zero_grad()
            for i in range(2):
                # train G
                gen_seqs, gen_log_probs = self.g.rollout(
                    self.init_toks, self.opts.seqlen)
                gen_seqs = torch.cat(gen_seqs, -1)  # N*T
                if i == 0:
                    replay_buf.add_samples(gen_seqs)

                gen_log_probs = torch.stack(gen_log_probs)  # T*N*V
                seq_log_probs = self._gather_act_probs(gen_seqs, gen_log_probs)

                advantages = self._get_advantages(gen_seqs)  # N*T
                g_score = (seq_log_probs * advantages).sum(1).mean()

                disc_entropy_g, entropy_g = self._get_entropy(
                    gen_log_probs, discount_rate=self.opts.discount)

                loss_g = -g_score - disc_entropy_g * self.opts.g_ent_reg
                loss_g.backward(create_graph=bool(self.opts.grad_reg))
            if not self.opts.grad_reg:
                self.optim_g.step()

            # train D
            self.optim_d.zero_grad()

            REAL_WEIGHT = 0.5
            GEN_WEIGHT = (1 - REAL_WEIGHT)
            RBUF_WEIGHT = 0.5

            oracle_toks = next(oracle_ds_it)[0].cuda()
            labels = self._labels.clone().fill_(LABEL_REAL)
            loss_d, d_log_probs = self._forward_d((oracle_toks, labels))
            loss_d *= REAL_WEIGHT
            entropy_d = REAL_WEIGHT * self._get_entropy(d_log_probs)

            n_rbuf_batches = min(epoch - 1, 4)
            cur_weight = GEN_WEIGHT
            if n_rbuf_batches:
                cur_weight *= RBUF_WEIGHT
                rbuf_batch_weight = GEN_WEIGHT * RBUF_WEIGHT / n_rbuf_batches

            self._labels.fill_(LABEL_GEN)
            loss_d_g, d_log_probs = self._forward_d(
                (gen_seqs.data, self._labels), has_init=False)

            loss_d += cur_weight * loss_d_g
            entropy_d += cur_weight * self._get_entropy(d_log_probs)

            for _ in range(n_rbuf_batches):
                loss_d_g, d_log_probs = self._forward_d(
                    (replay_buf.get_samples(self.opts.batch_size).cuda(),
                     self._labels),
                    has_init=False)
                loss_d += rbuf_batch_weight * loss_d_g
                entropy_d += rbuf_batch_weight * self._get_entropy(d_log_probs)

            loss_d = loss_d - entropy_d * self.opts.d_ent_reg
            loss_d.backward(create_graph=bool(self.opts.grad_reg))

            gnormg, gnormd = map(self._get_grad_norm, (self.g, self.d))
            gnorm = (gnormg * (self.opts.grad_reg * 50.) +
                     gnormd * (self.opts.grad_reg * 0.1))
            # nn.utils.clip_grad_norm(self.g.parameters(), 5)
            # nn.utils.clip_grad_norm(self.d.parameters(), 1)
            if self.opts.grad_reg:
                gnorm.backward()
                self.optim_g.step()
            self.optim_d.step()

            acc_gen, acc_oracle = self._compute_test_acc()
            test_nll = self._compute_test_nll()
            logging.info(
                f'[{epoch}] nll: {test_nll:.3f}  '
                f'acc: oracle={acc_oracle:.2f} gen={acc_gen:.2f}  '
                # f'gnorm: {gnorm.data[0]:.2f}  '
                f'H: {entropy_g.data[0]:.2f}')
