"""A class for training a SeqGAN model."""

import argparse
import logging
import os
import time

import torch
from torch.nn import functional as nnf
from torch.autograd import Variable

import common
from common import LABEL_GEN, LABEL_REAL
import dataset
from dataset import samplers
import model


class Environment(object):
    """A base class for training a SeqGAN model."""

    _EVAL_METRIC = None  # string describing generator eval metric

    _STATEFUL = ('hasher', 'g', 'd', 'optim_g', 'optim_d')

    def __init__(self, opts):
        """Creates an Environment."""

        opts.num_hash_buckets = opts.num_hash_buckets or 2**opts.code_len
        self.opts = opts

        torch.nn._functions.rnn.force_unfused = opts.grad_reg  # pylint: disable=protected-access

        self.train_dataset = self.test_dataset = None  # `Dataset`s of real data

        self.g = model.generator.create(**vars(opts)).cuda()
        self.d = model.discriminator.create(**vars(opts)).cuda()

        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=opts.lr_g)
        self.optim_d = torch.optim.Adam(self.d.parameters(), lr=opts.lr_d)

        if opts.exploration_bonus:
            self.hasher = model.hasher.create(**vars(opts)).cuda()
            self.optim_hasher = torch.optim.Adam(self.hasher.parameters(),
                                                 lr=opts.lr_hasher)
            self.state_counter = model.hash_counter.HashCounter(
                self.hasher, opts.num_hash_buckets).cuda()

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
        parser.add_argument('--log-freq', default=1, type=int)

        # data
        parser.add_argument('--nworkers', default=4, type=int)

        # model
        parser.add_argument('--d-type', choices=model.discriminator.TYPES,
                            default=model.discriminator.RNN)
        parser.add_argument('--vocab-size', type=int)
        parser.add_argument('--load-w2v', type=os.path.abspath)
        parser.add_argument('--g-tok-emb-dim', type=int)
        parser.add_argument('--d-tok-emb-dim', type=int)
        parser.add_argument('--rnn-dim', type=int)
        parser.add_argument('--num-gen-layers', default=1, type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--seqlen', type=int)
        parser.add_argument('--batch-size', type=int)
        parser.add_argument('--num-filters',
                            default=[100] + [200]*4 + [100]*5 + [160]*2,
                            nargs='+', type=int)
        parser.add_argument('--filter-widths',
                            default=list(range(1, 11)) + [15, 20],
                            nargs='+', type=int)
        parser.add_argument('--exploration-bonus', default=0, type=float)
        parser.add_argument('--code-len', type=int)
        parser.add_argument('--num-hash-buckets',
                            type=lambda x: 2**int(
                                torch.np.round(torch.np.log2(float(x)))))

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
        parser.add_argument('--hasher-ent-reg', default=1e-1, type=float)
        parser.add_argument('--grad-reg', default=0, type=float)
        parser.add_argument('--temperature', default=1, type=float)
        parser.add_argument('--rbuf-size', default=100, type=int)

        return parser

    @property
    def state(self):
        """Returns a dict containing the state of this Environment."""
        return {item: getattr(self, item).state_dict()
                for item in self._STATEFUL if hasattr(self, item)}

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

    def _compute_eval_metric(self):
        """Returns a number by which the generative model should be evaluate."""
        raise NotImplementedError()

    def _forward_seq2seq(self, fwd_fn, batch, volatile=False):
        """
        batch: (toks, labels); assumes toks is N*(seqlen + 1) with init toks

        Returns: loss, <whatever is returned by fwd_fn>
        """
        toks = Variable(batch[0], volatile=volatile).cuda()
        flat_tgts = toks[:, 1:].t().contiguous().view(-1)

        output = fwd_fn(toks)
        if not isinstance(output, (list, tuple)):
            output = (output,)
        gen_log_probs = output[0]
        flat_gen_log_probs = gen_log_probs.view(-1, gen_log_probs.size(-1))
        loss = nnf.nll_loss(flat_gen_log_probs, flat_tgts,
                            ignore_index=self.opts.padding_idx)
        return (loss, *output)

    def train_hasher(self, hook=None):
        """Train an auto-encoder on the dataset for use in hashing."""
        self.hasher.train()
        train_loader = self._create_dataloader(self.train_dataset)
        test_loader = self._create_dataloader(self.test_dataset)

        for epoch in range(1, self.opts.train_hasher_epochs + 1):
            tick = time.time()
            train_loss = train_entropy = 0
            for batch in train_loader:
                loss, _, code_logits = self._forward_seq2seq(self.hasher, batch)
                train_loss += loss.data[0]

                entropy = self._get_entropy(code_logits)[0]
                loss -= entropy * self.opts.hasher_ent_reg
                train_entropy += entropy.data[0]

                self.optim_hasher.zero_grad()
                loss.backward()
                self.optim_hasher.step()
            train_loss /= len(train_loader)
            train_entropy /= len(train_loader)

            test_loss = sum(self._forward_seq2seq(self.hasher, batch,
                                                  volatile=True)[0].data[0]
                            for batch in test_loader) / len(test_loader)
            logging.info(
                f'[{epoch:02d}] '
                f'loss: train={train_loss:.3f} test={test_loss:.3f}  '
                f'H: {train_entropy:.3f}  '
                f'({time.time() - tick:.1f})')

            if callable(hook):
                hook(self, epoch)

    def pretrain_g(self):
        """Pretrains G using maximum-likelihood."""
        logging.info(f'[00] {self._EVAL_METRIC}: '
                     f'{self._compute_eval_metric():.3f}')
        train_loader = self._create_dataloader(self.train_dataset)
        for epoch in range(1, self.opts.pretrain_g_epochs + 1):
            tick = time.time()
            train_loss = entropy = gnorm = 0
            for batch in train_loader:
                loss, gen_log_probs = self._forward_g_ml(batch)
                entropy += self._get_entropy(gen_log_probs)[1].data[0]
                train_loss += loss.data[0]

                self.optim_g.zero_grad()
                loss.backward()
                gnorm += self._get_grad_norm(self.g).data[0]
                self.optim_g.step()
            train_loss /= len(train_loader)
            entropy /= len(train_loader)
            gnorm /= len(train_loader)

            logging.info(
                f'[{epoch:02d}] loss: {train_loss:.3f}  '
                f'{self._EVAL_METRIC}: {self._compute_eval_metric():.3f}  '
                f'H: {entropy:.2f}  '
                f'gnorm: {self._get_grad_norm(self.g).data[0]:.2f}  '
                f'({time.time() - tick:.1f})')

    def _forward_g_ml(self, batch, volatile=False):
        """
        batch: (toks: N*T, labels: N)
        Returns: (tok_probs: T*N*V, next_state)
        """
        return self._forward_seq2seq(lambda toks: self.g(toks[:, :-1])[0],
                                     batch, volatile=volatile)

    def pretrain_d(self):
        """Pretrains D using pretrained G."""
        for epoch in range(1, self.opts.pretrain_d_epochs+1):
            tick = time.time()
            gen_dataset = self._create_gen_dataset(self.g, LABEL_GEN,
                                                   seed=self.opts.seed+epoch)
            dataloader = self._create_dataloader(torch.utils.data.ConcatDataset(
                (self.train_dataset, gen_dataset)))

            train_loss = 0
            for batch in dataloader:
                loss, _ = self._forward_d(batch)
                train_loss += loss.data[0]

                self.optim_d.zero_grad()
                loss.backward()
                self.optim_d.step()
            train_loss /= len(dataloader)

            acc_real, acc_gen = self._compute_d_test_acc()
            logging.info(f'[{epoch:02d}] loss: {train_loss:.3f}  '
                         f'acc: real={acc_real:.2f} gen={acc_gen:.2f}  '
                         f'({time.time() - tick:.1f})')

    def _forward_d(self, batch, volatile=False, has_init=True):
        toks, labels = batch
        toks = Variable(toks, volatile=volatile).cuda()
        labels = Variable(labels, volatile=volatile).cuda()
        d_log_probs = self.d(toks[:, has_init:])
        return nnf.nll_loss(d_log_probs, labels), d_log_probs

    def _compute_d_test_acc(self, num_samples=256):
        num_test_batches = max(num_samples // len(self.init_toks), 1)

        test_loader = self._create_dataloader(self.test_dataset)
        acc_real = 0
        for i, (batch_toks, _) in enumerate(test_loader):
            if i == num_test_batches:
                break
            toks = Variable(batch_toks[:, 1:].cuda())  # no init toks
            acc_real += self.compute_acc(self.d(toks), LABEL_REAL)
        acc_real /= num_test_batches

        acc_gen = 0
        with common.rand_state(torch.cuda, -1):
            for _ in range(num_test_batches):
                gen_seqs, _ = self.g.rollout(self.init_toks, self.opts.seqlen)
                acc_gen += self.compute_acc(self.d(gen_seqs), LABEL_GEN)
        acc_gen /= num_test_batches

        return acc_real, acc_gen

    def compute_acc(self, probs, label):
        """Computes the accuracy given prob Variable and and label."""
        self._labels.fill_(label)
        probs = probs.data if isinstance(probs, Variable) else probs
        return (probs.max(1)[1] == self._labels).float().mean()

    def train_adv(self):
        """Adversarially train G against D."""

        self.optim_g.param_groups[0]['lr'] *= 0.1
        # self.optim_d.param_groups[0]['lr'] *= 0.1
        if self.opts.exploration_bonus:
            self.hasher.eval()
            self._init_state_counter()

        real_dataloader = iter(
            self._create_dataloader(self.train_dataset, cycle=True))

        replay_buffer, rbuf_loader = self._create_replay_buffer(
            self.opts.rbuf_size, LABEL_GEN)
        replay_buffer_iter = None

        for i in range(1, self.opts.adv_train_iters+1):
            tick = time.time()

            loss_g, gen_seqs, entropy_g = self._train_adv_g(replay_buffer)
            self.optim_g.zero_grad()
            loss_g.backward(create_graph=bool(self.opts.grad_reg))
            if not self.opts.grad_reg:
                self.optim_g.step()

            if replay_buffer_iter is None:
                replay_buffer_iter = iter(rbuf_loader)

            loss_d = self._train_adv_d(gen_seqs, real_dataloader,
                                       replay_buffer_iter)
            self.optim_d.zero_grad()
            loss_d.backward(create_graph=bool(self.opts.grad_reg))

            gnormg, gnormd = map(self._get_grad_norm, (self.g, self.d))
            gnorm = (gnormg * (self.opts.grad_reg * 50.) +
                     gnormd * (self.opts.grad_reg * 0.1))
            if self.opts.grad_reg:
                gnorm.backward()
                self.optim_g.step()
            self.optim_d.step()

            if (i-1) % self.opts.log_freq == 0:
                acc_oracle, acc_gen = self._compute_d_test_acc()
                logging.info(
                    f'[{i:03d}] '
                    f'{self._EVAL_METRIC}: {self._compute_eval_metric():.3f}  '
                    f'acc: o={acc_oracle:.2f} g={acc_gen:.2f}  '
                    f'gnorm: g={gnormg.data[0]:.2f} d={gnormd.data[0]:.2f}  '
                    f'H: {entropy_g.data[0]:.2f}  '
                    f'({time.time() - tick:.1f})')

    def _init_state_counter(self):
        self.state_counter.train()
        for toks, _ in self._create_dataloader(self.train_dataset):
            self.state_counter(Variable(toks.cuda(), volatile=True))
        self.state_counter.eval()

    def _train_adv_g(self, replay_buffer):
        losses = []
        entropies = []
        for i in range(self.opts.adv_g_iters):
            # train G
            gen_seqs, gen_log_probs = self.g.rollout(
                self.init_toks, self.opts.seqlen,
                temperature=self.opts.temperature)
            gen_seqs = torch.cat(gen_seqs, -1)  # N*T
            if i == 0:
                replay_buffer.add_samples(gen_seqs)

            gen_log_probs = torch.stack(gen_log_probs)  # T*N*V
            seq_log_probs = gen_log_probs.transpose(0, 1).gather(  # N*T
                -1, gen_seqs.unsqueeze(-1)).squeeze(-1)

            advantages = self._get_advantages(gen_seqs)  # N*T
            score = (seq_log_probs * advantages).sum(1).mean()

            disc_entropy, entropy = self._get_entropy(
                gen_log_probs, discount_rate=self.opts.discount)

            # _, roomtemp_lprobs = self.g.rollout(
            #     self.init_toks, self.opts.seqlen, temperature=1)
            # roomtemp_lprobs = torch.stack(roomtemp_lprobs)
            # _, entropy = self._get_entropy(roomtemp_lprobs)

            entropies.append(entropy)
            losses.append(-score - disc_entropy * self.opts.g_ent_reg)

        loss = sum(losses)
        avg_entropy = sum(entropies) / len(entropies)
        return loss, gen_seqs, avg_entropy

    def _get_advantages(self, gen_seqs):
        rep_gen_seqs = gen_seqs.repeat(max(1, self.opts.num_rollouts), 1)
        qs_g = self._get_qs(self.g, rep_gen_seqs)  # N*T

        advs = qs_g  # something clever like PPO would be inserted here

        # advs = advs[:, self._inv_idx].cumsum(1)[:, self._inv_idx]  # adv to go
        advs -= advs.mean(1, keepdim=True)
        advs /= advs.std(1, keepdim=True)
        return advs.detach()

    def _get_qs(self, g_ro, rep_gen_seqs):
        rep_gen_seqs.volatile = True

        qs = torch.cuda.FloatTensor(
            self.opts.seqlen, self.opts.batch_size).zero_()
        bonus = torch.cuda.FloatTensor(1, 1).zero_().expand_as(qs)

        gen_seqs = rep_gen_seqs[:self.opts.batch_size]
        qs[-1] = self.d(gen_seqs)[:, LABEL_REAL].data

        if self.opts.exploration_bonus:
            # bonus = self._get_exploration_bonus(gen_seqs).repeat(  # T*N
            #     self.opts.seqlen, 1)
            bonus = bonus.contiguous()
            bonus[-1] = self._get_exploration_bonus(gen_seqs)

        if self.opts.num_rollouts == 0:
            return Variable(qs.t().exp_().add_(bonus.t()))

        ro_rng = torch.cuda.get_rng_state()
        _, ro_hid = g_ro(self.ro_init_toks)
        for n in range(1, self.opts.seqlen):
            # ro_suff, _  = g_ro.rollout(rep_gen_seqs[:,:n], self.opts.seqlen-n)

            torch.cuda.set_rng_state(ro_rng)
            ro_state = (rep_gen_seqs[:, n-1].unsqueeze(-1), ro_hid)
            ro_suffix, _, (ro_hid, ro_rng) = g_ro.rollout(
                ro_state, self.opts.seqlen - n, return_first_state=True)
            ro = torch.cat([rep_gen_seqs[:, :n]] + ro_suffix, -1)
            assert ro.size(1) == self.opts.seqlen

            qs[n-1] = self._ro_mean(self.d(ro), (-1, 2))[:, LABEL_REAL].data
            # LABEL_G gives cost, LABEL_REAL gives reward
            # if self.opts.exploration_bonus:
            #     bonus[n-1] = self._ro_mean(self._get_exploration_bonus(ro))

        return Variable(qs.t().exp_().add_(bonus.t()))

    def _ro_mean(self, t, sizes=(-1,)):
        """Averages a tensor over rollouts.
        t: (num_rollouts*N)*sizes

        Returns: N*sizes
        """
        return t.view(self.opts.num_rollouts, *sizes).mean(0)

    def _get_exploration_bonus(self, gen_seqs):
        seq_buckets = self.state_counter(
            Variable(gen_seqs.data, volatile=True))
        reachable = self.state_counter.counts_train
        visit_counts = self.state_counter.counts[seq_buckets]
        bonus_weights = (reachable + 0.1).log() * self.opts.exploration_bonus
        return bonus_weights[seq_buckets] / visit_counts**0.5

    def _train_adv_d(self, gen_seqs, real_dataloader, replay_buffer_iter):
        REAL_W = 0.5
        GEN_W = (1 - REAL_W)
        RBUF_W = 0.5

        loss_d, d_log_probs = self._forward_d(next(real_dataloader))
        loss_d *= REAL_W
        entropy_d = REAL_W * self._get_entropy(d_log_probs)[0]

        n_rbuf_batches = min(len(replay_buffer_iter) // self.opts.batch_size, 4)
        cur_w = GEN_W
        if n_rbuf_batches:
            cur_w *= RBUF_W
            rbuf_batch_w = GEN_W * RBUF_W / n_rbuf_batches

        self._labels.fill_(LABEL_GEN)
        loss_d_g, d_log_probs = self._forward_d(
            (gen_seqs.data, self._labels), has_init=False)

        loss_d += cur_w * loss_d_g
        entropy_d += cur_w * self._get_entropy(d_log_probs)[0]

        for _ in range(n_rbuf_batches):
            loss_d_g, d_log_probs = self._forward_d(
                next(replay_buffer_iter), has_init=False)
            loss_d += rbuf_batch_w * loss_d_g
            entropy_d += rbuf_batch_w * self._get_entropy(d_log_probs)[0]

        return loss_d - entropy_d * self.opts.d_ent_reg

    def _create_gen_dataset(self, gen, label, num_samples=None, seed=None):
        num_samples = num_samples or len(self.train_dataset)
        seed = seed or self.opts.seed
        return dataset.GenDataset(generator=gen,
                                  label=label,
                                  seqlen=self.opts.seqlen,
                                  seed=seed,
                                  gen_init_toks=self.ro_init_toks,
                                  num_samples=num_samples,
                                  eos_idx=self.opts.eos_idx)

    def _create_dataloader(self, src_dataset, cycle=False):
        dl_opts = {'batch_size': self.opts.batch_size,
                   'num_workers': self.opts.nworkers,
                   'pin_memory': True,
                   'shuffle': not cycle}
        if cycle:
            dl_opts['sampler'] = samplers.InfiniteRandomSampler(src_dataset)
        return torch.utils.data.DataLoader(src_dataset, **dl_opts)

    def _create_replay_buffer(self, max_history, label):
        replay_buffer = dataset.ReplayBuffer(max_history, label)
        sampler = samplers.ReplayBufferSampler(replay_buffer,
                                               self.opts.batch_size)
        loader = torch.utils.data.DataLoader(replay_buffer,
                                             batch_sampler=sampler,
                                             num_workers=0,  # TODO
                                             pin_memory=True)
        return replay_buffer, loader

    @staticmethod
    def _get_grad_norm(mod):
        return sum((p.grad**2).sum() for p in mod.parameters(dx2=True))

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
