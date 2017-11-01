"""Main training script for the model."""

import argparse
import copy
import os
import pickle
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable

from common import PHASES, G_ML, D_ML, ADV, LABEL_G, LABEL_O
import common
import model
import dataset


GEN_BATCHES = 8


def _create_oracle(opts):
    """Returns a randomly initialized generator."""
    orig_rand_state = torch.get_rng_state()
    torch.manual_seed(opts.oracle_seed)
    oracle = model.Generator(word_emb_dim=opts.g_word_emb_dim, **vars(opts))
    for param in oracle.parameters():
        nn.init.normal(param, std=1)
    torch.set_rng_state(orig_rand_state)
    return oracle


def main():
    parser = argparse.ArgumentParser()

    n_gpu = torch.cuda.device_count()

    # general
    parser.add_argument('--oracle-seed', default=42, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--debug', action='store_true')

    # data
    parser.add_argument('--num-gen-samps', default=10000, type=int)
    parser.add_argument('--nworkers', default=6, type=int)

    # model
    parser.add_argument('--vocab-size', default=5000, type=int)
    parser.add_argument('--g-word-emb-dim', default=32, type=int)
    parser.add_argument('--d-word-emb-dim', default=64, type=int)
    parser.add_argument('--gen-dim', default=32, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--seqlen', default=20, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--mix-batches', default=16, type=int)
    parser.add_argument('--num-filters',
                        default=[100] + [200]*4 + [100]*5 + [160]*2,
                        nargs='+', type=int)
    parser.add_argument('--filter-widths',
                        default=list(range(1, 11)) + [15, 20],
                        nargs='+', type=int)

    # training
    parser.add_argument('--lr-g', default=0.01, type=float)
    parser.add_argument('--lr-d', default=0.001, type=float)
    parser.add_argument('--pretrain-g-epochs', default=50, type=int)
    parser.add_argument('--pretrain-d-epochs', default=150, type=int)
    parser.add_argument('--adv-train-iters', default=150, type=int)
    parser.add_argument('--adv-g-iters', default=150, type=int)
    parser.add_argument('--adv-d-iters', default=5, type=int)
    parser.add_argument('--adv-d-epochs', default=3, type=int)
    parser.add_argument('--num-rollouts', default=16, type=int)
    # parser.add_argument('--resume', type=int)

    # output
    parser.add_argument('--dispfreq', default=10, type=int)
    parser.add_argument('--exp-name', '-o', default='seqgan')

    opts = parser.parse_args()
    opts.n_gpu = n_gpu
    opts.synth = True
    assert len(opts.filter_widths) == len(opts.num_filters)
    # --------------------------------------------------------------------------

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    rundir = os.path.join(common.RUN_DIR, opts.exp_name)
    for phase in PHASES:
        os.makedirs(os.path.join(rundir, phase), exist_ok=True)

    phase_dir = phase_log = None
    def _set_phase(phase):
        nonlocal phase_log, phase_dir
        phase_dir = os.path.join(rundir, phase)
        if phase_log:
            phase_log.close()
        phase_log = None
        with open(os.path.join(phase_dir, common.OPTS_FILE), 'w') as f_opts:
            pprint.pprint(vars(opts), stream=f_opts)

    last_log = ''
    def _log(txt):
        nonlocal phase_log, last_log
        if not phase_log:
            phase_log = open(os.path.join(phase_dir, 'log.txt'), 'w')
        print(txt)
        print(txt, file=phase_log, flush=True)
        last_log = txt

    g = model.Generator(word_emb_dim=opts.g_word_emb_dim, **vars(opts)).cuda()
    d = model.Discriminator(word_emb_dim=opts.d_word_emb_dim, **vars(opts)).cuda()
    oracle = _create_oracle(opts).cuda()

    optim_g = torch.optim.Adam(g.parameters(), lr=opts.lr_g)
    optim_d = torch.optim.Adam(d.parameters(), lr=opts.lr_d)

    def _gen2cpu(gen):
        cpu_gen = common.create_generator(opts)
        cpu_gen.load_state_dict(common.state2cpu(gen.state_dict()))
        return cpu_gen

    ds_opts = {'seqlen': opts.seqlen,
               'batch_size': opts.batch_size // opts.mix_batches,
               'num_samples': opts.num_gen_samps}
    dl_opts = {'batch_size': opts.mix_batches,
               'pin_memory': True,
               'shuffle': True,
               'num_workers': opts.nworkers}

    oracle_ds = dataset.SynthDataset(generator=_gen2cpu(oracle), label=LABEL_O,
                                     seed=opts.oracle_seed, **ds_opts)
    oracle_ds_loader = torch.utils.data.DataLoader(oracle_ds, **dl_opts)

    init_toks = Variable(
        torch.LongTensor(opts.batch_size, 1).cuda().fill_(1))

    oracle_data = zip(*(
        oracle_ds[-i] for i in range(1, opts.mix_batches+1)))
    oracle_test_toks, oracle_labels = common.ship_batch(
        list(map(lambda x: torch.cat(x, 0), oracle_data)), True)
    gen_labels = oracle_labels * 0

    def _compute_test_nll(n_samples=128):
        test_nll = 0
        n_test_batches = n_samples // len(init_toks)
        with common.rand_state(torch.cuda, opts.seed):
            for i in range(n_test_batches):
                gen_seqs, _ = g.rollout(init_toks, opts.seqlen)
                test_nll += common.compute_oracle_nll(oracle, gen_seqs)
        test_nll /= n_test_batches
        return test_nll

    _set_phase(G_ML)
    if os.path.isfile(os.path.join(phase_dir, 'model.pth')):
        g.load_state_dict(torch.load(os.path.join(phase_dir, 'model.pth')))
        optim_g.load_state_dict(
            torch.load(os.path.join(phase_dir, 'optim_state.pth')))
    else:
        _log('# generator pre-training')
        for epoch in range(1, opts.pretrain_g_epochs+1):
            epoch_loss = 0
            for toks, _ in oracle_ds_loader:
                toks = Variable(toks.view(-1, toks.size(-1))).cuda()
                flat_tgts = toks[:, 1:].t().contiguous().view(-1)

                gen_probs, _ = g(toks[:, :-1])
                flat_gen_probs = gen_probs.view(-1, gen_probs.size(-1))
                loss = nnf.nll_loss(flat_gen_probs, flat_tgts)
                epoch_loss += loss.data[0]

                optim_g.zero_grad()
                loss.backward()
                optim_g.step()

            test_nll = _compute_test_nll()
            epoch_loss /= len(oracle_ds_loader)
            _log(f'[{epoch}] loss: {epoch_loss:.3f}  nll: {test_nll:.3f}')

        torch.save(g.state_dict(), os.path.join(phase_dir, 'model.pth'))
        torch.save(optim_g.state_dict(), os.path.join(phase_dir, 'optim_state.pth'))

    _set_phase(D_ML)
    if os.path.isfile(os.path.join(phase_dir, 'model.pth')):
        d.load_state_dict(torch.load(os.path.join(phase_dir, 'model.pth')))
        optim_d.load_state_dict(
            torch.load(os.path.join(phase_dir, 'optim_state.pth')))
    else:
        gen_ds = dataset.SynthDataset(generator=_gen2cpu(g), label=LABEL_G,
                                      seed=opts.seed, **ds_opts)

        d_pretrain_ds = torch.utils.data.ConcatDataset((oracle_ds, gen_ds))
        d_pretrain_loader = torch.utils.data.DataLoader(d_pretrain_ds,
                                                        **dl_opts)

        _log('# discriminator pre-training')
        for epoch in range(1, opts.pretrain_d_epochs+1):
            epoch_loss = 0
            for batch in d_pretrain_loader:
                toks, labels = common.ship_batch(batch)

                d_probs = d(toks[:, 1:])
                loss = nnf.nll_loss(d_probs, labels)
                epoch_loss += loss.data[0]

                optim_d.zero_grad()
                loss.backward()
                optim_d.step()

            gen_seqs, _ = g.rollout(init_toks, opts.seqlen)
            d_acc_oracle = common.compute_acc(d(oracle_test_toks), oracle_labels)
            d_acc_gen = common.compute_acc(d(gen_seqs), gen_labels)

            _log(f'[{epoch}] loss: {epoch_loss:.3f}  '
                 f'acc_oracle: {d_acc_oracle:.2f}  acc_gen: {d_acc_gen:.2f}')

        torch.save(d.state_dict(), os.path.join(phase_dir, 'model.pth'))
        torch.save(optim_d.state_dict(), os.path.join(phase_dir, 'optim_state.pth'))

    _set_phase(ADV)

    _log('# adversarial training')
    rand_state = opts.seed
    g_ro = common.create_generator(opts).cuda()
    g_ro.load_state_dict(g.state_dict())
    g_ro = g #oracle

    qs_zeros = torch.zeros(opts.seqlen, init_toks.size(0)).cuda()
    if opts.num_rollouts > 0:
        ro_init_toks = init_toks.repeat(opts.num_rollouts, 1)

    try:
        for epoch in range(1, opts.adv_train_iters+1):

            # train G
            for i in range(1, opts.adv_g_iters+1):
                gen_seqs, gen_probs = g.rollout(init_toks, opts.seqlen)
                gen_probs = torch.stack(gen_probs)  # T*N

                # compute Q values
                qs_zeros.zero_()
                qs = Variable(qs_zeros)

                if opts.num_rollouts > 0:
                    rep_gen_seqs = [gs.repeat(opts.num_rollouts, 1)
                                    for gs in gen_seqs]
                    ro_rng = torch.cuda.get_rng_state()
                    _, ro_hid = g_ro(ro_init_toks)
                    for n in range(1, opts.seqlen):
                        ro_state = (rep_gen_seqs[n-1], ro_hid)
                        ro_seqs, ro_probs, (ro_hid, ro_rng) = g_ro.rollout(
                            ro_state, opts.seqlen - n, return_first_state=True)
                        full_ro = torch.cat(rep_gen_seqs[:n] + ro_seqs, -1)
                        assert full_ro.size(1) == opts.seqlen

                        q = d(full_ro)[:, LABEL_O].exp()
                        # LABEL_G gives cost, LABEL_O gives reward
                        qs[n-1] = q.view(opts.num_rollouts, -1).mean(0)

                        torch.cuda.set_rng_state(ro_rng)
                else:
                    qs = qs[-1].unsqueeze(0)

                cat_gen_seqs = torch.cat(gen_seqs, -1)
                qs[-1] = d(cat_gen_seqs)[:, LABEL_O].exp()
                qs = qs.detach()

                # TODO: reward-to-go

                gen_seq_probs = gen_probs.gather(  # T*N*V -> T*N
                    -1, cat_gen_seqs.t().unsqueeze(-1)).squeeze(-1)
                qs -= qs.mean()  # TODO: learned baseline
                loss = -(qs * gen_seq_probs).sum(0).mean()

                optim_g.zero_grad()
                loss.backward()
                optim_g.step()

                if i % 10 == 0:
                    # acc_g should go to zero, acc_oracle should remain unchanged
                    gen_seqs, _ = g.rollout(init_toks, opts.seqlen)
                    d_acc_gen = common.compute_acc(d(gen_seqs), gen_labels)
                    d_acc_oracle = common.compute_acc(
                        d(oracle_test_toks), oracle_labels)
                    test_nll = _compute_test_nll(128)
                    _log(f'[{epoch}] (G{i}) nll: {test_nll:.3f}  '
                         f'acc_oracle: {d_acc_oracle:.2f}  acc_gen: {d_acc_gen:.2f}')

            g_ro.load_state_dict(g.state_dict())  # rollout policy <- policy

            # train D
            gen_ds = dataset.SynthDataset(generator=_gen2cpu(g), label=LABEL_G,
                                          seed=opts.seed, **ds_opts)
            d_adv_ds = torch.utils.data.ConcatDataset((oracle_ds, gen_ds))
            d_adv_loader = torch.utils.data.DataLoader(d_adv_ds, **dl_opts)

            for i in range(1, opts.adv_d_iters+1):
                for j in range(1, opts.adv_d_epochs+1):
                    for k, batch in enumerate(d_adv_loader):
                        toks, labels = common.ship_batch(batch)

                        d_probs = d(toks[:, 1:])
                        loss = nnf.nll_loss(d_probs, labels)

                        optim_d.zero_grad()
                        loss.backward()
                        optim_d.step()

                    gen_seqs, _ = g.rollout(init_toks, opts.seqlen)
                    d_acc_gen = common.compute_acc(d(gen_seqs), gen_labels)
                    d_acc_oracle = common.compute_acc(
                        d(oracle_test_toks), oracle_labels)
                    test_nll = _compute_test_nll(128)

                    _log(f'[{epoch}] (D{i}-{j}) nll: {test_nll:.3f}  '
                         f'acc_oracle: {d_acc_oracle:.2f}  acc_gen: {d_acc_gen:.2f}')

                gen_ds.advance()
                oracle_ds.advance()
    except KeyboardInterrupt:
        snap_dir_name = input('\nsnap dir name: ')
        if snap_dir_name:
            snap_dir = os.path.join(phase_dir, snap_dir_name)
            os.makedirs(snap_dir, exist_ok=True)
            torch.save(g.state_dict(), os.path.join(snap_dir, 'g_state.pth'))
            torch.save(d.state_dict(), os.path.join(snap_dir, 'd_state.pth'))
            torch.save(oracle.state_dict(), os.path.join(snap_dir, 'oracle_state.pth'))
            with open(os.path.join(snap_dir, 'opts.pkl'), 'wb') as f_opts:
                pickle.dump(vars(opts), f_opts)
            with open(os.path.join(snap_dir, 'last_log.txt'), 'w') as f_ll:
                print(last_log, file=f_ll, flush=True)
            print(f'Saved model to {snap_dir}')


if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     pass
