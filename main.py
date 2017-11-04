"""Main training script."""

from contextlib import contextmanager
import argparse
import os
import pickle
import logging

import torch

import common
from common import PHASES, G_ML, D_ML, ADV, RUN_DIR
import environ


def main():
    """Trains the model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=environ.ENVS, default=environ.QA)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    init_opts, remaining_opts = parser.parse_known_args()

    opts_file = os.path.join(RUN_DIR, 'opts.pkl')
    if init_opts.resume:
        new_opts = environ.parse_env_opts(
            init_opts, remaining_opts, no_defaults=True)
        opts = argparse.Namespace(**common.unpickle(opts_file))
        for k, v in vars(new_opts).items():
            if v is not None:
                setattr(opts, k, v)
    else:
        opts = environ.parse_env_opts(init_opts, remaining_opts)
        os.mkdir(RUN_DIR)
        with open(opts_file, 'wb') as f_opts:
            pickle.dump(vars(opts), f_opts)

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logger = logging.getLogger()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    env = environ.create(opts.env, opts)

    for phase in PHASES:
        logger.debug(f'Beginning phase: {phase}')
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        with _phase(env, phase) as phase_runner:
            phase_runner()


@contextmanager
def _phase(env, phase):
    phase_dir = os.path.join(RUN_DIR, phase)
    if not os.path.isdir(phase_dir):
        os.mkdir(phase_dir)

    snap_file = os.path.join(phase_dir, 'state.pth')
    if os.path.isfile(snap_file):
        env.state = torch.load(snap_file)
        yield lambda: None
        return

    if phase == G_ML:
        runner = env.pretrain_g
    elif phase == D_ML:
        runner = env.pretrain_d
    elif phase == ADV:
        runner = env.train_adv

    logger = logging.getLogger()
    file_logger = logging.FileHandler(os.path.join(phase_dir, 'log.txt'))
    file_logger.setLevel(logging.INFO)
    logger.addHandler(file_logger)

    yield runner

    torch.save(env.state, snap_file)
    logger.removeHandler(file_logger)


if __name__ == '__main__':
    main()
