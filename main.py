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
    parser.add_argument('--env', choices=environ.ENVS, default=environ.SYNTH)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log-prefix')
    init_opts, remaining_opts = parser.parse_known_args()

    opts_file = os.path.join(RUN_DIR, 'opts.pkl')
    if init_opts.resume:
        new_opts = environ.parse_env_opts(
            init_opts, remaining_opts, no_defaults=True)
        opts = argparse.Namespace(**common.unpickle(opts_file))
        for k, v in vars(new_opts).items():
            if k not in opts or v is not None:
                setattr(opts, k, v)
    else:
        opts = environ.parse_env_opts(init_opts, remaining_opts)
        os.mkdir(RUN_DIR)
        with open(opts_file, 'wb') as f_opts:
            pickle.dump(vars(opts), f_opts)

    logging.basicConfig(format='%(message)s', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    env = environ.create(opts.env, opts)

    for phase in PHASES:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        with _phase(env, phase, opts) as phase_runner:
            if phase_runner:
                logger.debug(f'# running phase: {phase}')
                phase_runner()


@contextmanager
def _phase(env, phase, opts):
    phase_dir = os.path.join(RUN_DIR, phase)
    if not os.path.isdir(phase_dir):
        os.mkdir(phase_dir)

    snap_file = os.path.join(phase_dir, 'state.pth')
    if os.path.isfile(snap_file):
        env.state = torch.load(snap_file)
        yield None
        return

    if phase == G_ML:
        runner = env.pretrain_g
    elif phase == D_ML:
        runner = env.pretrain_d
    elif phase == ADV:
        runner = env.train_adv

    logger = logging.getLogger()
    def _add_file_handler(lvl, prefixes=[]):
        log_path = os.path.join(phase_dir, '_'.join(prefixes + ['log.txt']))
        handler = logging.FileHandler(log_path, mode='w')
        handler.setLevel(lvl)
        logger.addHandler(handler)

    log_prefixes = [opts.log_prefix]*bool(opts.log_prefix)
    _add_file_handler(logging.INFO, log_prefixes)
    _add_file_handler(logging.DEBUG, log_prefixes + ['debug'])

    yield runner

    torch.save(env.state, snap_file)
    logger.removeHandler(file_logger)


if __name__ == '__main__':
    main()
