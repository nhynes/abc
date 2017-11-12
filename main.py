"""Main training script."""

from contextlib import contextmanager
import argparse
import os
import pickle
import logging

import torch

import common
from common import PHASES, G_ML, D_ML, ADV
from common import RUN_DIR, STATE_FILE, OPTS_FILE, LOG_FILE
import environ


def main():
    """Trains the model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=environ.ENVS, default=environ.SYNTH)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prefix')
    init_opts, remaining_opts = parser.parse_known_args()

    opts_file = os.path.join(RUN_DIR, OPTS_FILE)
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
                phase_runner()  # pylint: disable=not-callable


@contextmanager
def _phase(env, phase, opts):
    phase_dir = os.path.join(RUN_DIR, phase)
    if not os.path.isdir(phase_dir):
        os.mkdir(phase_dir)

    prefixes = [opts.prefix]*bool(opts.prefix)
    def _prefix(suffixes):
        suffixes = suffixes if isinstance(suffixes, list) else [suffixes]
        return '_'.join(prefixes + suffixes)

    snap_file = os.path.join(phase_dir, STATE_FILE)
    prefix_snap_file = os.path.join(phase_dir, _prefix(STATE_FILE))
    if os.path.isfile(prefix_snap_file):
        snap_file = prefix_snap_file

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
    def _add_file_handler(lvl, log_prefix=None):
        suffixes = [log_prefix]*bool(log_prefix) + [LOG_FILE]
        log_path = os.path.join(phase_dir, _prefix(suffixes))
        handler = logging.FileHandler(log_path, mode='w')
        handler.setLevel(lvl)
        logger.addHandler(handler)
        return handler

    file_handlers = [
        _add_file_handler(logging.INFO),
        _add_file_handler(logging.DEBUG, 'debug'),
    ]

    yield runner

    torch.save(env.state, snap_file)
    for handler in file_handlers:
        logger.removeHandler(handler)


if __name__ == '__main__':
    main()
