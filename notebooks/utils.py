"""Utilities for analyzing results in notebooks."""

from collections import defaultdict
import re

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FG = r'(\d+\.\d+)'  # Float Group
LOG_RES = {
    'loss': {
        'expr': rf'loss: train={FG} test={FG}',
        'groups': ['train', 'test'],
    },
    'iter': r'\[(\d+)]',
    'nll': rf'nll: {FG}',
    'acc': {
        'expr': rf'acc: o={FG} g={FG}',
        'groups': ['o', 'g'],
    },
    'gnorm': {
        'expr': rf'gnorm: g={FG} d={FG}',
        'groups': ['g', 'd'],
    },
}


def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def ngram_count(seqs, n):
    """Returns a dict of {ngram: counts} for a numpy array of tokens."""
    rw = rolling_window_lastaxis(seqs, n).reshape(-1, n)
    ngrams, counts = np.unique(rw, axis=0, return_counts=True)
    counts = counts / counts.sum()
    return {tuple(ngram): count for ngram, count in zip(ngrams, counts)}


def sample_gen(env, num_samps=10000, gen='g', temperature=1,
               return_probs=False):
    """Samples from a generative model."""
    samps = []
    probs = []
    gen = getattr(env, gen)
    num_batches = (num_samps + len(env.ro_init_toks)) // len(env.ro_init_toks)
    for _ in range(num_batches):
        ro, ro_probs = gen.rollout(env.ro_init_toks, 20,
                                   temperature=temperature)
        samps.append(torch.cat(ro, -1).data.cpu())
        if return_probs:
            probs.append(torch.stack(ro_probs).data.cpu())

    samps = torch.cat(samps, 0).numpy()
    if return_probs:
        return samps, torch.cat(probs, 0).numpy()
    return samps


def load_log(log_path):
    """Loads a log file."""

    if not LOG_RES.get('_compiled'):
        for stat, spec in LOG_RES.items():
            if isinstance(spec, dict):
                spec['expr'] = re.compile(spec['expr'])
            else:
                LOG_RES[stat] = re.compile(spec)
        LOG_RES['_compiled'] = True

    cols = defaultdict(list)
    with open(log_path) as f_log:
        for line in f_log:
            if line.startswith('#'):
                continue
            for stat, spec in sorted(LOG_RES.items()):
                if stat.startswith('_'):
                    continue

                if isinstance(spec, dict):
                    matcher = spec['expr']
                    colnames = [f'{stat}_{substat}'
                                for substat in spec['groups']]
                else:
                    matcher = spec
                    colnames = [stat]

                match = matcher.search(line)
                if not match:
                    continue

                for colname, val in zip(colnames, match.groups()):
                    cols[colname].append(val)
    log_df = pd.DataFrame.from_dict(cols)
    log_df = log_df.set_index('iter').astype(float)
    return log_df


def do_plot(get_data, logs, filt=None, baseline=None):
    """Plots data from several logs.

    Args:
        get_data: a function (log_name, log_data) -> plot_data
    """
    for exp_name, log in logs.items():
        if filt and (exp_name != baseline and not filt in exp_name):
            continue
        get_data(exp_name, log).plot(label=exp_name)
    plt.legend()


def plot_ts(col, *args, **kwargs):
    """Plots a column from logs as a time series."""
    do_plot(lambda name, log: log[col], *args, **kwargs)
