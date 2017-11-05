"""Utilities for analyzing results in notebooks."""

import torch
import numpy as np

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
    rw = rolling_window_lastaxis(seqs, n).reshape(-1, n)
    ngrams, counts = np.unique(rw, axis=0, return_counts=True)
    counts = counts / counts.sum()
    return {tuple(ngram): count for ngram, count in zip(ngrams, counts)}

def sample_gen(env, num_samps=10000):
    samps = []
    num_batches = (num_samps + len(env.init_toks)) // len(env.init_toks)
    for _ in range(num_batches):
        ro, ro_probs = env.g.rollout(env.init_toks, 20)
        samps.append(torch.cat([r.cpu() for r in ro], -1))
    return torch.cat(samps, 0).data.numpy()
