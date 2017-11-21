"""Prunes token vectors to the subset actually present in a vocabulary."""

import argparse
import os
import pickle
import sys

import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_ROOT, 'data')

sys.path.insert(0, PROJ_ROOT)
from common import EXTRA_VOCAB, UNK, BOS, EOS
import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab',
                        default='../data/qa/vocab.pkl')
    parser.add_argument('--word-vecs', default='../data/glove.840B.300d')
    parser.add_argument('--vocab-size', default=20000, type=int)
    args = parser.parse_args()

    tok_vocab = EXTRA_VOCAB
    tok_vocab.extend(tok for tok, _ in common.unpickle(args.vocab).tok_counts)
    tok_vocab = tok_vocab[:args.vocab_size]

    vecs_vocab = common.unpickle(f'{args.word_vecs}_vocab.pkl')
    vecs = np.load(f'{args.word_vecs}.npy')
    vecs_w2i = {w: i for i, w in enumerate(vecs_vocab)}
    for i, w in enumerate(vecs_vocab):
        if not w.lower() in vecs_w2i:
            vecs_w2i[w.lower()] = i

    filt_vecs = np.random.randn(len(tok_vocab), vecs.shape[1])
    n = 0
    for i, w in enumerate(tok_vocab):
        if w in vecs_w2i:
            filt_vecs[i] = vecs[vecs_w2i[w]]
            n += 1
    print(f'found {n} words')

    np.save(os.path.join(DATA_DIR, 'tok_vecs_pruned.npy'), filt_vecs)


if __name__ == '__main__':
    main()
