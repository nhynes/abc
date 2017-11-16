"""Formats a Wikipedia dump dataset."""

from collections import Counter
import argparse
import gzip
import json
import os
import pickle
import random
import sys

from tqdm import tqdm
import spacy

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJ_ROOT)

import common

DATA_DIR = os.path.join(PROJ_ROOT, 'data')
CACHE_DIR = os.path.join(DATA_DIR, '_cache', 'wiki')
OUT_DIR = os.path.join(DATA_DIR, 'wiki')
WIKI_PATH = os.path.join(CACHE_DIR, 'simplewiki-20171103.json.gz')


def _unpickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def _load():
    paragraphs = []
    with gzip.open(WIKI_PATH, 'rt') as f_wiki:
        for line in f_wiki:
            text = json.loads(line)['text']
            paragraphs.extend(filter(bool, text.split('\n')))
    return paragraphs


def _tokenize(paragraphs):
    nlp = spacy.load('en')
    tok_sents = []
    for i, para in enumerate(nlp.pipe(tqdm(paragraphs, desc='tokenize', leave=False))):
        for sent in para.sents:
            if not sent[-1].is_punct:
                continue
            tok_sents.append(tuple(tok.text for tok in sent if tok))
    return tok_sents


def _concatenate(tok_sents):
    return tuple(map(' '.join, tok_sents))


def _run_pipeline(pipeline, cache_dir):
    for i, stage in enumerate(pipeline):
        cache_path = os.path.join(cache_dir, stage.__name__[1:]) + '.pkl'
        if os.path.isfile(cache_path):
            output = _unpickle(cache_path)
        else:
            output = stage(output) if i > 0 else stage()
            _pickle(output, cache_path)
    return output


def main():
    """Runs a pipeline that formats the Wikipedia dump dataset."""
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--cased', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    for d in (CACHE_DIR, OUT_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)

    sents = _run_pipeline([_load, _tokenize, _concatenate], CACHE_DIR)

    random.seed(args.seed)
    if not args.cased:
        sents = list(map(str.lower, sents))
    random.shuffle(sents)

    n_val = n_test = int(len(sents) * (1 - args.train_frac) / 2)
    n_train = len(sents) - n_val - n_test
    part_bounds = [None, n_train, n_train + n_val, None]

    for i, part in enumerate(('train', 'val', 'test')):
        part_sents = sents[slice(*part_bounds[i:i+2])]

        if part == 'train':
            vocab = Counter(tok for q in part_sents for tok in q.split(' '))
            _pickle(common.Vocab(vocab.most_common()),
                    os.path.join(OUT_DIR, 'vocab.pkl'))

        _pickle(part_sents, os.path.join(OUT_DIR, part + '.pkl'))


if __name__ == '__main__':
    main()
