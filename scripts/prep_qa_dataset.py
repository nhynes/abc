"""Formats the Yahoo Answers dataset."""

from collections import Counter
import argparse
import os
import random
import re
import sys
import pickle

from tqdm import tqdm
from lxml import etree
import spacy

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJ_ROOT)

import common

DATA_DIR = os.path.join(PROJ_ROOT, 'data')
CACHE_DIR = os.path.join(DATA_DIR, '_cache', 'qa')
OUT_DIR = os.path.join(DATA_DIR, 'qa')
QS_PATH = os.path.join(DATA_DIR, 'FullOct2007.xml.part{}.gz')


QUESTION_WORDS = {
    'any', 'are', 'can', 'could', 'did', 'do', 'does', 'has', 'have', 'how',
    'is', 'must', 'should', 'was', 'what', 'when', 'where', 'which', 'who',
    'whom', 'whose', 'why', 'will', 'would'}


def _unpickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def _load():
    qs = []
    for part in range(1, 3):
        try:
            data_path = QS_PATH.format(part)
            for _, subj in etree.iterparse(data_path, tag='subject'):
                qs.append(subj.text)
        except etree.XMLSyntaxError:
            pass
    return qs


def _tokenize(qs):
    nlp = spacy.load('en', disable=['ner'])
    qtoks = []
    for q in nlp.pipe(tqdm(qs, desc='tokenize', leave=False)):
        for sent in q.sents:
            sent_toks = [tok.text for tok in sent if not tok.is_space]
            if sent_toks:
                qtoks.append(sent_toks)
    return qtoks


def _filter(qs):
    good_qs = []
    for q in qs:
        if not q[0] in QUESTION_WORDS or q[-1] != '?':
            continue
        qcat = ' '.join(q)
        qcat = re.sub('( [?!.]+)+ \?$', ' ?', qcat)
        good_qs.append(qcat)
    return good_qs


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
    """Runs a pipeline that formats the Yahoo Answers dataset."""
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

    qs = _run_pipeline([_load, _tokenize, _filter], CACHE_DIR)

    random.seed(args.seed)
    if not args.cased:
        qs = [q.lower() for q in qs]
    random.shuffle(qs)

    n_val = n_test = int(len(qs) * (1 - args.train_frac) / 2)
    n_train = len(qs) - n_val - n_test
    part_bounds = [None, n_train, n_train + n_val, None]

    for i, part in enumerate(('train', 'val', 'test')):
        part_qs = qs[slice(*part_bounds[i:i+2])]

        if part == 'train':
            vocab = Counter(tok for q in part_qs for tok in q.split(' '))
            _pickle(common.Vocab(vocab.most_common()),
                    os.path.join(OUT_DIR, 'vocab.pkl'))

        _pickle(part_qs, os.path.join(OUT_DIR, part + '.pkl'))


if __name__ == '__main__':
    main()
