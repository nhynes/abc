"""Utility functions for training neural networks."""

import os
import pickle
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


RUN_DIR = 'run'
OPTS_FILE = 'opts.txt'

PHASES = ('g_ml', 'd_ml', 'adv')
G_ML, D_ML, ADV = PHASES
LABEL_G, LABEL_O = 0, 1

EXTRA_VOCAB = ['PAD', 'UNK', '<s>', '</s>']
PAD, UNK, BOS, EOS = EXTRA_VOCAB

def resume(opts):
    """Returns a model, its inputs, and optimizer loaded from a given epoch."""
    opts_path = OPTS_PATH
    with open(opts_path, 'rb') as f_opts:
        init_opts = pickle.load(f_opts)

    for opt_name, opt_val in vars(init_opts).items():
        if opt_name not in OVERWRITE_OPTS:
            setattr(opts, opt_name, opt_val)

    i = 1
    while os.path.isfile(opts_path):
        opts_path = f'run/opts_{i}.pkl'
        i += 1
    with open(opts_path, 'wb') as f_new_opts:
        pickle.dump(opts, f_new_opts)

    net, inputs = create_model(opts)
    net.load_state_dict(torch.load(MSNAP_PATH.format(opts.resume_epoch)))

    optimizer = create_optimizer(opts, net)
    optimizer.load_state_dict(torch.load(OSNAP_PATH.format(opts.resume_epoch)))

    return net, inputs, optimizer


def create_datasets(opts, partitions=('train', 'val')):
    """Returns a mapping from the provided partitions to `Dataset`s."""
    import dataset
    part_datasets = {part: dataset.create(part=part, **vars(opts))
                     for part in partitions}
    return part_datasets


def create_loaders(opts, datasets):
    """Returns loaders for a mapping from partition to `Dataset`."""
    loader_opts = {'batch_size': opts.batch_size, 'pin_memory': True,
                   'num_workers': opts.nworkers}
    return {
        f'{part}_loader':
        torch.utils.data.DataLoader(ds, shuffle=part == 'train', **loader_opts)
        for part, ds in datasets.items()}


def create_generator(opts):
    """Creates a token generator model."""
    import model
    return model.Generator(word_emb_dim=opts.g_word_emb_dim, **vars(opts))


def create_discriminator(opts):
    """Creates a token discriminator model."""
    import model
    return model.Discriminator(word_emb_dim=opts.d_word_emb_dim, **vars(opts))


def create_optimizer(opts, model):
    """Returns optimizers for the model."""
    class GANOptimizer(object):
        def __init__(self, model, opts):
            self.g = optim.Adam(model.g.parameters(), lr=opts.lr_g)
            self.d = optim.Adam(model.d.parameters(), lr=opts.lr_d)
    return GANOptimizer(model, opts)


def state2cpu(state):
    """Moves `Tensor`s in state dict to the CPU."""
    if isinstance(state, dict):
        return type(state)({k: state2cpu(v) for k, v in state.items()})
    elif torch.is_tensor(state):
        return state.cpu()


def ship_batch(batch, volatile=False):
    """Ships a batch of data to the GPU."""
    toks, labels = batch
    toks = Variable(toks.view(-1, toks.size(-1)), volatile=volatile).cuda()
    labels = Variable(labels.view(-1), volatile=volatile).cuda()
    return toks, labels


def copy_inputs(cpu_inputs, inputs, volatile=False):
    """Copies Tensors into Variables."""
    for input_name, inp in inputs.items():
        if input_name not in cpu_inputs:
            continue
        cpu_tensor = cpu_inputs[input_name]
        inp.data.resize_(cpu_tensor.size()).copy_(cpu_tensor)
        if isinstance(inp, Variable):
            inp.volatile = volatile


def unpickle(path_pkl):
    """Loads the contents of a pickle file."""
    with open(path_pkl, 'rb') as f_pkl:
        return pickle.load(f_pkl)


def load_txt(path_txt):
    """Loads a text file."""
    with open(path_txt) as f_txt:
        return [line.rstrip() for line in f_txt]


class Vocab(object):
    """Represents a token2index and index2token map."""

    def __init__(self, tok_counts, unk_tok=None):
        """Constructs a Vocab ADT."""
        self.tok_counts = tok_counts
        self.w2i = {w: i for i, (w, _) in enumerate(self.tok_counts)}

        self.unk_tok = unk_tok
        if unk_tok is not None:
            assert unk_tok in self.w2i
            self.unk_idx = self.w2i[unk_tok]

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < len(self):
                return self.tok_counts[index][0]
            elif self.unk_tok:
                return self.unk_idx
            else:
                raise IndexError(f'No token in position {index}!')
        elif isinstance(index, str):
            if index in self.w2i:
                return self.w2i[index]
            elif self.unk_tok:
                return self.unk_idx
            else:
                raise KeyError(f'{index} not in vocab!')
        else:
            raise ValueError('Index to Vocab must be string or int.')

    def add_extra_vocab(self, extra_vocab):
        """Returns a new Vocab with extra tokens prepended."""
        extra_tok_counts = [(w, float('inf')) for w in extra_vocab]
        return Vocab(extra_tok_counts + self.tok_counts,
                     unk_tok=self.unk_tok)

    def set_unk_tok(self, unk_tok):
        """Sets the token/index to return when looking up an OOV token."""
        return Vocab(self.tok_counts, unk_tok=unk_tok)

    def truncate(self, size):
        """Returns a new Vocab containing the top `size` tokens."""
        return Vocab(self.tok_counts[:size], unk_tok=self.unk_tok)

    def __len__(self):
        return len(self.tok_counts)


def compute_oracle_nll(oracle, toks, return_probs=False):
    """
    oracle: a Generator
    toks: [N]*T
    """
    init_toks = Variable(
        torch.LongTensor(toks[0].size(0), 1).fill_(1).type_as(toks[0].data))
    gen_probs, _ = oracle(torch.cat([init_toks] + toks, 1))
    gen_probs = gen_probs[:-1]
    flat_gen_probs = gen_probs.view(-1, gen_probs.size(-1))
    flat_toks = torch.cat(toks).squeeze(1)
    nll = nnf.nll_loss(flat_gen_probs, flat_toks).data[0]
    if return_probs:
        return nll, gen_probs
    return nll


def compute_acc(probs, labels):
    """Computes the accuracy given prob and labels Variables."""
    preds = probs.max(1)[1]
    return ((preds == labels).sum().data[0] / len(labels))


@contextmanager
def rand_state(th, rand_state):
    orig_rand_state = th.get_rng_state()
    if isinstance(rand_state, int):
        th.manual_seed(rand_state)  # this is a slow operation!
        rand_state = th.get_rng_state()
    th.set_rng_state(rand_state)
    yield rand_state
    th.set_rng_state(orig_rand_state)
