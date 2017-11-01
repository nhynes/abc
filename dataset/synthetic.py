"""A Dataset that loads the dataset."""
import copy

import torch
import torch.utils.data
from torch.autograd import Variable


class SynthDataset(torch.utils.data.Dataset):
    """Loads synthetic (oracle provided) data."""

    def __init__(self, generator, label,
                 seqlen, batch_size, num_samples, seed):
        super(SynthDataset, self).__init__()
        self.generator = generator  # should be on CPU
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.label = label
        self.seed = seed

    def __getitem__(self, index):
        orig_rand_state = torch.get_rng_state()
        torch.manual_seed(self.seed + index)

        init_toks = Variable(torch.LongTensor(self.batch_size, 1).fill_(1),
                             volatile=True)
        gen_seqs, gen_probs = self.generator.rollout(init_toks, self.seqlen)
        labels = torch.LongTensor(self.batch_size).fill_(self.label)
        gen_seq = torch.cat([init_toks] + gen_seqs, -1).data

        torch.set_rng_state(orig_rand_state)

        return gen_seq, labels

    def __len__(self):
        return self.num_samples // self.batch_size

    def advance(self):
        self.seed += len(self)


def _test_dataset():
    import os
    import sys
    PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, PROJ_ROOT)
    import model

    generator = model.Generator(vocab_size=50,
                                word_emb_dim=32,
                                gen_dim=16)

    batch_size = 24
    seqlen = 20
    num_samples = 1000
    dataset = SynthDataset(generator, 1,
                           batch_size=batch_size,
                           num_samples=num_samples,
                           seqlen=seqlen,
                           seed=42)
    for i in range(len(dataset)):
        gen_seqs, labels = dataset[i]
        assert gen_seqs.size(0) == batch_size
        assert gen_seqs.size(1) == seqlen + 1
        assert len(labels) == batch_size
        assert len(labels) == len(gen_seqs)
    last_gen_seqs, _ = dataset[i]
    assert (gen_seqs == last_gen_seqs).all()
    assert i == num_samples // batch_size - 1


if __name__ == '__main__':
    _test_dataset()
