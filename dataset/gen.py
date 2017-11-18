"""A Dataset that loads the output of a generative model."""
import torch
import torch.utils.data
from torch.autograd import Variable

import common


class GenDataset(torch.utils.data.Dataset):
    """Loads data from a generative model."""

    def __init__(self, generator, label, seqlen, num_samples,
                 gen_init_toks, seed, eos_idx=None, **unused_kwargs):
        super(GenDataset, self).__init__()

        self.label = label

        th = torch.cuda if gen_init_toks.is_cuda else torch
        with common.rand_state(th, seed):
            init_toks = gen_init_toks.data.cpu()
            batch_size = gen_init_toks.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            samples = []
            for _ in range(num_batches):
                gen_seqs, _ = generator.rollout(gen_init_toks, seqlen)
                samps = torch.cat([gen_init_toks] + gen_seqs, -1).data
                if eos_idx:
                    # 1. create a mask of ones up until the first eos token
                    mask = (samps != eos_idx).cumprod(-1)
                    # 2. create a mask for the first the eos token
                    # this method requires that the init tok exists
                    eos_pos = (samps == eos_idx)[:, 1:] * mask[:, :-1]
                    # 3. zero out everything including+after the first eos tok
                    samps.masked_fill_(1 - mask, 0)
                    # 4. put back the eos tok
                    samps[:, 1:].masked_fill_(eos_pos, eos_idx)
                samples.append(samps.cpu())
            self.samples = torch.cat(samples)

    def __getitem__(self, index):
        label = self.label
        if not isinstance(index, int):
            label = torch.LongTensor(len(index)).fill_(self.label)
        return self.samples[index], label

    def __len__(self):
        return len(self.samples)


def test_dataset():
    """Tests the Dataset."""
    import model

    # pylint: disable=unused-variable
    vocab_size = 50
    batch_size = 32
    label = 0
    num_samples = 1000
    seqlen = 21
    seed = 42

    generator = model.generator.RNNGenerator(
        vocab_size=50, tok_emb_dim=32, rnn_dim=16, num_layers=1)
    gen_init_toks = Variable(torch.LongTensor(batch_size, 1).fill_(1))

    ds = GenDataset(**locals())
    toks, labels = ds[0]
    print(toks)
    print(labels)

    assert len(ds) == torch.np.ceil(num_samples / batch_size) * batch_size

    for i in torch.randperm(len(ds)):
        toks, labels = ds[i]
        assert (toks >= 0).all() and (toks < vocab_size).all()

    batch_toks, batch_labels = ds[torch.randperm(batch_size)]
    assert len(batch_toks) == batch_size
    assert len(batch_labels) == batch_size

def test_dataset_mask():
    """Tests the Dataset."""
    import model

    # pylint: disable=unused-variable
    vocab_size = 50
    batch_size = 5
    label = 0
    num_samples = batch_size
    seqlen = 21
    seed = 42
    eos_idx = 2

    gen_samps = torch.LongTensor([
        [1, 1, 2, 1, 2, 1],
        [1, 1, 1, 2, 1, 1],
        [2, 1, 1, 2, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 2],
    ])

    expected_samps = torch.LongTensor([
        [-1, 1, 1, 2, 0, 0, 0],
        [-1, 1, 1, 1, 2, 0, 0],
        [-1, 2, 0, 0, 0, 0, 0],
        [-1, 1, 1, 1, 1, 1, 1],
        [-1, 1, 1, 1, 1, 1, 2],
    ])

    class MockGenerator(object):
        @staticmethod
        def rollout(*args, **kwargs):
            return list(Variable(gen_samps).split(1, dim=1)), None

    generator = MockGenerator()
    gen_init_toks = Variable(torch.LongTensor(batch_size, 1).fill_(-1))

    ds = GenDataset(**locals())
    assert (ds.samples == expected_samps).all()
