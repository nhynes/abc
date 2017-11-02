"""A Dataset that loads the output of a generative model."""
import copy

import torch
import torch.utils.data
from torch.autograd import Variable

from common import EXTRA_VOCAB, UNK, BOS, EOS
import common


class GenDataset(torch.utils.data.Dataset):
    """Loads data from a generative model."""

    def __init__(self, generator, label, gen_init_toks,
                 seqlen, num_samples, seed, **unused_kwargs):
        super(GenDataset, self).__init__()

        self.label = label

        orig_rand_state = torch.get_rng_state()
        torch.manual_seed(seed)

        th = torch.cuda if gen_init_toks.is_cuda else torch
        with common.rand_state(th, seed):
            init_toks = gen_init_toks.data.cpu()
            batch_size = gen_init_toks.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            samples = []
            for i in range(num_batches):
                gen_seqs, _ = generator.rollout(gen_init_toks, seqlen)
                samples.append(init_toks)
                samples.extend(map(lambda x: x.data.cpu(), gen_seqs))
            self.samples = torch.cat(samples, -1).view(-1, seqlen + 1)


    def __getitem__(self, index):
        return {
            'toks': self.samples[index],
            'labels': self.label,
        }

    def __len__(self):
        return len(self.samples)


def test_dataset():
    import model

    vocab_size = 50
    batch_size = 32
    label = 0
    num_samples = 1000
    seqlen = 21
    seed = 42

    generator = model.Generator(vocab_size=50, word_emb_dim=32, gen_dim=16)
    gen_init_toks = Variable(torch.LongTensor(batch_size, 1).fill_(1))

    ds = GenDataset(**locals())
    datum = ds[0]
    print(datum)

    assert len(ds) == torch.np.ceil(num_samples / batch_size) * batch_size

    for i in torch.randperm(len(ds)):
        datum = ds[i]
        toks = datum['toks']
        assert (toks >= 0).all() and (toks < vocab_size).all()
