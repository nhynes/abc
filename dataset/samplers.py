"Dataset samplers."

import sys

import torch.utils.data


class InfiniteRandomSampler(torch.utils.data.sampler.RandomSampler):
    """A RandomSampler that cycles forever."""
    def __iter__(self):
        index_iter = iter(())
        while True:
            try:
                yield next(index_iter)
            except StopIteration:
                index_iter = super(InfiniteRandomSampler, self).__iter__()

    def __len__(self):
        return sys.maxsize


class ReplayBufferSampler(torch.utils.data.sampler.Sampler):
    """A Sampler that uniforly samples batches of indices forever."""
    def __init__(self, replay_buffer, batch_size):
        super(ReplayBufferSampler, self).__init__(replay_buffer)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            rbuf_len = len(self.replay_buffer)
            num_samps = min(rbuf_len, self.batch_size)
            yield torch.LongTensor(num_samps).random_(rbuf_len)

    def __len__(self):
        return sys.maxsize


def test_inf_rand_sampler():
    """Tests the InfiniteRandomSampler."""
    import itertools

    sampler = InfiniteRandomSampler(torch.randn(4))
    inds = list(itertools.islice(iter(sampler), 8))

    assert len(sampler) > 1e10
    assert len(inds) == 8


def test_replay_buffer_sampler():
    """Tests the ReplayBufferSampler."""
    t = torch.randn(2)

    sampler_it = iter(ReplayBufferSampler(t, 4))

    assert len(next(sampler_it)) == 2

    t.resize_(8)
    assert len(next(sampler_it)) == 4
