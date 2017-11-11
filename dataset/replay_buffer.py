"""A Dataset that acts as a replay buffer."""
from collections import deque

import torch
import torch.utils.data
from torch.autograd import Variable


class ReplayBuffer(torch.utils.data.ConcatDataset):
    """Loads data from a replay buffer."""

    def __init__(self, max_history, **unused_kwargs):
        # pylint: disable=super-init-not-called
        self.datasets = deque(maxlen=max_history)
        self.cummulative_sizes = []  # [sic]

    def add_samples(self, samples):
        """Adds a batch of samples to the replay buffer."""
        if samples.is_cuda:
            samples = samples.cpu()
        if isinstance(samples, Variable):
            samples = samples.data

        if self.datasets:
            assert self.datasets[-1].size()[1:] == samples.size()[1:]
        self.datasets.append(samples)
        self.cummulative_sizes = self.cumsum(self.datasets)

    def get_samples(self, num_samples):
        """Returns a batch of random samples from the replay buffer."""
        num_samples = min(len(self), num_samples)
        samps = self.datasets[0].new(num_samples, *self.datasets[0].size()[1:])
        for i, idx in enumerate(torch.randperm(len(self))[:num_samples]):
            samps[i] = self[idx]
        return samps


def test_replay_buffer():
    """Tests the replay buffer."""

    rbuf = ReplayBuffer(label=1, max_history=2)

    rbuf.add_samples(torch.zeros(2, 3))
    rbuf.add_samples(torch.ones(3, 3))

    assert len(rbuf) == 5

    assert (rbuf[0] == 0).all()
    assert (rbuf[len(rbuf) - 1] == 1).all()

    rbuf.add_samples(Variable(torch.ones(4, 3)*2))
    assert len(rbuf) == 7
    assert (rbuf[0] == 1).all()
    assert (rbuf[len(rbuf) - 1] == 2).all()
