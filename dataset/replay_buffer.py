"""A Dataset that acts as a replay buffer."""
from collections import deque

import torch
import torch.utils.data
from torch.autograd import Variable


class ReplayBuffer(torch.utils.data.ConcatDataset):
    """Loads data from a replay buffer."""

    def __init__(self, max_history, label, **unused_kwargs):
        # pylint: disable=super-init-not-called
        self.datasets = deque(maxlen=max_history)
        self.cummulative_sizes = [0]  # [sic]
        self.label = label

    def add_samples(self, samples):
        """Adds a batch of samples to the replay buffer."""
        if samples.is_cuda:
            samples = samples.cpu()
        if isinstance(samples, Variable):
            samples = samples.data

        dataset = torch.utils.data.TensorDataset(
            samples, torch.LongTensor(len(samples)).fill_(self.label))
        self.datasets.append(dataset)
        self.cummulative_sizes = self.cumsum(self.datasets)


def test_replay_buffer():
    """Tests the replay buffer."""

    rbuf = ReplayBuffer(label=-2, max_history=2)

    rbuf.add_samples(torch.zeros(2, 3))
    rbuf.add_samples(torch.ones(3, 3))

    assert len(rbuf) == 5

    assert (rbuf[0][0] == 0).all()
    assert rbuf[0][1] == -2
    assert (rbuf[len(rbuf) - 1][0] == 1).all()

    rbuf.add_samples(Variable(torch.ones(4, 3)*2))
    assert len(rbuf) == 7
    assert (rbuf[0][0] == 1).all()
    assert (rbuf[len(rbuf) - 1][0] == 2).all()
