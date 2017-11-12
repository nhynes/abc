import torch
from torch import nn


class DontTrain(nn.Module):
    """Wraps a module to detach its output."""

    def __init__(self, mod):
        super(DontTrain, self).__init__()
        self.mod = mod

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs).detach()


def load_w2v_file(w2v_file):
    """Loads a textual word2vec file in which the tokens are numeric."""
    return torch.np.loadtxt(w2v_file)[:, 1:]
